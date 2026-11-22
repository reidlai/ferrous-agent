//! Streamable HTTP MCP Client & Declarative Registry for `ferrous-agent` (Epic 20, AD-33).
//!
//! Connects `ferrous-agent` (`rig-core` 0.46) to independently deployed Docker/Kubernetes MCP
//! server `Deployment`s strictly over MCP Streamable HTTP — keeping `ferrous-agent` `FROM scratch`
//! (AD-24) with zero `stdio` child processes, enforcing AD-8 `allowed_tools` allowlists,
//! wrapping all inputs/outputs in [`ContextSanitizer`] (AD-18), propagating Kanidm M2M JWT
//! and W3C `traceparent` headers (AD-23), and degrading gracefully on partial server outages.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tracing::{info, warn};

use crate::sanitizer::ContextSanitizer;

/// Forbidden financial mutation tool names blocked by AD-8 Structural HITL Financial Safety.
pub const FORBIDDEN_MUTATION_TOOLS: &[&str] = &[
    "issue_credit",
    "issue-credit",
    "issue_courtesy_credit",
    "cancel_order",
    "cancel-order",
    "cancel",
];

/// Supported MCP transports. Production `FROM scratch` containers permit only `StreamableHttp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    StreamableHttp,
    Stdio,
}

/// Configuration for a single remote MCP server in `mcp-servers.toml` (K8s `ConfigMap`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub url: String,
    #[serde(default = "default_transport")]
    pub transport: McpTransport,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub auth_env: Option<String>,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
}

fn default_transport() -> McpTransport {
    McpTransport::StreamableHttp
}

fn default_timeout_ms() -> u64 {
    5000
}

/// Top-level declarative MCP registry loaded from `mcp-servers.toml` (`MCP_SERVERS_CONFIG`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpRegistryConfig {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

/// Errors arising from MCP registry validation, transport, or AD-8/AD-18 guardrails.
#[derive(Debug, Error)]
pub enum McpClientError {
    #[error("MCP configuration error: {0}")]
    Config(String),

    #[error("AD-24 violation: stdio MCP transport is forbidden in production scratch containers (server '{0}')")]
    ForbiddenTransport(String),

    #[error("AD-8 violation: mutation tool '{tool}' is forbidden in allowed_tools for server '{server}'")]
    ForbiddenMutationTool { server: String, tool: String },

    #[error("Tool '{tool}' is not in allowed_tools for server '{server}'")]
    ToolNotAllowed { server: String, tool: String },

    #[error("AD-18 ContextSanitizer blocked MCP payload: {0}")]
    SanitizerBlocked(String),

    #[error("MCP HTTP transport error for '{server}': {message}")]
    Transport { server: String, message: String },

    #[error("MCP protocol error for '{server}': {message}")]
    Protocol { server: String, message: String },
}

impl McpRegistryConfig {
    /// Parses and validates an `mcp-servers.toml` string, enforcing AD-24 (Streamable HTTP only)
    /// and AD-8 (no direct financial mutation tools in `allowed_tools`).
    pub fn from_toml_str(raw: &str) -> Result<Self, McpClientError> {
        let cfg: Self =
            toml::from_str(raw).map_err(|e| McpClientError::Config(format!("invalid TOML: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Loads and validates `mcp-servers.toml` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, McpClientError> {
        let raw = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            McpClientError::Config(format!(
                "failed to read '{}': {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_toml_str(&raw)
    }

    /// Validates every server entry against AD-24 and AD-8 invariants.
    pub fn validate(&self) -> Result<(), McpClientError> {
        for srv in &self.servers {
            if srv.name.trim().is_empty() {
                return Err(McpClientError::Config("server name cannot be blank".into()));
            }
            if srv.transport != McpTransport::StreamableHttp {
                return Err(McpClientError::ForbiddenTransport(srv.name.clone()));
            }
            if !srv.url.starts_with("http://") && !srv.url.starts_with("https://") {
                return Err(McpClientError::Config(format!(
                    "server '{}' url must use http:// or https://, got '{}'",
                    srv.name, srv.url
                )));
            }
            for tool in &srv.allowed_tools {
                let lower = tool.trim().to_ascii_lowercase();
                if FORBIDDEN_MUTATION_TOOLS.contains(&lower.as_str()) {
                    return Err(McpClientError::ForbiddenMutationTool {
                        server: srv.name.clone(),
                        tool: tool.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Metadata for a remote MCP tool discovered via `tools/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub server_name: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_input_schema")]
    pub input_schema: Value,
}

fn default_input_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

/// Streamable HTTP MCP client for a single containerized / Kubernetes MCP server.
#[derive(Debug, Clone)]
pub struct McpStreamableHttpClient {
    config: McpServerConfig,
    http: reqwest::Client,
    bearer_token: Option<String>,
    allowed_set: HashSet<String>,
    sanitizer: ContextSanitizer,
}

impl McpStreamableHttpClient {
    /// Constructs a new Streamable HTTP MCP client, resolving optional bearer token from `auth_env`.
    pub fn new(config: McpServerConfig) -> Result<Self, McpClientError> {
        let registry = McpRegistryConfig {
            servers: vec![config.clone()],
        };
        registry.validate()?;

        let bearer_token = config
            .auth_env
            .as_deref()
            .and_then(|env_name| std::env::var(env_name).ok())
            .filter(|v| !v.trim().is_empty());

        Self::with_token(config, bearer_token)
    }

    /// Constructs a client with an explicit bearer token (useful for testing and M2M injection).
    pub fn with_token(
        config: McpServerConfig,
        bearer_token: Option<String>,
    ) -> Result<Self, McpClientError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms.max(100)))
            .build()
            .map_err(|e| McpClientError::Transport {
                server: config.name.clone(),
                message: e.to_string(),
            })?;

        let allowed_set = config.allowed_tools.iter().cloned().collect();

        Ok(Self {
            config,
            http,
            bearer_token,
            allowed_set,
            sanitizer: ContextSanitizer::new(),
        })
    }

    pub fn server_name(&self) -> &str {
        &self.config.name
    }

    fn build_headers(&self, traceparent: Option<&str>) -> Result<HeaderMap, McpClientError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );

        if let Some(ref token) = self.bearer_token {
            let val = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| {
                McpClientError::Config(format!("invalid bearer token header: {e}"))
            })?;
            headers.insert(AUTHORIZATION, val);
        }

        let tp = traceparent.unwrap_or("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        let tp_val = HeaderValue::from_str(tp)
            .map_err(|e| McpClientError::Config(format!("invalid traceparent header: {e}")))?;
        headers.insert("traceparent", tp_val);

        Ok(headers)
    }

    async fn rpc_call(
        &self,
        method: &str,
        params: Value,
        traceparent: Option<&str>,
    ) -> Result<Value, McpClientError> {
        let headers = self.build_headers(traceparent)?;
        let body = json!({
            "jsonrpc": "2.0",
            "id": uuid::Uuid::now_v7().to_string(),
            "method": method,
            "params": params,
        });

        let resp = self
            .http
            .post(&self.config.url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpClientError::Transport {
                server: self.config.name.clone(),
                message: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(McpClientError::Transport {
                server: self.config.name.clone(),
                message: format!("HTTP {}", resp.status()),
            });
        }

        let envelope: Value = resp.json().await.map_err(|e| McpClientError::Protocol {
            server: self.config.name.clone(),
            message: format!("invalid JSON-RPC response: {e}"),
        })?;

        if let Some(err) = envelope.get("error") {
            return Err(McpClientError::Protocol {
                server: self.config.name.clone(),
                message: err.to_string(),
            });
        }

        envelope
            .get("result")
            .cloned()
            .ok_or_else(|| McpClientError::Protocol {
                server: self.config.name.clone(),
                message: "missing 'result' in JSON-RPC response".into(),
            })
    }

    /// Discovers tools via MCP `tools/list` and filters strictly to `allowed_tools` (AD-8).
    pub async fn list_allowed_tools(
        &self,
        traceparent: Option<&str>,
    ) -> Result<Vec<McpToolDescriptor>, McpClientError> {
        let result = self.rpc_call("tools/list", json!({}), traceparent).await?;
        let raw_tools = result
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut filtered = Vec::new();
        for item in raw_tools {
            let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                continue;
            };
            if !self.allowed_set.contains(name) {
                warn!(
                    server = %self.config.name,
                    tool = %name,
                    "Excluding non-allowlisted MCP tool (AD-8)"
                );
                continue;
            }
            let description = item
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let input_schema = item
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(default_input_schema);

            filtered.push(McpToolDescriptor {
                server_name: self.config.name.clone(),
                name: name.to_string(),
                description,
                input_schema,
            });
        }

        Ok(filtered)
    }

    /// Invokes an allowlisted MCP tool via `tools/call`, sanitizing both arguments and response
    /// through [`ContextSanitizer`] (AD-18).
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        traceparent: Option<&str>,
    ) -> Result<String, McpClientError> {
        if !self.allowed_set.contains(tool_name) {
            return Err(McpClientError::ToolNotAllowed {
                server: self.config.name.clone(),
                tool: tool_name.to_string(),
            });
        }

        // AD-18 Ingress firewall & PII masking on outbound tool arguments
        let args_str = serde_json::to_string(&arguments).map_err(|e| McpClientError::Protocol {
            server: self.config.name.clone(),
            message: e.to_string(),
        })?;
        let (sanitized_args_str, _) = self
            .sanitizer
            .sanitize_for_llm(&args_str)
            .map_err(|alert| McpClientError::SanitizerBlocked(format!("{alert:?}")))?;
        let sanitized_args: Value =
            serde_json::from_str(&sanitized_args_str).unwrap_or(arguments);

        let result = self
            .rpc_call(
                "tools/call",
                json!({
                    "name": tool_name,
                    "arguments": sanitized_args,
                }),
                traceparent,
            )
            .await?;

        // Extract text content from standard MCP CallToolResult or fallback to JSON string
        let raw_output = result
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| result.to_string());

        // AD-18 Egress PII masking & prompt-injection check on MCP tool response
        let (sanitized_output, _) = self
            .sanitizer
            .sanitize_for_llm(&raw_output)
            .map_err(|alert| McpClientError::SanitizerBlocked(format!("{alert:?}")))?;

        Ok(sanitized_output)
    }
}

/// Resilient pool of containerized MCP server clients (CAP-5).
///
/// Unreachable MCP servers are logged and skipped so healthy servers continue serving tools.
#[derive(Debug, Default)]
pub struct McpClientPool {
    clients: HashMap<String, Arc<McpStreamableHttpClient>>,
    tools: HashMap<String, (McpToolDescriptor, Arc<McpStreamableHttpClient>)>,
    failed_servers: Vec<(String, String)>,
}

impl McpClientPool {
    /// Connects to all servers in `registry`, discovering allowlisted tools with partial-outage resilience.
    pub async fn connect_registry(
        registry: &McpRegistryConfig,
        explicit_tokens: &HashMap<String, String>,
        traceparent: Option<&str>,
    ) -> Result<Self, McpClientError> {
        registry.validate()?;
        let mut pool = Self::default();

        for srv in &registry.servers {
            let token = explicit_tokens.get(&srv.name).cloned().or_else(|| {
                srv.auth_env
                    .as_deref()
                    .and_then(|k| std::env::var(k).ok())
                    .filter(|v| !v.trim().is_empty())
            });

            let client = match McpStreamableHttpClient::with_token(srv.clone(), token) {
                Ok(c) => Arc::new(c),
                Err(err) => {
                    warn!(server = %srv.name, error = %err, "Failed to build MCP client; degrading gracefully");
                    pool.failed_servers.push((srv.name.clone(), err.to_string()));
                    continue;
                }
            };

            match client.list_allowed_tools(traceparent).await {
                Ok(discovered) => {
                    info!(
                        server = %srv.name,
                        tool_count = discovered.len(),
                        "Connected to Streamable HTTP MCP server"
                    );
                    for desc in discovered {
                        pool.tools
                            .insert(desc.name.clone(), (desc, Arc::clone(&client)));
                    }
                    pool.clients.insert(srv.name.clone(), client);
                }
                Err(err) => {
                    warn!(
                        server = %srv.name,
                        url = %srv.url,
                        error = %err,
                        "MCP server unreachable during discovery; continuing with remaining healthy servers (CAP-5)"
                    );
                    pool.failed_servers.push((srv.name.clone(), err.to_string()));
                }
            }
        }

        Ok(pool)
    }

    pub fn discovered_tools(&self) -> Vec<&McpToolDescriptor> {
        self.tools.values().map(|(d, _)| d).collect()
    }

    pub fn failed_servers(&self) -> &[(String, String)] {
        &self.failed_servers
    }

    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools.contains_key(tool_name)
    }

    /// Invokes a discovered, allowlisted MCP tool by name through its owning server client.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        traceparent: Option<&str>,
    ) -> Result<String, McpClientError> {
        let (_, client) = self
            .tools
            .get(tool_name)
            .ok_or_else(|| McpClientError::ToolNotAllowed {
                server: "pool".into(),
                tool: tool_name.to_string(),
            })?;
        client.call_tool(tool_name, arguments, traceparent).await
    }
}
