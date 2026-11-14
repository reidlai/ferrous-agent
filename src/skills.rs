//! Declarative modular agent skills catalog loaded from `skills/*/SKILL.md`.

use std::collections::HashSet;
use std::fmt;

const ORDER_INVESTIGATION_MD: &str = include_str!("../skills/order-investigation/SKILL.md");
const COURTESY_CREDIT_MD: &str = include_str!("../skills/courtesy-credit/SKILL.md");
const CARRIER_ESCALATION_MD: &str = include_str!("../skills/carrier-escalation/SKILL.md");

/// Metadata, allowed tools, and system prompt fragment for a single declarative skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDefinition {
    /// Canonical skill identifier (matches `skills/<name>/SKILL.md`).
    pub name: &'static str,
    /// Human- and model-facing summary of when to apply this skill.
    pub description: &'static str,
    /// Tool names (`rig::tool::Tool::NAME`) authorized for this skill.
    pub allowed_tools: &'static [&'static str],
    /// Markdown runbook body injected into the Copilot system preamble.
    pub prompt_fragment: &'static str,
    /// Complete raw `SKILL.md` document including YAML frontmatter.
    pub raw_markdown: &'static str,
}

impl SkillDefinition {
    /// Extract the markdown body after YAML frontmatter (`--- ... ---`), if present.
    fn from_embedded(
        name: &'static str,
        description: &'static str,
        allowed_tools: &'static [&'static str],
        raw_markdown: &'static str,
    ) -> Self {
        let prompt_fragment = strip_frontmatter(raw_markdown);
        Self {
            name,
            description,
            allowed_tools,
            prompt_fragment,
            raw_markdown,
        }
    }
}

impl fmt::Display for SkillDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} tools)", self.name, self.allowed_tools.len())
    }
}

/// Registry of built-in declarative CX Copilot skills.
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    skills: Vec<SkillDefinition>,
}

impl SkillRegistry {
    /// Load the built-in FerrousCX skills catalog (`order-investigation`, `courtesy-credit`,
    /// `carrier-escalation`). Replay-control skills are intentionally excluded.
    pub fn builtin() -> Self {
        Self {
            skills: vec![
                SkillDefinition::from_embedded(
                    "order-investigation",
                    "Multi-step investigation diagnosing order status, shipment milestones, carrier delay duration, and customer history.",
                    &[
                        "api_get_order_details",
                        "api_get_tracking",
                        "transcripts_rag",
                        "policy_rag",
                    ],
                    ORDER_INVESTIGATION_MD,
                ),
                SkillDefinition::from_embedded(
                    "courtesy-credit",
                    "Evaluates delay criteria against cx_policy_knowledge SOPs and formats a Human-in-the-Loop (HITL) courtesy credit proposal card up to $20.00 USD.",
                    &[
                        "policy_rag",
                        "api_get_order_details",
                        "api_get_tracking",
                        "api_submit_credit_command",
                    ],
                    COURTESY_CREDIT_MD,
                ),
                SkillDefinition::from_embedded(
                    "carrier-escalation",
                    "Detects carrier SLA breach patterns (e.g., Correios SEDEX > 4 days) and prepares formal carrier dispute escalation proposals.",
                    &[
                        "api_get_order_details",
                        "api_get_tracking",
                        "policy_rag",
                        "transcripts_rag",
                    ],
                    CARRIER_ESCALATION_MD,
                ),
            ],
        }
    }

    /// Return all registered skills in deterministic order.
    pub fn skills(&self) -> &[SkillDefinition] {
        &self.skills
    }

    /// Look up a skill by its canonical name.
    pub fn get(&self, name: &str) -> Option<&SkillDefinition> {
        let normalized = name.trim();
        self.skills.iter().find(|s| s.name == normalized)
    }

    /// Compose a system preamble combining an optional base preamble with the selected skills.
    /// Unknown skill names are ignored.
    pub fn compose_preamble(&self, base_preamble: Option<&str>, skill_names: &[&str]) -> String {
        let mut sections: Vec<String> = Vec::new();
        if let Some(base) = base_preamble.map(str::trim).filter(|s| !s.is_empty()) {
            sections.push(base.to_string());
        }

        let mut seen = HashSet::new();
        for name in skill_names {
            if let Some(skill) = self.get(name) {
                if seen.insert(skill.name) {
                    sections.push(format!(
                        "## Skill: {}\nAllowed tools: {}\n\n{}",
                        skill.name,
                        skill.allowed_tools.join(", "),
                        skill.prompt_fragment
                    ));
                }
            }
        }

        sections.join("\n\n")
    }

    /// Compose a system preamble including all registered built-in skills.
    pub fn compose_all_preamble(&self, base_preamble: Option<&str>) -> String {
        let names: Vec<&str> = self.skills.iter().map(|s| s.name).collect();
        self.compose_preamble(base_preamble, &names)
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

fn strip_frontmatter(raw: &str) -> &str {
    let trimmed = raw.trim_start();
    if let Some(after_open) = trimmed.strip_prefix("---") {
        if let Some(end_idx) = after_open.find("\n---") {
            let body_start = end_idx + "\n---".len();
            return after_open[body_start..].trim();
        }
    }
    raw.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_exposes_three_skills_without_replay() {
        let registry = SkillRegistry::builtin();
        let names: Vec<&str> = registry.skills().iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            vec![
                "order-investigation",
                "courtesy-credit",
                "carrier-escalation"
            ]
        );

        for skill in registry.skills() {
            assert!(!skill.description.is_empty());
            assert!(!skill.allowed_tools.is_empty());
            assert!(!skill.prompt_fragment.is_empty());
            assert!(
                !skill.prompt_fragment.starts_with("---"),
                "frontmatter should be stripped from prompt_fragment for {}",
                skill.name
            );
            assert!(
                !skill.name.contains("replay"),
                "replay skills are strictly forbidden"
            );
            for tool in skill.allowed_tools {
                assert!(
                    !tool.contains("replay"),
                    "replay tools are strictly forbidden"
                );
            }
        }
    }

    #[test]
    fn compose_preamble_includes_base_and_requested_skills() {
        let registry = SkillRegistry::builtin();
        let composed = registry.compose_preamble(
            Some("You are FerrousCX Copilot."),
            &[
                "order-investigation",
                "courtesy-credit",
                "order-investigation",
                "nonexistent",
            ],
        );

        assert!(composed.starts_with("You are FerrousCX Copilot."));
        assert_eq!(composed.matches("## Skill: order-investigation").count(), 1);
        assert!(composed.contains("api_get_order_details"));
        assert!(composed.contains("## Skill: courtesy-credit"));
        assert!(composed.contains("api_submit_credit_command"));
        assert!(!composed.contains("carrier-escalation"));
    }

    #[test]
    fn compose_all_preamble_includes_every_builtin_skill() {
        let registry = SkillRegistry::builtin();
        let composed = registry.compose_all_preamble(None);
        assert!(composed.contains("## Skill: order-investigation"));
        assert!(composed.contains("## Skill: courtesy-credit"));
        assert!(composed.contains("## Skill: carrier-escalation"));
    }
}
