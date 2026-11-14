//! Apache Arrow `RecordBatch` conversion and atomic Delta Lake (`delta/llm_benchmarks/`) persistence.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use chrono::Utc;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde_json::json;
use thiserror::Error;
use uuid::Uuid;

use crate::benchmark::runner::BenchmarkCaseResult;

/// Default relative directory for persisted LLM benchmark Delta tables.
pub const DEFAULT_DELTA_DIR: &str = "delta/llm_benchmarks";

/// Errors returned during Arrow `RecordBatch` creation or Delta Lake persistence.
#[derive(Debug, Error)]
pub enum BenchmarkStorageError {
    #[error("cannot persist empty benchmark result batch")]
    EmptyBatch,

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Canonical Apache Arrow schema for `delta/llm_benchmarks/`.
pub fn llm_benchmarks_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("case_id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("case_name", DataType::Utf8, false),
        Field::new("provider", DataType::Utf8, false),
        Field::new("model", DataType::Utf8, false),
        Field::new("dry_run", DataType::Boolean, false),
        Field::new("passed", DataType::Boolean, false),
        Field::new("ttft_us", DataType::Int64, true),
        Field::new("itl_mean_us", DataType::Float64, true),
        Field::new("itl_jitter_us", DataType::Float64, true),
        Field::new("tps", DataType::Float64, true),
        Field::new("token_count", DataType::Int64, true),
        Field::new("total_duration_us", DataType::Int64, false),
        Field::new("tool_call_valid", DataType::Boolean, true),
        Field::new("guardrail_passed", DataType::Boolean, true),
        Field::new("validation_us", DataType::Int64, true),
        Field::new("error", DataType::Utf8, true),
        Field::new("timestamp_utc", DataType::Utf8, false),
    ]))
}

/// Convert a non-empty slice of [`BenchmarkCaseResult`]s into an Arrow [`RecordBatch`].
pub fn results_to_record_batch(
    results: &[BenchmarkCaseResult],
) -> Result<RecordBatch, BenchmarkStorageError> {
    if results.is_empty() {
        return Err(BenchmarkStorageError::EmptyBatch);
    }

    let schema = llm_benchmarks_schema();

    let run_ids = StringArray::from_iter_values(results.iter().map(|r| r.run_id.as_str()));
    let case_ids = StringArray::from_iter_values(results.iter().map(|r| r.case_id.as_str()));
    let categories = StringArray::from_iter_values(results.iter().map(|r| r.category.as_str()));
    let case_names = StringArray::from_iter_values(results.iter().map(|r| r.case_name.as_str()));
    let providers = StringArray::from_iter_values(results.iter().map(|r| r.provider.as_str()));
    let models = StringArray::from_iter_values(results.iter().map(|r| r.model.as_str()));
    let dry_run = BooleanArray::from(results.iter().map(|r| r.dry_run).collect::<Vec<_>>());
    let passed = BooleanArray::from(results.iter().map(|r| r.passed).collect::<Vec<_>>());

    let ttft_us: Int64Array = results
        .iter()
        .map(|r| r.metrics.as_ref().map(|m| m.ttft_us as i64))
        .collect();
    let itl_mean_us: Float64Array = results
        .iter()
        .map(|r| r.metrics.as_ref().map(|m| m.itl_mean_us))
        .collect();
    let itl_jitter_us: Float64Array = results
        .iter()
        .map(|r| r.metrics.as_ref().map(|m| m.itl_jitter_us))
        .collect();
    let tps: Float64Array = results
        .iter()
        .map(|r| r.metrics.as_ref().map(|m| m.tps))
        .collect();
    let token_count: Int64Array = results
        .iter()
        .map(|r| r.metrics.as_ref().map(|m| m.token_count as i64))
        .collect();

    let total_duration_us = Int64Array::from(
        results
            .iter()
            .map(|r| r.total_duration_us as i64)
            .collect::<Vec<_>>(),
    );
    let tool_call_valid: BooleanArray = results.iter().map(|r| r.tool_call_valid).collect();
    let guardrail_passed: BooleanArray = results.iter().map(|r| r.guardrail_passed).collect();
    let validation_us: Int64Array = results
        .iter()
        .map(|r| r.validation_us.map(|v| v as i64))
        .collect();
    let error: StringArray = results.iter().map(|r| r.error.as_deref()).collect();
    let timestamp_utc =
        StringArray::from_iter_values(results.iter().map(|r| r.timestamp_utc.as_str()));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(run_ids),
            Arc::new(case_ids),
            Arc::new(categories),
            Arc::new(case_names),
            Arc::new(providers),
            Arc::new(models),
            Arc::new(dry_run),
            Arc::new(passed),
            Arc::new(ttft_us),
            Arc::new(itl_mean_us),
            Arc::new(itl_jitter_us),
            Arc::new(tps),
            Arc::new(token_count),
            Arc::new(total_duration_us),
            Arc::new(tool_call_valid),
            Arc::new(guardrail_passed),
            Arc::new(validation_us),
            Arc::new(error),
            Arc::new(timestamp_utc),
        ],
    )?;

    Ok(batch)
}

/// Atomically append a [`RecordBatch`] as a Parquet data file + `_delta_log/{version:020}.json` commit.
///
/// Uses `OpenOptions::create_new(true)` for atomic version allocation so concurrent writers cannot
/// overwrite an existing Delta log version. Returns the committed Delta version number.
pub fn append_delta_batch(
    table_dir: &Path,
    batch: &RecordBatch,
) -> Result<u64, BenchmarkStorageError> {
    if batch.num_rows() == 0 {
        return Err(BenchmarkStorageError::EmptyBatch);
    }

    fs::create_dir_all(table_dir)?;
    let log_dir = table_dir.join("_delta_log");
    fs::create_dir_all(&log_dir)?;

    let file_id = Uuid::now_v7();
    let parquet_filename = format!("part-00000-{file_id}.parquet");
    let parquet_path = table_dir.join(&parquet_filename);

    write_parquet_file(&parquet_path, batch)?;

    let commit_result = commit_delta_log(
        &log_dir,
        &parquet_filename,
        &parquet_path,
        batch.schema_ref(),
    );
    if commit_result.is_err() {
        let _ = fs::remove_file(&parquet_path);
    }
    commit_result
}

fn commit_delta_log(
    log_dir: &Path,
    parquet_filename: &str,
    parquet_path: &Path,
    schema: &SchemaRef,
) -> Result<u64, BenchmarkStorageError> {
    let file_size = fs::metadata(parquet_path)?.len();
    let now_ms = Utc::now().timestamp_millis();
    let mut version = discover_next_version(log_dir)?;

    loop {
        let payload =
            build_delta_commit_ndjson(version, parquet_filename, file_size, now_ms, schema)?;
        let log_path = delta_log_path(log_dir, version);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&log_path)
        {
            Ok(mut file) => {
                if let Err(err) = file
                    .write_all(payload.as_bytes())
                    .and_then(|()| file.sync_all())
                {
                    let _ = fs::remove_file(&log_path);
                    return Err(BenchmarkStorageError::Io(err));
                }
                return Ok(version);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                version += 1;
            }
            Err(err) => return Err(BenchmarkStorageError::Io(err)),
        }
    }
}

fn write_parquet_file(path: &Path, batch: &RecordBatch) -> Result<(), BenchmarkStorageError> {
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    let file = writer.into_inner()?;
    file.sync_all()?;
    Ok(())
}

fn delta_log_path(log_dir: &Path, version: u64) -> PathBuf {
    log_dir.join(format!("{version:020}.json"))
}

fn discover_next_version(log_dir: &Path) -> Result<u64, std::io::Error> {
    let mut max_version: Option<u64> = None;
    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if let Some(stem) = name_str.strip_suffix(".json") {
            if stem.len() == 20 && stem.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(v) = stem.parse::<u64>() {
                    max_version = Some(max_version.map_or(v, |cur| cur.max(v)));
                }
            }
        }
    }
    Ok(max_version.map_or(0, |v| v + 1))
}

fn build_delta_commit_ndjson(
    version: u64,
    parquet_filename: &str,
    file_size: u64,
    now_ms: i64,
    schema: &SchemaRef,
) -> Result<String, serde_json::Error> {
    let mut lines = Vec::new();

    if version == 0 {
        let protocol = json!({
            "protocol": {
                "minReaderVersion": 1,
                "minWriterVersion": 2
            }
        });
        lines.push(serde_json::to_string(&protocol)?);

        let delta_schema = arrow_schema_to_delta_schema_string(schema)?;
        let metadata = json!({
            "metaData": {
                "id": Uuid::now_v7().to_string(),
                "format": {
                    "provider": "parquet",
                    "options": {}
                },
                "schemaString": delta_schema,
                "partitionColumns": [],
                "configuration": {},
                "createdTime": now_ms
            }
        });
        lines.push(serde_json::to_string(&metadata)?);
    }

    let add = json!({
        "add": {
            "path": parquet_filename,
            "partitionValues": {},
            "size": file_size,
            "modificationTime": now_ms,
            "dataChange": true
        }
    });
    lines.push(serde_json::to_string(&add)?);

    let commit_info = json!({
        "commitInfo": {
            "timestamp": now_ms,
            "operation": "WRITE",
            "operationParameters": {
                "mode": "Append"
            }
        }
    });
    lines.push(serde_json::to_string(&commit_info)?);

    Ok(format!("{}\n", lines.join("\n")))
}

fn arrow_schema_to_delta_schema_string(schema: &SchemaRef) -> Result<String, serde_json::Error> {
    let fields: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|field| {
            let delta_type = match field.data_type() {
                DataType::Utf8 => "string",
                DataType::Boolean => "boolean",
                DataType::Int64 | DataType::UInt64 => "long",
                DataType::Float64 => "double",
                _ => "string",
            };
            json!({
                "name": field.name(),
                "type": delta_type,
                "nullable": field.is_nullable(),
                "metadata": {}
            })
        })
        .collect();

    serde_json::to_string(&json!({
        "type": "struct",
        "fields": fields
    }))
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;
    use crate::benchmark::metrics::BenchmarkMetrics;

    fn sample_latency_result(case_id: &str) -> BenchmarkCaseResult {
        BenchmarkCaseResult {
            run_id: "run-001".to_string(),
            case_id: case_id.to_string(),
            category: "latency".to_string(),
            case_name: "Sample latency case".to_string(),
            provider: "dry-run".to_string(),
            model: "synthetic-dry-run".to_string(),
            dry_run: true,
            passed: true,
            metrics: Some(BenchmarkMetrics {
                ttft_us: 12_500,
                itl_mean_us: 2_500.0,
                itl_jitter_us: 250.0,
                tps: 120.0,
                token_count: 8,
                total_duration_us: 32_500,
            }),
            total_duration_us: 32_500,
            tool_call_valid: None,
            guardrail_passed: None,
            validation_us: None,
            error: None,
            timestamp_utc: "2026-11-14T00:00:00Z".to_string(),
        }
    }

    fn sample_tool_result(case_id: &str) -> BenchmarkCaseResult {
        BenchmarkCaseResult {
            run_id: "run-001".to_string(),
            case_id: case_id.to_string(),
            category: "tools".to_string(),
            case_name: "Sample tool case".to_string(),
            provider: "dry-run".to_string(),
            model: "synthetic-dry-run".to_string(),
            dry_run: true,
            passed: true,
            metrics: None,
            total_duration_us: 42,
            tool_call_valid: Some(true),
            guardrail_passed: None,
            validation_us: Some(42),
            error: None,
            timestamp_utc: "2026-11-14T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn rejects_empty_batch() {
        let err = results_to_record_batch(&[]).expect_err("empty results must fail");
        assert!(matches!(err, BenchmarkStorageError::EmptyBatch));
    }

    #[test]
    fn non_latency_rows_emit_null_metrics_and_int64_physical_types() {
        let batch = results_to_record_batch(&[
            sample_latency_result("lat-001"),
            sample_tool_result("tool-001"),
        ])
        .unwrap();

        let ttft = batch
            .column_by_name("ttft_us")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(!ttft.is_null(0));
        assert_eq!(ttft.value(0), 12_500);
        assert!(ttft.is_null(1));

        let dry_run = batch
            .column_by_name("dry_run")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(dry_run.value(0));
        assert!(dry_run.value(1));
    }

    #[test]
    fn appends_two_delta_versions_atomically_and_reads_back_parquet() {
        let temp_dir = std::env::temp_dir().join(format!("ferrous-delta-test-{}", Uuid::now_v7()));
        let _ = fs::remove_dir_all(&temp_dir);

        let batch1 = results_to_record_batch(&[sample_latency_result("lat-001")]).unwrap();
        let v0 = append_delta_batch(&temp_dir, &batch1).expect("first append should succeed");
        assert_eq!(v0, 0);

        let batch2 = results_to_record_batch(&[
            sample_latency_result("lat-002"),
            sample_tool_result("tool-001"),
        ])
        .unwrap();
        let v1 = append_delta_batch(&temp_dir, &batch2).expect("second append should succeed");
        assert_eq!(v1, 1);

        let log0_path = temp_dir
            .join("_delta_log")
            .join("00000000000000000000.json");
        let log1_path = temp_dir
            .join("_delta_log")
            .join("00000000000000000001.json");
        assert!(log0_path.exists());
        assert!(log1_path.exists());

        let log0 = fs::read_to_string(&log0_path).unwrap();
        assert!(log0.contains("\"protocol\""));
        assert!(log0.contains("\"metaData\""));
        assert!(log0.contains("\"add\""));

        let log1 = fs::read_to_string(&log1_path).unwrap();
        assert!(!log1.contains("\"protocol\""));
        assert!(!log1.contains("\"metaData\""));
        assert!(log1.contains("\"add\""));

        // Verify Parquet files are readable via parquet::arrow::arrow_reader
        let mut total_rows = 0usize;
        for entry in fs::read_dir(&temp_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
                let file = File::open(&path).unwrap();
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .build()
                    .unwrap();
                for read_batch in reader {
                    let rb = read_batch.unwrap();
                    assert_eq!(rb.num_columns(), llm_benchmarks_schema().fields().len());
                    let case_ids = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();
                    assert!(!case_ids.is_null(0));
                    total_rows += rb.num_rows();
                }
            }
        }
        assert_eq!(total_rows, 3);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
