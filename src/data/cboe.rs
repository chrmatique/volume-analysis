use anyhow::{Context, Result};
use chrono::NaiveDate;
use std::io::Cursor;

use crate::data::cache;
use crate::data::models::{PutCallRecord, SkewRecord};

const TOTALPC_URL: &str =
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv";
const SKEW_URL: &str =
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv";
const CACHE_AGE_HOURS: u64 = 12;

/// Parse date from various formats (YYYY-MM-DD, M/D/YYYY, etc.)
fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
        .ok()
        .or_else(|| NaiveDate::parse_from_str(s.trim(), "%m/%d/%Y").ok())
        .or_else(|| NaiveDate::parse_from_str(s.trim(), "%m/%d/%y").ok())
}

/// Fetch and parse CBOE Total Put/Call ratio from totalpc.csv
pub async fn fetch_put_call_ratio() -> Result<Vec<PutCallRecord>> {
    let cache_file = "cboe_put_call.json";
    if cache::is_cache_fresh(cache_file, CACHE_AGE_HOURS) {
        if let Ok(cached) = cache::load_json::<Vec<PutCallRecord>>(cache_file) {
            tracing::info!("Using cached CBOE put/call ratio");
            return Ok(cached);
        }
    }

    tracing::info!("Fetching CBOE put/call ratio from totalpc.csv");
    let text = match reqwest::get(TOTALPC_URL).await {
        Ok(resp) => resp
            .text()
            .await
            .context("Failed to read totalpc.csv response")?,
        Err(e) => {
            tracing::warn!("Failed to fetch totalpc.csv: {} - trying cache", e);
            if let Ok(cached) = cache::load_json::<Vec<PutCallRecord>>(cache_file) {
                return Ok(cached);
            }
            return Err(e.into());
        }
    };

    let mut records = parse_totalpc_csv(&text)?;
    records.sort_by_key(|r| r.date);

    if let Err(e) = cache::save_json(cache_file, &records) {
        tracing::warn!("Failed to cache put/call ratio: {}", e);
    }

    Ok(records)
}

/// Parse totalpc.csv. Supports two formats:
/// 1. Column-based: Date, Call Volume, Put Volume, P/C Ratio as column headers
/// 2. Row-based (transposed): "DATE" row has dates across columns,
///    "P/C Ratio" or "TOTAL PUT/CALL RATIO" row has values
fn parse_totalpc_csv(text: &str) -> Result<Vec<PutCallRecord>> {
    // Try column-based first (standard CSV)
    if let Ok(records) = parse_totalpc_columns(text) {
        if !records.is_empty() {
            return Ok(records);
        }
    }

    // Fall back to transposed (DATE row) format
    parse_totalpc_transposed(text)
}

/// Parse column-based format: Date, Call Volume, Put Volume, P/C Ratio columns
fn parse_totalpc_columns(text: &str) -> Result<Vec<PutCallRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(Cursor::new(text));

    let headers = reader.headers().context("Missing CSV headers")?.clone();

    let date_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Date"))
        .context("No Date column in totalpc.csv")?;

    let pc_idx = headers
        .iter()
        .position(|h| {
            h.eq_ignore_ascii_case("P/C Ratio")
                || h.eq_ignore_ascii_case("PC Ratio")
                || h.eq_ignore_ascii_case("Put/Call Ratio")
                || h.eq_ignore_ascii_case("Total Put/Call Ratio")
                || h.contains("Ratio")
        })
        .context("No P/C Ratio column in totalpc.csv")?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.context("Invalid CSV row")?;
        if record.len() <= date_idx.max(pc_idx) {
            continue;
        }

        let date_str = record.get(date_idx).unwrap_or("");
        let pc_str = record.get(pc_idx).unwrap_or("");

        let Some(date) = parse_date(date_str) else {
            continue;
        };
        let Ok(pc_ratio) = pc_str.trim().parse::<f64>() else {
            continue;
        };
        if pc_ratio <= 0.0 || !pc_ratio.is_finite() {
            continue;
        }

        records.push(PutCallRecord { date, pc_ratio });
    }

    Ok(records)
}

/// Parse transposed format: "DATE" row has dates, "P/C Ratio" row has values
fn parse_totalpc_transposed(text: &str) -> Result<Vec<PutCallRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(Cursor::new(text));

    // Build a map: row_label -> Vec of column values (first column = label, rest = values)
    let mut rows: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    // Include header row - in transposed format, header may BE the DATE row
    let headers = reader.headers().context("Missing CSV headers")?;
    if !headers.is_empty() {
        let row_label = headers.get(0).unwrap_or("").trim().to_string();
        let values: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();
        if !row_label.is_empty() && !values.is_empty() {
            rows.insert(row_label, values);
        }
    }

    for result in reader.records() {
        let record = result.context("Invalid CSV row")?;
        if record.is_empty() {
            continue;
        }
        let row_label = record.get(0).unwrap_or("").trim().to_string();
        if row_label.is_empty() {
            continue;
        }
        let values: Vec<String> = record.iter().skip(1).map(|s| s.to_string()).collect();
        rows.insert(row_label, values);
    }

    // Find DATE row (case-insensitive)
    let date_values = rows
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Date") || k.eq_ignore_ascii_case("DATE"))
        .map(|(_, v)| v.clone())
        .context("No DATE row in totalpc.csv")?;

    // Find P/C Ratio row (TOTAL PUT/CALL RATIO or P/C Ratio)
    let pc_values = rows
        .iter()
        .find(|(k, _)| {
            let u = k.to_uppercase();
            u.contains("P/C RATIO")
                || u.contains("PUT/CALL RATIO")
                || u.eq("TOTAL PUT/CALL RATIO")
                || u.eq("P/C RATIO")
        })
        .map(|(_, v)| v.clone())
        .context("No P/C Ratio row in totalpc.csv")?;

    let mut records = Vec::new();
    let len = date_values.len().min(pc_values.len());
    for i in 0..len {
        let date_str = date_values.get(i).map(|s| s.as_str()).unwrap_or("");
        let pc_str = pc_values.get(i).map(|s| s.as_str()).unwrap_or("");

        let Some(date) = parse_date(date_str) else {
            continue;
        };
        let Ok(pc_ratio) = pc_str.trim().replace(',', "").parse::<f64>() else {
            continue;
        };
        if pc_ratio <= 0.0 || !pc_ratio.is_finite() {
            continue;
        }

        records.push(PutCallRecord { date, pc_ratio });
    }

    Ok(records)
}

/// Fetch and parse CBOE SKEW index history from SKEW_History.csv
pub async fn fetch_skew_history() -> Result<Vec<SkewRecord>> {
    let cache_file = "cboe_skew.json";
    if cache::is_cache_fresh(cache_file, CACHE_AGE_HOURS) {
        if let Ok(cached) = cache::load_json::<Vec<SkewRecord>>(cache_file) {
            tracing::info!("Using cached CBOE SKEW history");
            return Ok(cached);
        }
    }

    tracing::info!("Fetching CBOE SKEW from SKEW_History.csv");
    let text = match reqwest::get(SKEW_URL).await {
        Ok(resp) => resp
            .text()
            .await
            .context("Failed to read SKEW_History.csv response")?,
        Err(e) => {
            tracing::warn!("Failed to fetch SKEW_History.csv: {} - trying cache", e);
            if let Ok(cached) = cache::load_json::<Vec<SkewRecord>>(cache_file) {
                return Ok(cached);
            }
            return Err(e.into());
        }
    };

    let mut records = parse_skew_csv(&text)?;
    records.sort_by_key(|r| r.date);

    if let Err(e) = cache::save_json(cache_file, &records) {
        tracing::warn!("Failed to cache SKEW history: {}", e);
    }

    Ok(records)
}

/// Parse SKEW_History.csv: Date, Open, High, Low, Close (or Price), Volume, Change %
fn parse_skew_csv(text: &str) -> Result<Vec<SkewRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(Cursor::new(text));

    let headers = reader
        .headers()
        .context("Missing CSV headers")?
        .clone();

    let date_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Date"))
        .context("No Date column in SKEW_History.csv")?;

    let close_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Close") || h.eq_ignore_ascii_case("Price"))
        .or_else(|| headers.iter().position(|h| h.eq_ignore_ascii_case("SKEW")))
        .context("No Close/Price/SKEW column in SKEW_History.csv")?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.context("Invalid CSV row")?;
        if record.len() <= date_idx.max(close_idx) {
            continue;
        }

        let date_str = record.get(date_idx).unwrap_or("");
        let close_str = record.get(close_idx).unwrap_or("");

        let Some(date) = parse_date(date_str) else {
            continue;
        };
        let Ok(skew) = close_str.trim().replace(',', "").parse::<f64>() else {
            continue;
        };
        if skew <= 0.0 || !skew.is_finite() {
            continue;
        }

        records.push(SkewRecord { date, skew });
    }

    Ok(records)
}
