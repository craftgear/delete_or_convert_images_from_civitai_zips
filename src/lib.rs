use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use filetime::{set_file_times, FileTime};
use indicatif::{ProgressBar, ProgressStyle};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageFormat};
use jpegxl_rs::encode::{EncoderFrame, JxlEncoder, Metadata as JxlMetadata};
use jpegxl_rs::encode::EncoderSpeed;
use jpegxl_rs::encoder_builder as jxl_encoder_builder;
use jpegxl_rs::parallel::ParallelRunner;
use jpegxl_rs::ThreadsRunner;
use rayon::prelude::*;
use regex::Regex;
use sha2::{Digest, Sha256};
use signal_hook::consts::SIGINT;
use signal_hook::flag as signal_flag;
use thiserror::Error;
use webp::{Encoder as WebpEncoder, WebPConfig};
use walkdir::WalkDir;
use zip::read::ZipFile;
use zip::write::FileOptions;
use zip::ZipArchive;
use tempfile::Builder;

mod ffi;
mod gpu_jpeg;

use crate::gpu_jpeg::convert_png_to_jpeg_gpu;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("IO error at \u{1b}[91m{path}\u{1b}[0m: {source}")]
    IoWithPath {
        path: String,
        source: std::io::Error,
    },
    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("Zip error at \u{1b}[91m{path}\u{1b}[0m: {source}")]
    ZipWithPath {
        path: String,
        source: zip::result::ZipError,
    },
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("EXIF parse error: {0}")]
    Exif(#[from] exif::Error),
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("JXL encode error: {0}")]
    JxlEncode(#[from] jpegxl_rs::EncodeError),
    #[error("Invalid data: {0}")]
    Invalid(String),
    #[error("Interrupted")]
    Interrupted,
}

type CacheMap = HashMap<String, HashMap<String, String>>;

#[derive(Clone)]
struct EntryMeta {
    name: String,
    stem: String,
    kind: EntryKind,
}

#[derive(Clone, Copy)]
enum EntryKind {
    Json,
    Png,
    Webp,
    Jpeg,
    Other,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ConvertFormat {
    Webp,
    #[value(alias = "jpg")]
    Jpeg,
    #[value(name = "jpg_gpu")]
    JpegGpu,
    Jxl,
}

struct ProgressInfo {
    name: String,
    index: usize,
    total: usize,
    term_width: usize,
    start: Instant,
    scroll_wait: Duration,
    scroll_interval: Duration,
    base_elapsed_ms: AtomicU64,
}

const SCROLL_BASE_UNSET: u64 = u64::MAX;

struct ConversionPlan {
    format: ConvertFormat,
    by_png: HashMap<String, ConversionEntry>,
    target_names: HashSet<String>,
}

struct ConversionEntry {
    target_name: String,
    has_json: bool,
}

struct ConversionInput {
    name: String,
    stem: String,
    target_name: String,
    has_json: bool,
    bytes: Vec<u8>,
    text_chunks: Vec<(String, String)>,
}

struct PreparedConversions {
    conversions: HashMap<String, PreparedConversion>,
    skip_stems: HashSet<String>,
    skips: Vec<ConversionSkip>,
}

struct PreparedConversion {
    target_name: String,
    original_bytes: Vec<u8>,
    outcome: ConversionOutcome,
}


pub fn run(
    root: &Path,
    keywords_csv: &str,
    progress: bool,
    convert: Option<ConvertFormat>,
) -> Result<(), AppError> {
    init_rayon_threads(4)?;
    let keywords = parse_keywords(keywords_csv);
    let keyword_key = keyword_key(&keywords, convert);
    let cancel = Arc::new(AtomicBool::new(false));
    signal_flag::register(SIGINT, cancel.clone()).map_err(|e| AppError::Invalid(e.to_string()))?;
    run_with_cancel(root, &keyword_key, &keywords, progress, convert, cancel.as_ref())
}

fn run_with_cancel(
    root: &Path,
    keyword_key: &str,
    keywords: &[String],
    progress: bool,
    convert: Option<ConvertFormat>,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    let cache_path = cache_file_path();
    let mut cache = load_cache(&cache_path);
    let mut errors: Vec<(std::path::PathBuf, AppError)> = Vec::new();
    let root_path = io_path(root);
    let zip_paths: Vec<_> = WalkDir::new(&root_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .map(|ext| ext.eq_ignore_ascii_case("zip"))
                    == Some(true)
        })
        .map(|e| e.into_path())
        .collect();
    let term_width = terminal_width();

    if progress {
        eprintln!("Found {} zip files", zip_paths.len());
    }

    let total = zip_paths.len();
    let scroll_wait = Duration::from_secs(2);
    let scroll_interval = Duration::from_millis(300);
    for (index, path) in zip_paths.iter().enumerate() {
        let progress_info = ProgressInfo {
            name: file_display_name(path),
            index: index + 1,
            total,
            term_width,
            start: Instant::now(),
            scroll_wait,
            scroll_interval,
            base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
        };
        let display_prefix = progress_prefix(&progress_info, 0);
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::Interrupted);
        }
        warn_if_path_long(path);
        let meta = match io_ctx(path, fs::metadata(&io_path(path))) {
            Ok(m) => m,
            Err(e) => {
                errors.push((path.clone(), e));
                continue;
            }
        };
        let zip_hash_before = zip_hash(path, &meta);
        if cache_hit(&cache, &keyword_key, path, &zip_hash_before) {
            if progress {
                eprintln!(
                    "{} {}",
                    color_msg(&display_prefix, "1;37"),
                    color_msg("skipped (processed)", "90")
                );
            }
            continue;
        }
        let changed =
            match process_zip(path, &progress_info, keywords, progress, cancel, convert) {
                Ok(v) => v,
                Err(AppError::Interrupted) => return Err(AppError::Interrupted),
                Err(e) => {
                    errors.push((path.clone(), e));
                    continue;
                }
            };
        let final_hash = if changed {
            let meta_after = io_ctx(path, fs::metadata(&io_path(path)))?;
            zip_hash(path, &meta_after)
        } else {
            zip_hash_before
        };
        cache_insert(&mut cache, &keyword_key, path, final_hash);
        save_cache(&cache_path, &cache)?;
    }
    if !errors.is_empty() {
        eprintln!("{} {}", color_msg("Errors", "31"), errors.len());
        for (p, e) in errors.iter() {
            eprintln!("- {}: {}", display_path(p), e);
        }
        return Err(AppError::Invalid(format!(
            "{} file(s) failed; see log",
            errors.len()
        )));
    }
    Ok(())
}

pub fn clear_cache_file() -> Result<(), AppError> {
    let path = cache_file_path();
    if path.exists() {
        // WHY: 利用者が即座に再試行できるようにキャッシュ削除を提供
        io_ctx(&path, fs::remove_file(&io_path(&path)))?;
    }
    Ok(())
}

fn process_zip(
    path: &Path,
    progress_info: &ProgressInfo,
    keywords: &[String],
    progress: bool,
    cancel: &AtomicBool,
    convert: Option<ConvertFormat>,
) -> Result<bool, AppError> {
    let meta = io_ctx(path, fs::metadata(&io_path(path)))?;
    let atime = FileTime::from_last_access_time(&meta);
    let mtime = FileTime::from_last_modification_time(&meta);
    let dir_times = dir_times(path);

    // 読み取りフェーズ
    let (entries, scan_bar) = {
        let file = io_ctx(path, File::open(&io_path(path)))?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
        let _total = archive.len() as u64;

        let scan_bar = if progress {
            let b = ProgressBar::new(_total.max(1));
            // WHY: 進捗の視認性を優先
            b.set_style(
                ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            reset_scroll_base(progress_info);
            b.set_prefix(color_msg(&progress_prefix(progress_info, 0), "1;37"));
            b.set_message(color_msg("scanning", "36"));
            Some(b)
        } else {
            None
        };

        let mut list = Vec::new();
        for i in 0..archive.len() {
            let file = archive
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            list.push(entry_meta_from_name(file.name())?);
            advance_progress(scan_bar.as_ref(), progress_info);
        }
        (list, scan_bar)
    };

    if let Some(ref b) = scan_bar {
        // WHY: 次の解析表示で同じ行を使うため
        b.finish_and_clear();
    }

    let json_count = count_json_entries(&entries);
    let analyze_bar = if progress && json_count > 0 {
        let b = ProgressBar::new(json_count.max(1));
        b.set_style(
            ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        reset_scroll_base(progress_info);
        b.set_prefix(color_msg(&progress_prefix(progress_info, 0), "1;37"));
        b.set_message(color_msg("analyzing json", "36"));
        Some(b)
    } else {
        None
    };

    let deletions = decide_deletions(
        path,
        progress_info,
        &entries,
        keywords,
        analyze_bar.as_ref(),
        cancel,
    )?;
    let conversion_plan = match convert {
        Some(format) => build_conversion_plan(&entries, &deletions, format)?,
        None => None,
    };
    if deletions.is_empty() && conversion_plan.is_none() {
        if progress {
            eprintln!(
                "{} {}",
                color_msg(&progress_prefix(progress_info, 0), "1;37"),
                color_msg("no changes", "90")
            );
        }
        set_times(path, atime, mtime)?;
        return Ok(false);
    }
    let write_bar = if progress {
        let kept_len = entries.len() as u64 - deletions.len() as u64;
        let b = ProgressBar::new(kept_len.max(1));
        b.set_style(
            ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        reset_scroll_base(progress_info);
        b.set_prefix(color_msg(&progress_prefix(progress_info, 0), "1;37"));
        b.set_message(color_msg("writing", "32"));
        Some(b)
    } else {
        None
    };

    let skip_logs = write_filtered_zip(
        path,
        progress_info,
        &entries,
        &deletions,
        conversion_plan.as_ref(),
        write_bar.as_ref(),
        cancel,
        atime,
        mtime,
    )?;

    if !deletions.is_empty() || conversion_plan.is_some() {
        // WHY: 書き換えが発生したときだけ元の時刻を復元して変更検知を抑制
        set_times(path, atime, mtime)?;
    }
    if let Some((dir_path, dir_atime, dir_mtime)) = dir_times {
        // WHY: tmp作成やrenameで更新されたディレクトリ日時を元に戻す
        let _ = set_times(&dir_path, dir_atime, dir_mtime);
    }
    if !skip_logs.is_empty() {
        warn_skip_pngs(path, &skip_logs);
    }
    Ok(true)
}

fn write_filtered_zip(
    path: &Path,
    progress_info: &ProgressInfo,
    entries: &[EntryMeta],
    deletions: &HashSet<String>,
    conversion_plan: Option<&ConversionPlan>,
    bar: Option<&ProgressBar>,
    cancel: &AtomicBool,
    atime: FileTime,
    mtime: FileTime,
) -> Result<Vec<ConversionSkip>, AppError> {
    let kept_len = entries.len().saturating_sub(deletions.len());

    let dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let orig_permissions = io_ctx(path, fs::metadata(&io_path(path)))?.permissions();
    let prepared_conversions = match conversion_plan {
        Some(plan) => Some(prepare_conversions(
            path,
            plan,
            deletions,
            cancel,
            bar,
            progress_info,
        )?),
        None => None,
    };
    let skip_logs = prepared_conversions
        .as_ref()
        .map(|prepared| prepared.skips.clone())
        .unwrap_or_default();
    if let Some(b) = bar {
        b.set_length(kept_len as u64);
        b.set_position(0);
        reset_scroll_base(progress_info);
        b.set_prefix(color_msg(&progress_prefix(progress_info, 0), "1;37"));
        b.set_message(color_msg("writing", "32"));
        b.tick();
    }

    let mut tmp = Builder::new()
        .prefix("civitai_tmp")
        .suffix(".zip")
        .tempfile_in(dir)?;

    // WHY: 一時ファイルの作成日時を元ZIPに合わせ、後工程でのタイムスタンプ差異を減らす
    set_times(tmp.path(), atime, mtime)?;

    {
        let buf_size = env::var("BUF_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb * (1 << 20))
            .unwrap_or(16 * (1 << 20));

        // WHY: HDD/遅いストレージ向けに書き込み呼び出し回数をさらに減らす
        // WHY: BUF_MBで可変にしベンチしやすくする
        let buf = BufWriter::with_capacity(buf_size, tmp.as_file_mut());
        let mut writer = zip::ZipWriter::new(buf);
        if let Some(b) = bar {
            b.tick(); // WHY: 0件書き込みでもバーを一度描画するため
        }
        // WHY: 再圧縮を避け、元の圧縮データをそのままコピーして速度を優先
        let mut src = ZipArchive::new(io_ctx(path, File::open(&io_path(path)))?)
            .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
        let convert_targets = conversion_plan.map(|p| &p.target_names);
        let entry_stems = build_entry_stems(entries);
        let entry_kinds = build_entry_kinds(entries);
        let skip_stems = prepared_conversions
            .as_ref()
            .map(|prepared| &prepared.skip_stems);
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            let name = file.name().to_string();
            if let Some(skip) = skip_stems {
                if let Some(kind) = entry_kinds.get(&name) {
                    if matches!(kind, EntryKind::Png | EntryKind::Json) {
                        let stem = entry_stems
                            .get(&name)
                            .cloned()
                            .unwrap_or_else(|| {
                                entry_meta_from_name(&name)
                                    .map(|m| m.stem)
                                    .unwrap_or(name.clone())
                            });
                        if skip.contains(&stem) {
                            advance_progress(bar, progress_info);
                            continue;
                        }
                    }
                }
            }
            if deletions.contains(&name) {
                continue;
            }
            if let Some(targets) = convert_targets {
                if targets.contains(&name) {
                    continue;
                }
            }
            if let Some(ref prepared) = prepared_conversions {
                if let Some(entry) = prepared.conversions.get(&name) {
                    let opts = file_options_for_write(&file);
                    match &entry.outcome {
                        ConversionOutcome::Converted(bytes) => {
                            writer.start_file(&entry.target_name, opts)?;
                            writer.write_all(bytes)?;
                        }
                        ConversionOutcome::KeepOriginal => {
                            writer.start_file(&name, opts)?;
                            writer.write_all(&entry.original_bytes)?;
                        }
                    }
                    advance_progress(bar, progress_info);
                    continue;
                }
            }
            writer.raw_copy_file(file)?;
            advance_progress(bar, progress_info);
        }
        writer.finish()?;
    }

    if let Some(b) = bar {
        // WHY: syncやタイムスタンプ復元など後処理中であることを示すため
        b.set_message(color_msg("finalizing", "90"));
        b.tick();
    }

    // WHY: 書き込み後のfsyncコストを避けるため、renameベースで差し替える
    io_ctx(
        tmp.path(),
        fs::set_permissions(&io_path(tmp.path()), orig_permissions.clone()),
    )?;

    if io_path(path).exists() {
        // WHY: Windows上書き問題回避
        io_ctx(path, fs::remove_file(&io_path(path)))?;
    }
    let persist_result = tmp.persist(&io_path(path));
    if let Err(err) = persist_result {
        return Err(AppError::IoWithPath {
            path: display_path(path),
            source: err.error,
        });
    }

    if let Some(b) = bar {
        if kept_len == 0 {
            // WHY: 全削除時も書き込みフェーズを完了表示にするため
            if let Some(len) = b.length() {
                b.set_position(len);
            }
        }
        b.set_message(color_msg("done", "32"));
        b.finish_with_message(color_msg("done", "32"));
    }
    Ok(skip_logs)
}

fn prepare_conversions(
    path: &Path,
    plan: &ConversionPlan,
    deletions: &HashSet<String>,
    cancel: &AtomicBool,
    bar: Option<&ProgressBar>,
    progress_info: &ProgressInfo,
) -> Result<PreparedConversions, AppError> {
    let mut archive = ZipArchive::new(io_ctx(path, File::open(&io_path(path)))?)
        .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
    let mut inputs = Vec::new();
    for i in 0..archive.len() {
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::Interrupted);
        }
        let mut file = archive
            .by_index(i)
            .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
        if file.name().ends_with('/') {
            continue;
        }
        let name = file.name().to_string();
        if deletions.contains(&name) {
            continue;
        }
        let Some(info) = plan.by_png.get(&name) else {
            continue;
        };
        let bytes = read_zipfile_bytes(&mut file)?;
        let text_chunks = png_text_chunks(&bytes);
        let stem = entry_meta_from_name(&name)?.stem;
        inputs.push(ConversionInput {
            name,
            stem,
            target_name: info.target_name.clone(),
            has_json: info.has_json,
            bytes,
            text_chunks,
        });
    }
    if inputs.is_empty() {
        return Ok(PreparedConversions {
            conversions: HashMap::new(),
            skip_stems: HashSet::new(),
            skips: Vec::new(),
        });
    }
    if let Some(b) = bar {
        b.set_length(inputs.len() as u64);
        b.set_position(0);
        reset_scroll_base(progress_info);
        b.set_prefix(color_msg(&progress_prefix(progress_info, 0), "1;37"));
        b.set_message(color_msg("converting", "36"));
        b.tick();
    }
    let ticker_stop = Arc::new(AtomicBool::new(false));
    let ticker_handle = bar.map(|b| {
        let info = progress_info_for_ticker(progress_info);
        start_prefix_ticker(b, info, ticker_stop.clone())
    });
    let format = plan.format;
    // WHY: 画像変換はCPU負荷が高いため先に並列化して待ち時間を減らす
    let results = inputs
        .into_par_iter()
        .map(|input| {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let outcome = convert_png_for_format_with_skip(
                &input.name,
                &input.bytes,
                format,
                &input.text_chunks,
                input.has_json,
            )?;
            if let Some(b) = bar {
                b.inc(1);
            }
            Ok((input, outcome))
        })
        .collect::<Result<Vec<_>, AppError>>(); 
    ticker_stop.store(true, Ordering::Relaxed);
    if let Some(handle) = ticker_handle {
        let _ = handle.join();
    }
    let results = results?;
    let mut map = HashMap::with_capacity(results.len());
    let mut skip_stems = HashSet::new();
    let mut skips = Vec::new();
    for (input, outcome) in results {
        match outcome {
            ConversionAttempt::Outcome(outcome) => {
                map.insert(
                    input.name,
                    PreparedConversion {
                        target_name: input.target_name,
                        original_bytes: input.bytes,
                        outcome,
                    },
                );
            }
            ConversionAttempt::Skipped(skip) => {
                skip_stems.insert(input.stem);
                skips.push(skip);
            }
        }
    }
    Ok(PreparedConversions {
        conversions: map,
        skip_stems,
        skips,
    })
}

fn file_options_for_write(file: &ZipFile) -> FileOptions {
    let mut opts = FileOptions::default().compression_method(file.compression());
    if let Some(mode) = file.unix_mode() {
        opts = opts.unix_permissions(mode);
    }
    opts.last_modified_time(file.last_modified())
}

fn build_conversion_plan(
    entries: &[EntryMeta],
    deletions: &HashSet<String>,
    format: ConvertFormat,
) -> Result<Option<ConversionPlan>, AppError> {
    let json_stems: HashSet<String> = entries
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::Json))
        .filter(|e| !deletions.contains(&e.name))
        .map(|e| e.stem.clone())
        .collect();
    let mut by_png = HashMap::new();
    let mut target_names = HashSet::new();
    for e in entries {
        if !matches!(e.kind, EntryKind::Png) {
            continue;
        }
        if deletions.contains(&e.name) {
            continue;
        }
        let target_name = replace_extension(&e.name, convert_extension(format))?;
        let has_json = json_stems.contains(&e.stem);
        by_png.insert(
            e.name.clone(),
            ConversionEntry {
                target_name: target_name.clone(),
                has_json,
            },
        );
        target_names.insert(target_name);
    }
    if by_png.is_empty() {
        return Ok(None);
    }
    Ok(Some(ConversionPlan {
        format,
        by_png,
        target_names,
    }))
}

fn build_entry_stems(entries: &[EntryMeta]) -> HashMap<String, String> {
    let mut map = HashMap::with_capacity(entries.len());
    for entry in entries {
        map.insert(entry.name.clone(), entry.stem.clone());
    }
    map
}

fn build_entry_kinds(entries: &[EntryMeta]) -> HashMap<String, EntryKind> {
    let mut map = HashMap::with_capacity(entries.len());
    for entry in entries {
        map.insert(entry.name.clone(), entry.kind);
    }
    map
}

fn decide_deletions(
    path: &Path,
    progress_info: &ProgressInfo,
    entries: &[EntryMeta],
    keywords: &[String],
    bar: Option<&ProgressBar>,
    cancel: &AtomicBool,
) -> Result<HashSet<String>, AppError> {
    let mut deletions = HashSet::new();
    let mut stem_to_entries: HashMap<String, Vec<String>> = HashMap::new();
    let mut json_name_to_stem: HashMap<String, String> = HashMap::new();
    let mut image_name_to_meta: HashMap<String, (String, EntryKind)> = HashMap::new();

    for e in entries {
        stem_to_entries
            .entry(e.stem.clone())
            .or_default()
            .push(e.name.clone());
        match e.kind {
            EntryKind::Json => {
                json_name_to_stem.insert(e.name.clone(), e.stem.clone());
            }
            EntryKind::Png | EntryKind::Webp | EntryKind::Jpeg => {
                image_name_to_meta.insert(e.name.clone(), (e.stem.clone(), e.kind));
            }
            EntryKind::Other => {}
        }
    }

    let json_count = json_name_to_stem.len() as u64;

    let mut json_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    if json_count > 0 {
        let mut src = ZipArchive::new(io_ctx(path, File::open(&io_path(path)))?)
            .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let mut file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            let stem = match json_name_to_stem.get(file.name()) {
                Some(v) => v.clone(),
                None => continue,
            };
            let prompt = json_prompt_tags_from_reader(&mut file)?;
            json_prompts.insert(stem, prompt);
            advance_progress(bar, progress_info);
        }
    }

    let mut needs_image_check: HashSet<String> = HashSet::new();
    for (stem, prompt_opt) in json_prompts.iter() {
        if prompt_opt.is_none() {
            needs_image_check.insert(stem.clone());
        }
    }

    let mut image_names: Vec<String> = Vec::new();
    if !needs_image_check.is_empty() {
        for (name, (stem, _kind)) in image_name_to_meta.iter() {
            if needs_image_check.contains(stem) {
                image_names.push(name.clone());
            }
        }
    }

    if let Some(b) = bar {
        let new_len = json_count + image_names.len() as u64;
        b.set_length(new_len.max(1));
        if image_names.is_empty() {
            b.set_message(color_msg("analyzing json", "36"));
        } else {
            b.set_message(color_msg("analyzing images", "36"));
        }
    }

    let mut image_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    if !image_names.is_empty() {
        let image_name_set: HashSet<String> = image_names.into_iter().collect();
        let mut src = ZipArchive::new(io_ctx(path, File::open(&io_path(path)))?)
            .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let mut file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: display_path(path), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            if !image_name_set.contains(file.name()) {
                continue;
            }
            let (stem, kind) = match image_name_to_meta.get(file.name()) {
                Some(v) => v.clone(),
                None => continue,
            };
            let prompt = match kind {
                EntryKind::Png => png_prompt_tags(&read_zipfile_bytes(&mut file)?),
                EntryKind::Webp => webp_prompt_tags(&read_zipfile_bytes(&mut file)?),
                EntryKind::Jpeg => jpeg_prompt_tags(&read_zipfile_bytes(&mut file)?),
                _ => None,
            };
            image_prompts.insert(stem, prompt);
            advance_progress(bar, progress_info);
        }
    }

    if let Some(b) = bar {
        // WHY: 次の行に進ませないため表示行を消す
        b.finish_and_clear();
    }

    for (stem, prompt_opt) in json_prompts.iter() {
        match prompt_opt {
            Some(tags) => {
                if tags_match(tags, keywords) {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            maybe_insert_model_safe(&mut deletions, name);
                        }
                    }
                }
            }
            None => match image_prompts.get(stem) {
                Some(None) => {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            maybe_insert_model_safe(&mut deletions, name);
                        }
                    }
                }
                Some(Some(_)) => {}
                None => {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            if Path::new(name).extension().map(|e| e == "json") == Some(true) {
                                maybe_insert_model_safe(&mut deletions, name);
                            }
                        }
                    }
                }
            },
        }
    }

    Ok(deletions)
}

fn parse_keywords(csv: &str) -> Vec<String> {
    csv.split(',')
        .filter_map(|s| {
            let t = s.trim().to_ascii_lowercase();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        })
        .collect()
}

fn file_display_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| display_path(path))
}

fn progress_prefix(info: &ProgressInfo, offset: usize) -> String {
    let name_limit = name_display_limit(info.term_width);
    let display_name = scrolling_name(&info.name, name_limit, offset);
    format!("{}/{} {}", info.index, info.total, display_name)
}

fn update_progress_prefix(bar: &ProgressBar, info: &ProgressInfo) {
    let elapsed = info.start.elapsed();
    let offset = scroll_offset(info, elapsed);
    let prefix = progress_prefix(info, offset);
    bar.set_prefix(color_msg(&prefix, "1;37"));
}

fn advance_progress(bar: Option<&ProgressBar>, info: &ProgressInfo) {
    if let Some(b) = bar {
        b.inc(1);
        update_progress_prefix(b, info);
    }
}

fn reset_scroll_base(info: &ProgressInfo) {
    info.base_elapsed_ms.store(SCROLL_BASE_UNSET, Ordering::Relaxed);
}

fn progress_info_for_ticker(info: &ProgressInfo) -> ProgressInfo {
    ProgressInfo {
        name: info.name.clone(),
        index: info.index,
        total: info.total,
        term_width: info.term_width,
        start: info.start,
        scroll_wait: info.scroll_wait,
        scroll_interval: info.scroll_interval,
        base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
    }
}

fn start_prefix_ticker(
    bar: &ProgressBar,
    info: ProgressInfo,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    let bar = bar.clone();
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            update_progress_prefix(&bar, &info);
            bar.tick();
            thread::sleep(Duration::from_millis(100));
        }
    })
}

fn scroll_offset(info: &ProgressInfo, elapsed: Duration) -> usize {
    if elapsed < info.scroll_wait {
        return 0;
    }
    let base = info.base_elapsed_ms.load(Ordering::Relaxed);
    if base == SCROLL_BASE_UNSET {
        let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
        let _ = info.base_elapsed_ms.compare_exchange(
            SCROLL_BASE_UNSET,
            elapsed_ms,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
    let base_now = info.base_elapsed_ms.load(Ordering::Relaxed);
    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
    let interval_ms = info.scroll_interval.as_millis().min(u128::from(u64::MAX)) as u64;
    if interval_ms == 0 {
        return 0;
    }
    (elapsed_ms.saturating_sub(base_now) / interval_ms) as usize
}

fn name_display_limit(term_width: usize) -> usize {
    let half = term_width / 2;
    if half == 0 { 1 } else { half }
}

fn scrolling_name(text: &str, width: usize, offset: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len <= width {
        return text.to_string();
    }
    let gap = 8;
    let cycle_len = len + gap;
    let start = offset % cycle_len;
    let mut out = String::with_capacity(width);
    for i in 0..width {
        let idx = (start + i) % cycle_len;
        if idx < len {
            out.push(chars[idx]);
        } else {
            out.push(' ');
        }
    }
    out
}

fn terminal_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(80)
}

fn entry_meta_from_name(name: &str) -> Result<EntryMeta, AppError> {
    let file_name = Path::new(name)
        .file_name()
        .ok_or_else(|| AppError::Invalid(format!("Invalid file name: {}", name)))?
        .to_string_lossy()
        .to_string();
    let stem = Path::new(&file_name)
        .file_stem()
        .ok_or_else(|| AppError::Invalid(format!("Invalid stem: {}", name)))?
        .to_string_lossy()
        .to_string();
    let ext = Path::new(&file_name)
        .extension()
        .map(|s| s.to_string_lossy().to_ascii_lowercase());
    let kind = match ext.as_deref() {
        Some("json") => EntryKind::Json,
        Some("png") | Some("octet-stream") => EntryKind::Png,
        Some("webp") => EntryKind::Webp,
        Some("jpg") | Some("jpeg") => EntryKind::Jpeg,
        _ => EntryKind::Other,
    };

    Ok(EntryMeta {
        name: name.to_string(),
        stem,
        kind,
    })
}

fn json_prompt_tags_from_reader<R: Read>(reader: R) -> Result<Option<Vec<String>>, AppError> {
    let v: serde_json::Value = serde_json::from_reader(reader)?;
    let prompt = find_prompt(&v);
    Ok(prompt.map(|s| normalize_prompt(&s)))
}

fn read_zipfile_bytes(file: &mut ZipFile) -> Result<Vec<u8>, AppError> {
    let mut data = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut data)?;
    Ok(data)
}

fn normalize_prompt(s: &str) -> Vec<String> {
    let mut text = s.replace(['\n', '\r'], "");
    text = text.replace('\u{3000}', " "); // 全角スペースを半角に

    let re = Regex::new(r"[\(\[\{]+([^:\)\]\}\(]+?)(?::[0-9.]+)?[\)\]\}]+").unwrap();
    loop {
        let replaced = re
            .replace_all(&text, |caps: &regex::Captures| caps[1].trim().to_string())
            .to_string();
        if replaced == text {
            break;
        }
        text = replaced;
    }

    text.split(',')
        .filter_map(|t| {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_ascii_lowercase())
            }
        })
        .collect()
}

fn tags_match(tags: &[String], keywords: &[String]) -> bool {
    let kw: HashSet<&str> = keywords.iter().map(String::as_str).collect();
    tags.iter().any(|t| kw.contains(t.as_str()))
}

fn find_prompt(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if k.eq_ignore_ascii_case("prompt") {
                    if let Some(s) = v.as_str() {
                        return Some(s.to_string());
                    }
                }
            }
            for v in map.values() {
                if let Some(found) = find_prompt(v) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                if let Some(found) = find_prompt(v) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn png_prompt_tags(data: &[u8]) -> Option<Vec<String>> {
    let decoder = png::Decoder::new(data);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    for t in &info.uncompressed_latin1_text {
        if t.keyword.eq_ignore_ascii_case("prompt") {
            return Some(normalize_prompt(&t.text));
        }
    }
    for t in &info.compressed_latin1_text {
        if t.keyword.eq_ignore_ascii_case("prompt") {
            if let Ok(text) = t.get_text() {
                return Some(normalize_prompt(&text));
            }
        }
    }
    for t in &info.utf8_text {
        if t.keyword.eq_ignore_ascii_case("prompt") {
            if let Ok(text) = t.get_text() {
                return Some(normalize_prompt(&text));
            }
        }
    }
    let mut buf = vec![0; reader.output_buffer_size()];
    let _ = reader.next_frame(&mut buf).ok()?;
    None
}

fn png_text_chunks(data: &[u8]) -> Vec<(String, String)> {
    let decoder = png::Decoder::new(data);
    let reader = match decoder.read_info() {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let info = reader.info();
    let mut out = Vec::new();
    for t in &info.uncompressed_latin1_text {
        out.push((t.keyword.clone(), t.text.clone()));
    }
    for t in &info.compressed_latin1_text {
        if let Ok(text) = t.get_text() {
            out.push((t.keyword.clone(), text));
        }
    }
    for t in &info.utf8_text {
        if let Ok(text) = t.get_text() {
            out.push((t.keyword.clone(), text));
        }
    }
    out
}

fn webp_prompt_tags(data: &[u8]) -> Option<Vec<String>> {
    if let Some(exif_data) = extract_webp_chunk(data, b"EXIF") {
        if let Ok(prompt) = extract_exif_prompt(&exif_data) {
            if let Some(p) = prompt {
                return Some(normalize_prompt(&p));
            }
        }
    }
    if let Some(xmp) = extract_webp_chunk(data, b"XMP ") {
        if let Some(prompt) = extract_xmp_prompt(&xmp) {
            return Some(normalize_prompt(&prompt));
        }
    }
    None
}

fn jpeg_prompt_tags(data: &[u8]) -> Option<Vec<String>> {
    for segment in extract_jpeg_app1_segments(data) {
        if let Some(exif_data) = extract_jpeg_exif(&segment) {
            if let Ok(prompt) = extract_exif_prompt(&exif_data) {
                if let Some(p) = prompt {
                    return Some(normalize_prompt(&p));
                }
            }
        }
        if let Some(xmp) = extract_jpeg_xmp(&segment) {
            if let Some(prompt) = extract_xmp_prompt(&xmp) {
                return Some(normalize_prompt(&prompt));
            }
        }
    }
    None
}

fn extract_jpeg_app1_segments(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return out;
    }
    let mut offset = 2;
    while offset + 4 <= data.len() {
        if data[offset] != 0xFF {
            break;
        }
        let marker = data[offset + 1];
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        if (0xD0..=0xD7).contains(&marker) {
            offset += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        if seg_len < 2 || offset + 2 + seg_len > data.len() {
            break;
        }
        let seg_start = offset + 4;
        let seg_end = offset + 2 + seg_len;
        if marker == 0xE1 {
            out.push(data[seg_start..seg_end].to_vec());
        }
        offset += 2 + seg_len;
    }
    out
}

fn extract_jpeg_exif(segment: &[u8]) -> Option<Vec<u8>> {
    if segment.starts_with(b"Exif\0\0") && segment.len() > 6 {
        return Some(segment[6..].to_vec());
    }
    None
}

fn extract_jpeg_xmp(segment: &[u8]) -> Option<Vec<u8>> {
    const XMP_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    if segment.starts_with(XMP_HEADER) && segment.len() > XMP_HEADER.len() {
        return Some(segment[XMP_HEADER.len()..].to_vec());
    }
    None
}

fn extract_exif_prompt(data: &[u8]) -> Result<Option<String>, AppError> {
    let mut cursor = std::io::Cursor::new(data);
    match exif::Reader::new().read_from_container(&mut cursor) {
        Ok(exif) => {
            for f in exif.fields() {
                if f.tag == exif::Tag::UserComment {
                    if let exif::Value::Ascii(ref vec) = f.value {
                        if let Some(bytes) = vec.get(0) {
                            let s = String::from_utf8_lossy(bytes).to_string();
                            if s.trim_start().starts_with('{') {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                                    if let Some(prompt) = find_prompt(&v) {
                                        return Ok(Some(prompt));
                                    }
                                    return Ok(None);
                                }
                            }
                            return Ok(Some(s));
                        }
                    }
                }
            }
            Ok(None)
        }
        Err(exif::Error::NotFound(_)) => Ok(None),
        Err(e) => Err(AppError::Exif(e)),
    }
}

fn extract_xmp_prompt(xmp: &[u8]) -> Option<String> {
    let content = String::from_utf8_lossy(xmp);
    if let Some(val) = extract_xmp_tag_value(&content, "prompt") {
        return Some(val);
    }
    extract_xmp_key_value(&content, "prompt")
}

fn extract_xmp_tag_value(content: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = content.find(&open)?;
    let end = content[start + open.len()..].find(&close)?;
    let raw = &content[start + open.len()..start + open.len() + end];
    Some(xml_unescape(raw))
}

fn extract_xmp_key_value(content: &str, key: &str) -> Option<String> {
    let needle = format!("key=\"{}\"", key);
    let pos = content.find(&needle)?;
    let tag_start = content[..pos].rfind('<')?;
    let tag_close = content[tag_start..].find('>')?;
    let body_start = tag_start + tag_close + 1;
    let end_tag = "</civitai:text>";
    let body_end = content[body_start..].find(end_tag)?;
    let raw = &content[body_start..body_start + body_end];
    Some(xml_unescape(raw))
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

fn extract_webp_chunk(data: &[u8], chunk: &[u8; 4]) -> Option<Vec<u8>> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return None;
    }
    let mut offset = 12;
    while offset + 8 <= data.len() {
        let fourcc = &data[offset..offset + 4];
        let size_bytes = &data[offset + 4..offset + 8];
        let size =
            u32::from_le_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]])
                as usize;
        let data_start = offset + 8;
        let data_end = data_start + size;
        if data_end > data.len() {
            break;
        }
        if fourcc == chunk {
            return Some(data[data_start..data_end].to_vec());
        }
        let aligned = size + (size % 2);
        offset = data_start + aligned;
    }
    None
}

fn convert_extension(format: ConvertFormat) -> &'static str {
    match format {
        ConvertFormat::Webp => "webp",
        ConvertFormat::Jpeg => "jpg",
        ConvertFormat::JpegGpu => "jpg",
        ConvertFormat::Jxl => "jxl",
    }
}

fn replace_extension(name: &str, ext: &str) -> Result<String, AppError> {
    let slash = name.rfind('/');
    let (dir, file) = match slash {
        Some(pos) => (&name[..pos], &name[pos + 1..]),
        None => ("", name),
    };
    let dot = file.rfind('.').unwrap_or(file.len());
    let stem = &file[..dot];
    if stem.is_empty() {
        return Err(AppError::Invalid(format!("Invalid file name: {}", name)));
    }
    if dir.is_empty() {
        Ok(format!("{}.{}", stem, ext))
    } else {
        Ok(format!("{}/{}.{}", dir, stem, ext))
    }
}

enum ConversionOutcome {
    Converted(Vec<u8>),
    KeepOriginal,
}

enum ConversionAttempt {
    Outcome(ConversionOutcome),
    Skipped(ConversionSkip),
}

#[derive(Clone)]
struct ConversionSkip {
    name: String,
    error: String,
}

fn convert_png_for_format(
    data: &[u8],
    format: ConvertFormat,
    text_chunks: &[(String, String)],
    has_json: bool,
) -> Result<ConversionOutcome, AppError> {
    match format {
        ConvertFormat::Jxl => convert_png_to_jxl(data, text_chunks, has_json),
        ConvertFormat::Webp | ConvertFormat::Jpeg | ConvertFormat::JpegGpu => {
            let converted = convert_png_bytes_basic(data, format)?;
            if text_chunks.is_empty() {
                return Ok(ConversionOutcome::Converted(converted));
            }
            match embed_converted_metadata(&converted, format, text_chunks) {
                Ok(with_meta) => Ok(ConversionOutcome::Converted(with_meta)),
                Err(_) => {
                    if has_json {
                        Ok(ConversionOutcome::Converted(converted))
                    } else {
                        Ok(ConversionOutcome::KeepOriginal)
                    }
                }
            }
        }
    }
}

fn convert_png_for_format_with_skip(
    name: &str,
    data: &[u8],
    format: ConvertFormat,
    text_chunks: &[(String, String)],
    has_json: bool,
) -> Result<ConversionAttempt, AppError> {
    match convert_png_for_format(data, format, text_chunks, has_json) {
        Ok(outcome) => Ok(ConversionAttempt::Outcome(outcome)),
        Err(AppError::Image(err)) => {
            Ok(ConversionAttempt::Skipped(ConversionSkip {
                name: name.to_string(),
                error: err.to_string(),
            }))
        }
        Err(e) => Err(e),
    }
}

fn warn_skip_pngs(path: &Path, skips: &[ConversionSkip]) {
    for skip in skips {
        eprintln!(
            "{} {}: {}: {}",
            color_msg("skipped", "33"),
            display_path(path),
            skip.name,
            skip.error
        );
    }
}

fn convert_png_bytes_basic(data: &[u8], format: ConvertFormat) -> Result<Vec<u8>, AppError> {
    let img = image::load_from_memory_with_format(data, ImageFormat::Png)?;
    let mut out = Vec::new();
    match format {
        ConvertFormat::Webp => {
            let rgba = img.to_rgba8();
            let encoder = WebpEncoder::from_rgba(rgba.as_raw(), rgba.width(), rgba.height());
            let mut config = WebPConfig::new()
                .map_err(|_| AppError::Invalid("Invalid WebP config".to_string()))?;
            // WHY: 速度優先のためlossy設定で品質とmethodを指定する
            config.lossless = 0;
            config.quality = 85.0;
            config.method = 5;
            let webp = encoder
                .encode_advanced(&config)
                .map_err(|e| AppError::Invalid(format!("{:?}", e)))?;
            out.extend_from_slice(&webp);
        }
        ConvertFormat::Jpeg => {
            let rgb = img.to_rgb8();
            let mut encoder = JpegEncoder::new_with_quality(&mut out, 90);
            encoder.encode(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                ExtendedColorType::Rgb8,
            )?;
        }
        ConvertFormat::JpegGpu => {
            let rgba = img.to_rgba8();
            out = convert_png_to_jpeg_gpu(&rgba, 90)?;
        }
        ConvertFormat::Jxl => {
            return Err(AppError::Invalid("JXL is not supported here".to_string()));
        }
    }
    Ok(out)
}

fn build_xmp_packet(chunks: &[(String, String)]) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    let mut text_body = String::new();
    let mut prompt_value = None;
    for (key, value) in chunks {
        if prompt_value.is_none() && key.eq_ignore_ascii_case("prompt") {
            prompt_value = Some(value.clone());
        }
        text_body.push_str("<civitai:text key=\"");
        text_body.push_str(&xml_escape(key));
        text_body.push_str("\">");
        text_body.push_str(&xml_escape(value));
        text_body.push_str("</civitai:text>");
    }
    let mut prompt_tag = String::new();
    if let Some(prompt) = prompt_value {
        prompt_tag.push_str("<prompt>");
        prompt_tag.push_str(&xml_escape(&prompt));
        prompt_tag.push_str("</prompt>");
    }
    format!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\
<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\
<rdf:Description xmlns:civitai=\"https://civitai.com/ns/1.0/\">{}{}\
</rdf:Description></rdf:RDF></x:xmpmeta>",
        prompt_tag, text_body
    )
}

fn xml_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn build_exif_user_comment(chunks: &[(String, String)]) -> Result<Vec<u8>, AppError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let mut map = serde_json::Map::new();
    for (key, value) in chunks {
        map.insert(key.clone(), serde_json::Value::String(value.clone()));
    }
    let json = serde_json::Value::Object(map).to_string();
    let field = exif::Field {
        tag: exif::Tag::UserComment,
        ifd_num: exif::In::PRIMARY,
        value: exif::Value::Ascii(vec![json.into_bytes()]),
    };
    let mut writer = exif::experimental::Writer::new();
    writer.push_field(&field);
    let mut buf = std::io::Cursor::new(Vec::new());
    writer.write(&mut buf, false)?;
    Ok(buf.into_inner())
}

fn embed_converted_metadata(
    data: &[u8],
    format: ConvertFormat,
    chunks: &[(String, String)],
) -> Result<Vec<u8>, AppError> {
    if chunks.is_empty() {
        return Ok(data.to_vec());
    }
    let xmp = build_xmp_packet(chunks);
    let exif = build_exif_user_comment(chunks)?;
    match format {
        ConvertFormat::Webp => embed_webp_metadata(data, &exif, &xmp),
        ConvertFormat::Jpeg | ConvertFormat::JpegGpu => embed_jpeg_metadata(data, &exif, &xmp),
        ConvertFormat::Jxl => Err(AppError::Invalid("JXL metadata is handled elsewhere".to_string())),
    }
}

fn embed_webp_metadata(data: &[u8], exif: &[u8], xmp: &str) -> Result<Vec<u8>, AppError> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(AppError::Invalid("Invalid WebP header".to_string()));
    }
    let mut chunks: Vec<([u8; 4], Vec<u8>)> = Vec::new();
    let mut offset = 12;
    while offset + 8 <= data.len() {
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&data[offset..offset + 4]);
        let size_bytes = &data[offset + 4..offset + 8];
        let size =
            u32::from_le_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]])
                as usize;
        let data_start = offset + 8;
        let data_end = data_start + size;
        if data_end > data.len() {
            break;
        }
        if &tag != b"EXIF" && &tag != b"XMP " {
            chunks.push((tag, data[data_start..data_end].to_vec()));
        }
        let aligned = size + (size % 2);
        offset = data_start + aligned;
    }
    if !exif.is_empty() {
        chunks.push((*b"EXIF", exif.to_vec()));
    }
    if !xmp.is_empty() {
        chunks.push((*b"XMP ", xmp.as_bytes().to_vec()));
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(b"WEBP");
    for (tag, chunk) in chunks {
        out.extend_from_slice(&tag);
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&chunk);
        if chunk.len() % 2 == 1 {
            out.push(0);
        }
    }
    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Ok(out)
}

fn embed_jpeg_metadata(data: &[u8], exif: &[u8], xmp: &str) -> Result<Vec<u8>, AppError> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(AppError::Invalid("Invalid JPEG header".to_string()));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&data[0..2]);
    if !exif.is_empty() {
        let mut payload = b"Exif\0\0".to_vec();
        payload.extend_from_slice(exif);
        out.extend_from_slice(&build_jpeg_app1_segment(&payload)?);
    }
    if !xmp.is_empty() {
        let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
        payload.extend_from_slice(xmp.as_bytes());
        out.extend_from_slice(&build_jpeg_app1_segment(&payload)?);
    }
    let mut offset = 2;
    while offset + 4 <= data.len() {
        if data[offset] != 0xFF {
            out.extend_from_slice(&data[offset..]);
            return Ok(out);
        }
        let marker = data[offset + 1];
        if marker == 0xDA {
            out.extend_from_slice(&data[offset..]);
            return Ok(out);
        }
        if marker == 0xD9 {
            out.extend_from_slice(&data[offset..offset + 2]);
            return Ok(out);
        }
        if (0xD0..=0xD7).contains(&marker) {
            out.extend_from_slice(&data[offset..offset + 2]);
            offset += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        if seg_len < 2 || offset + 2 + seg_len > data.len() {
            return Err(AppError::Invalid("Invalid JPEG segment".to_string()));
        }
        let seg_start = offset + 4;
        let seg_end = offset + 2 + seg_len;
        let payload = &data[seg_start..seg_end];
        if marker == 0xE1
            && (payload.starts_with(b"Exif\0\0")
                || payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0"))
        {
            offset += 2 + seg_len;
            continue;
        }
        out.extend_from_slice(&data[offset..offset + 2 + seg_len]);
        offset += 2 + seg_len;
    }
    Ok(out)
}

fn build_jpeg_app1_segment(payload: &[u8]) -> Result<Vec<u8>, AppError> {
    let size = payload.len() + 2;
    if size > u16::MAX as usize {
        return Err(AppError::Invalid("APP1 payload too large".to_string()));
    }
    let mut out = Vec::with_capacity(2 + 2 + payload.len());
    out.push(0xFF);
    out.push(0xE1);
    out.extend_from_slice(&(size as u16).to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn jxl_metadata_compress(text_chunks: &[(String, String)]) -> bool {
    if text_chunks.is_empty() {
        return false;
    }
    false
}

fn build_jxl_encoder(
    use_container: bool,
    parallel_runner: Option<&dyn ParallelRunner>,
) -> Result<JxlEncoder<'_, '_>, AppError> {
    let builder = jxl_encoder_builder()
        .has_alpha(true)
        .lossless(false)
        .quality(4.0)
        .use_container(use_container)
        .uses_original_profile(true)
        .speed(EncoderSpeed::Thunder);
    let encoder = if let Some(runner) = parallel_runner {
        builder.parallel_runner(runner).build()?
    } else {
        builder.build()?
    };
    Ok(encoder)
}

fn jxl_parallel_threads() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let half = cores / 2;
    if half == 0 { 1 } else { half }
}

fn convert_png_to_jxl(
    data: &[u8],
    text_chunks: &[(String, String)],
    has_json: bool,
) -> Result<ConversionOutcome, AppError> {
    let img = image::load_from_memory_with_format(data, ImageFormat::Png)?;
    let rgba = img.to_rgba8();
    if rgba.width() < 2 || rgba.height() < 2 {
        // WHY: jxlエンコーダの制約に合わせてPNGを維持する
        return Ok(ConversionOutcome::KeepOriginal);
    }
    // WHY: JXLのエンコードを高速化するためCPUコア数で固定並列にする
    let runner = ThreadsRunner::new(None, Some(jxl_parallel_threads()));
    let parallel_runner = runner.as_ref().map(|r| r as &dyn ParallelRunner);
    let mut encoder = build_jxl_encoder(!text_chunks.is_empty(), parallel_runner)?;
    if !text_chunks.is_empty() {
        let mut metadata_failed = false;
        // WHY: メタデータ圧縮はCPU負荷が高いため無圧縮にする
        let compress_metadata = jxl_metadata_compress(text_chunks);
        let xmp = build_xmp_packet(text_chunks);
        let exif = match build_exif_user_comment(text_chunks) {
            Ok(v) => v,
            Err(_) => {
                metadata_failed = true;
                Vec::new()
            }
        };
        if !metadata_failed {
            let exif_box = build_jxl_exif_box(&exif);
            if encoder
                .add_metadata(&JxlMetadata::Exif(&exif_box), compress_metadata)
                .is_err()
            {
                metadata_failed = true;
            } else if encoder
                .add_metadata(&JxlMetadata::Xmp(xmp.as_bytes()), compress_metadata)
                .is_err()
            {
                metadata_failed = true;
            }
        }
        if metadata_failed {
            if !has_json {
                return Ok(ConversionOutcome::KeepOriginal);
            }
            encoder = build_jxl_encoder(false, parallel_runner)?;
        }
    }
    let frame = EncoderFrame::new(rgba.as_raw()).num_channels(4);
    let result = encoder.encode_frame::<u8, u8>(&frame, rgba.width(), rgba.height())?;
    Ok(ConversionOutcome::Converted(result.data))
}

fn build_jxl_exif_box(exif: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + exif.len());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(exif);
    out
}


fn color_msg(msg: &str, code: &str) -> String {
    format!("\x1b[{}m{}\x1b[0m", code, msg)
}

fn keyword_key(keywords: &[String], convert: Option<ConvertFormat>) -> String {
    let mut list = keywords.to_vec();
    list.sort();
    let base = list.join(",");
    let convert_key = match convert {
        Some(ConvertFormat::Webp) => "convert=webp",
        Some(ConvertFormat::Jpeg) => "convert=jpg",
        Some(ConvertFormat::JpegGpu) => "convert=jpg_gpu",
        Some(ConvertFormat::Jxl) => "convert=jxl",
        None => "convert=none",
    };
    format!("{}|{}", base, convert_key)
}

fn cache_file_path() -> std::path::PathBuf {
    let base = env::var_os("DELETE_IMAGES_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| env::var_os("HOME").map(std::path::PathBuf::from))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join(".cache").join("delete_images_from_zips").join("processed.json")
}

fn load_cache(path: &std::path::Path) -> CacheMap {
    let data = match fs::read_to_string(&io_path(path)) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
}

fn save_cache(path: &std::path::Path, cache: &CacheMap) -> Result<(), AppError> {
    if let Some(dir) = path.parent() {
        io_ctx(dir, fs::create_dir_all(&io_path(dir)))?;
    }
    let data = serde_json::to_string(cache)?;
    io_ctx(path, fs::write(&io_path(path), data))?;
    Ok(())
}

fn cache_hit(cache: &CacheMap, keyword: &str, path: &Path, hash: &str) -> bool {
    cache
        .get(keyword)
        .and_then(|m| m.get(&display_path(path)))
        .map(|v| v == hash)
        .unwrap_or(false)
}

fn cache_insert(cache: &mut CacheMap, keyword: &str, path: &Path, hash: String) {
    cache
        .entry(keyword.to_string())
        .or_default()
        .insert(display_path(path), hash);
}

fn zip_hash(path: &Path, meta: &fs::Metadata) -> String {
    let mtime = FileTime::from_last_modification_time(meta);
    let mut hasher = Sha256::new();
    hasher.update(display_path(path).as_bytes());
    hasher.update(mtime.unix_seconds().to_le_bytes());
    hasher.update(mtime.nanoseconds().to_le_bytes());
    hasher.update(meta.len().to_le_bytes());
    let out = hasher.finalize();
    to_hex(&out)
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn dir_times(path: &Path) -> Option<(std::path::PathBuf, FileTime, FileTime)> {
    let dir = path.parent()?;
    let meta = fs::metadata(&io_path(dir)).ok()?;
    let atime = FileTime::from_last_access_time(&meta);
    let mtime = FileTime::from_last_modification_time(&meta);
    Some((dir.to_path_buf(), atime, mtime))
}

fn is_model_info(name: &str) -> bool {
    Path::new(name)
        .file_name()
        .map(|n| n == "model_info.json")
        == Some(true)
}

fn maybe_insert_model_safe(target: &mut HashSet<String>, name: &str) {
    if !is_model_info(name) {
        target.insert(name.to_string());
    }
}

fn count_json_entries(entries: &[EntryMeta]) -> u64 {
    entries
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::Json))
        .count() as u64
}

#[cfg(windows)]
fn io_path(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str.starts_with(r"\\?\") || path_str.starts_with(r"\\.\") {
        return path.to_path_buf();
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let normalized = normalize_windows_path(&absolute);
    let normalized_str = normalized.to_string_lossy();
    if let Some(stripped) = normalized_str.strip_prefix(r"\\") {
        return PathBuf::from(format!(r"\\?\UNC\{}", stripped));
    }
    PathBuf::from(format!(r"\\?\{}", normalized_str))
}

#[cfg(not(windows))]
fn io_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(windows)]
fn normalize_windows_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut prefix: Option<String> = None;
    let mut has_root = false;
    let mut parts: Vec<String> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => prefix = Some(p.as_os_str().to_string_lossy().to_string()),
            Component::RootDir => has_root = true,
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = parts.pop();
            }
            Component::Normal(s) => parts.push(s.to_string_lossy().to_string()),
        }
    }

    let mut out = String::new();
    if let Some(prefix) = prefix {
        out.push_str(&prefix);
    }
    if has_root && !out.ends_with('\\') {
        out.push('\\');
    }
    for part in parts {
        if !out.ends_with('\\') {
            out.push('\\');
        }
        out.push_str(&part);
    }
    PathBuf::from(out)
}

#[cfg(windows)]
fn display_path(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    if let Some(stripped) = path_str.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{}", stripped);
    }
    if let Some(stripped) = path_str.strip_prefix(r"\\?\") {
        return stripped.to_string();
    }
    if let Some(stripped) = path_str.strip_prefix(r"\\.\") {
        return stripped.to_string();
    }
    path_str.to_string()
}

#[cfg(not(windows))]
fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn io_ctx<T>(path: &Path, res: std::io::Result<T>) -> Result<T, AppError> {
    res.map_err(|e| AppError::IoWithPath {
        path: display_path(path),
        source: e,
    })
}

fn set_times(path: &Path, atime: FileTime, mtime: FileTime) -> Result<(), AppError> {
    set_file_times(&io_path(path), atime, mtime).map_err(|e| AppError::IoWithPath {
        path: display_path(path),
        source: e,
    })
}

fn init_rayon_threads(threads: usize) -> Result<(), AppError> {
    static INIT: OnceLock<()> = OnceLock::new();
    if INIT.get().is_some() {
        return Ok(());
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        Ok(()) => {
            let _ = INIT.set(());
            Ok(())
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("already been initialized") {
                let _ = INIT.set(());
                Ok(())
            } else {
                Err(AppError::Invalid(msg))
            }
        }
    }
}

fn path_len_warning(path: &Path) -> Option<String> {
    let _ = path;
    // WHY: パス長警告は出さない方針に変更した
    None
}

fn warn_if_path_long(path: &Path) {
    if let Some(msg) = path_len_warning(path) {
        eprintln!("{}", msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;
    use webp::{BitstreamFeatures, BitstreamFormat};
    use zip::CompressionMethod;
    use zip::write::FileOptions;
    use zip::ZipWriter;

    static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_MUTEX.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn test_progress_info(path: &Path) -> ProgressInfo {
        ProgressInfo {
            name: file_display_name(path),
            index: 1,
            total: 1,
            term_width: 80,
            start: Instant::now(),
            scroll_wait: Duration::from_secs(0),
            scroll_interval: Duration::from_millis(300),
            base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
        }
    }

    #[test]
    fn interrupts_run_when_cancel_set() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer.write_all(&make_png_with_prompt("cat")).unwrap();
        writer.finish().unwrap();

        let keywords = parse_keywords("dog");
        let key = keyword_key(&keywords, None);
        let cancel = AtomicBool::new(true);
        let result = run_with_cancel(dir.path(), &key, &keywords, false, None, &cancel);
        assert!(matches!(result, Err(AppError::Interrupted)));
    }

    #[test]
    fn converts_png_to_webp_even_without_deletion() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[
                ("prompt", "cat"),
                ("workflow", "flow"),
            ]))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Webp),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.webp".to_string()));
        assert!(!names.contains(&"a.png".to_string()));

        let mut webp = archive.by_name("a.webp").unwrap();
        let webp_bytes = read_zipfile_bytes(&mut webp).unwrap();
        let xmp = extract_webp_chunk(&webp_bytes, b"XMP ").unwrap();
        let xmp_text = String::from_utf8_lossy(&xmp).to_string();
        assert!(xmp_text.contains("workflow"));
        assert!(xmp_text.contains("prompt"));

        let exif = extract_webp_chunk(&webp_bytes, b"EXIF").unwrap();
        let exif_map = exif_user_comment_map(&exif);
        assert_eq!(exif_map["prompt"], "cat");
        assert_eq!(exif_map["workflow"], "flow");

        let features = BitstreamFeatures::new(&webp_bytes).unwrap();
        assert!(matches!(features.format(), Some(BitstreamFormat::Lossy)));
    }

    #[test]
    fn converts_multiple_pngs_to_webp() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[("prompt", "cat")]))
            .unwrap();
        writer.start_file("b.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[("prompt", "dog")]))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["bird".into()],
            false,
            &cancel,
            Some(ConvertFormat::Webp),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.webp".to_string()));
        assert!(names.contains(&"b.webp".to_string()));
        assert!(!names.contains(&"a.png".to_string()));
        assert!(!names.contains(&"b.png".to_string()));
    }

    #[test]
    fn converts_png_to_jpg_overwrites_existing() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[("prompt", "cat")]))
            .unwrap();
        writer.start_file("a.jpg", opts).unwrap();
        writer.write_all(&[1, 2, 3, 4]).unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Jpeg),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.jpg".to_string()));
        assert!(!names.contains(&"a.png".to_string()));

        let mut jpg = archive.by_name("a.jpg").unwrap();
        let jpg_bytes = read_zipfile_bytes(&mut jpg).unwrap();
        assert_ne!(jpg_bytes, vec![1, 2, 3, 4]);
        assert!(jpg_bytes.len() > 4);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn converts_png_to_jpg_gpu_on_windows() {
        let png = make_png_with_text_chunks(&[("prompt", "cat")]);
        let jpg = convert_png_bytes_basic(&png, ConvertFormat::JpegGpu).unwrap();
        assert!(jpg.len() > 3);
        assert!(jpg.as_slice().starts_with(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn converts_png_to_jxl_even_without_deletion() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks_size(
                &[("prompt", "cat"), ("workflow", "flow")],
                2,
                2,
            ))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Jxl),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.jxl".to_string()));
        assert!(!names.contains(&"a.png".to_string()));
    }

    #[test]
    fn keeps_png_when_jxl_is_too_small() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[("prompt", "cat")]))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Jxl),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.png".to_string()));
        assert!(!names.contains(&"a.jxl".to_string()));
    }

    #[test]
    fn jxl_metadata_compress_is_disabled_for_speed() {
        let chunks = vec![("prompt".to_string(), "cat".to_string())];
        assert!(!jxl_metadata_compress(&chunks));
    }

    #[test]
    fn jxl_parallel_threads_uses_available_parallelism() {
        let cores = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let expected = if cores / 2 == 0 { 1 } else { cores / 2 };
        assert_eq!(jxl_parallel_threads(), expected);
    }

    #[test]
    fn skips_invalid_png_conversion_and_keeps_original() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        let mut broken_png = make_png_with_prompt("cat");
        broken_png.truncate(broken_png.len() / 2);

        writer.start_file("a.png", opts).unwrap();
        writer.write_all(&broken_png).unwrap();
        writer.start_file("a.json", opts).unwrap();
        writer.write_all(r#"{"prompt":"cat"}"#.as_bytes()).unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Webp),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(!names.contains(&"a.png".to_string()));
        assert!(!names.contains(&"a.webp".to_string()));
        assert!(!names.contains(&"a.json".to_string()));
    }

    #[test]
    fn keeps_png_when_metadata_embed_fails_without_json() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let big = "a".repeat(70000);
        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_text_chunks(&[("prompt", &big)]))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["dog".into()],
            false,
            &cancel,
            Some(ConvertFormat::Jpeg),
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"a.png".to_string()));
        assert!(!names.contains(&"a.jpg".to_string()));
    }

    #[test]
    fn deletes_group_when_json_prompt_matches_keyword() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("a.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat,dog"}"#.as_bytes())
            .unwrap();
        writer.start_file("a.png", opts).unwrap();
        writer
            .write_all(&make_png_with_prompt("other"))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn deletes_json_and_image_when_prompt_absent_in_both() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("b.json", opts).unwrap();
        writer
            .write_all(r#"{"title":"no prompt"}"#.as_bytes())
            .unwrap();
        writer.start_file("b.png", opts).unwrap();
        writer.write_all(&make_png_without_prompt()).unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn keeps_files_when_image_has_prompt_even_if_json_missing_prompt() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("c.json", opts).unwrap();
        writer
            .write_all(r#"{"title":"no prompt"}"#.as_bytes())
            .unwrap();
        writer.start_file("c.png", opts).unwrap();
        writer
            .write_all(&make_png_with_prompt("keeps"))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(!changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn keeps_files_when_json_prompt_non_matching_even_if_image_matches() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("z.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.start_file("z.png", opts).unwrap();
        writer
            .write_all(&make_png_with_prompt("dog"))
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(!changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn deletes_when_json_prompt_matches_even_if_image_first() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("q.png", opts).unwrap();
        writer
            .write_all(&make_png_with_prompt("cat"))
            .unwrap();
        writer.start_file("q.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"dog"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn zip_error_shows_red_path() {
        let err = AppError::ZipWithPath {
            path: "bad.zip".into(),
            source: zip::result::ZipError::FileNotFound,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("\u{1b}[91mbad.zip\u{1b}[0m"));
    }

    #[test]
    fn deletes_when_prompt_is_under_meta() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);

        writer.start_file("m.json", opts).unwrap();
        writer
            .write_all(r#"{ "meta": { "prompt": "anthro,cat" } }"#.as_bytes())
            .unwrap();
        writer.start_file("m.jpeg", opts).unwrap();
        writer.write_all(&[0u8; 4]).unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed = process_zip(
            &zip_path,
            &progress_info,
            &vec!["anthro".into()],
            false,
            &cancel,
            None,
        )
        .unwrap();
        assert!(changed);

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn preserves_timestamp_after_rewrite() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("sample.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("d.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"bird"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let meta = fs::metadata(&zip_path).unwrap();
        let mtime_before = FileTime::from_last_modification_time(&meta);
        let ctime_before = FileTime::from_creation_time(&meta).unwrap_or(mtime_before);

        std::thread::sleep(std::time::Duration::from_millis(10));
        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["cat".into()], false, &cancel, None).unwrap();
        assert!(!changed);

        let meta_after = fs::metadata(&zip_path).unwrap();
        let mtime_after = FileTime::from_last_modification_time(&meta_after);
        let ctime_after = FileTime::from_creation_time(&meta_after).unwrap_or(mtime_after);

        assert_eq!(mtime_before, mtime_after);
        assert_eq!(ctime_before, ctime_after);
    }

    #[test]
    fn run_with_progress_bar_completes() {
        let _guard = env_guard();
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("p.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("p.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        run(dir.path(), "dog", true, None).unwrap();
    }

    #[test]
    fn normalize_removes_emphasis_and_weights() {
        let tags = normalize_prompt("((Hoge)), (foo:1.5), [bar:2], {baz}, ( red fox :2.0)");
        assert_eq!(tags, vec!["hoge", "foo", "bar", "baz", "red fox"]);
    }

    #[test]
    fn color_msg_wraps_with_ansi_code() {
        assert_eq!(
            color_msg("reading", "36"),
            "\u{1b}[36mreading\u{1b}[0m".to_string()
        );
    }

    #[test]
    fn keeps_model_info_even_when_prompt_matches() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("model.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("a.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.start_file("model_info.json", opts).unwrap();
        writer
            .write_all(r#"{"meta":"keep"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &progress_info, &vec!["cat".into()], false, &cancel, None).unwrap();

        let file = File::open(&zip_path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 1);
        let kept = archive.by_index(0).unwrap();
        assert_eq!(kept.name(), "model_info.json");
    }

    #[test]
    fn skips_rewrite_when_no_deletions() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("skip.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("x.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let meta_before = fs::metadata(&zip_path).unwrap();
        let mtime_before = FileTime::from_last_modification_time(&meta_before);

        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["dog".into()], false, &cancel, None).unwrap();
        assert!(!changed);

        let meta_after = fs::metadata(&zip_path).unwrap();
        let mtime_after = FileTime::from_last_modification_time(&meta_after);
        assert_eq!(mtime_before, mtime_after);
    }

    #[test]
    fn preserves_directory_mtime_after_tmp_creation() {
        let dir = tempdir().unwrap();
        let zip_path = dir.path().join("dirtime.zip");

        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("e.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let meta_before = fs::metadata(dir.path()).unwrap();
        let mtime_before = FileTime::from_last_modification_time(&meta_before);

        std::thread::sleep(std::time::Duration::from_millis(10));
        let progress_info = test_progress_info(&zip_path);
        let cancel = AtomicBool::new(false);
        let changed =
            process_zip(&zip_path, &progress_info, &vec!["cat".into()], false, &cancel, None).unwrap();
        assert!(changed);

        let meta_after = fs::metadata(dir.path()).unwrap();
        let mtime_after = FileTime::from_last_modification_time(&meta_after);
        assert_eq!(mtime_before, mtime_after);
    }

    #[test]
    fn cache_skips_same_zip_and_keyword() {
        let dir = tempdir().unwrap();

        let zip_path = dir.path().join("x.zip");
        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("a.json", opts).unwrap();
        writer.write_all(r#"{"prompt":"cat"}"#.as_bytes()).unwrap();
        writer.finish().unwrap();

        let meta = fs::metadata(&zip_path).unwrap();
        let hash = zip_hash(&zip_path, &meta);
        let mut cache = CacheMap::new();
        cache_insert(&mut cache, "cat", &zip_path, hash);

        assert!(cache_hit(&cache, "cat", &zip_path, &zip_hash(&zip_path, &meta)));
        assert!(!cache_hit(&cache, "dog", &zip_path, &zip_hash(&zip_path, &meta)));
    }

    #[test]
    fn count_json_entries_counts_only_json() {
        let entries = vec![
            EntryMeta {
                name: "a.json".to_string(),
                stem: "a".to_string(),
                kind: EntryKind::Json,
            },
            EntryMeta {
                name: "b.png".to_string(),
                stem: "b".to_string(),
                kind: EntryKind::Png,
            },
            EntryMeta {
                name: "c.json".to_string(),
                stem: "c".to_string(),
                kind: EntryKind::Json,
            },
        ];
        assert_eq!(count_json_entries(&entries), 2);
    }

    #[test]
    fn octet_stream_is_treated_as_png() {
        let meta = entry_meta_from_name("a.octet-stream").unwrap();
        assert!(matches!(meta.kind, EntryKind::Png));
    }

    #[test]
    fn clear_cache_file_removes_existing() {
        let _guard = env_guard();
        let dir = tempdir().unwrap();
        let prev = env::var("DELETE_IMAGES_CACHE_HOME").ok();
        unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", dir.path()) };

        let cache = dir
            .path()
            .join(".cache/delete_images_from_zips/processed.json");
        fs::create_dir_all(cache.parent().unwrap()).unwrap();
        fs::write(&cache, "{}").unwrap();

        clear_cache_file().unwrap();
        assert!(!cache.exists());

        match prev {
            Some(v) => unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", v) },
            None => unsafe { env::remove_var("DELETE_IMAGES_CACHE_HOME") },
        }
    }

    #[test]
    fn clear_cache_file_ok_when_missing() {
        let _guard = env_guard();
        let dir = tempdir().unwrap();
        let prev = env::var("DELETE_IMAGES_CACHE_HOME").ok();
        unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", dir.path()) };

        clear_cache_file().unwrap();

        match prev {
            Some(v) => unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", v) },
            None => unsafe { env::remove_var("DELETE_IMAGES_CACHE_HOME") },
        }
    }

    #[test]
    fn file_display_name_uses_file_name() {
        let path = Path::new("/tmp/sample.zip");
        assert_eq!(file_display_name(path), "sample.zip");
    }

    #[test]
    fn init_rayon_threads_is_idempotent() {
        init_rayon_threads(4).unwrap();
        init_rayon_threads(4).unwrap();
    }

    #[test]
    fn scrolling_name_keeps_short_text() {
        assert_eq!(scrolling_name("abc", 10, 0), "abc");
    }

    #[test]
    fn scrolling_name_wraps_from_right() {
        assert_eq!(scrolling_name("abcdef", 3, 5), "f  ");
    }

    #[test]
    fn progress_prefix_uses_half_width() {
        let info = ProgressInfo {
            name: "0123456789".to_string(),
            index: 3,
            total: 9,
            term_width: 10,
            start: Instant::now(),
            scroll_wait: Duration::from_secs(0),
            scroll_interval: Duration::from_millis(300),
            base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
        };
        let prefix = progress_prefix(&info, 0);
        assert_eq!(prefix, "3/9 01234");
    }

    #[test]
    fn convert_extension_uses_jpg_for_gpu() {
        assert_eq!(convert_extension(ConvertFormat::JpegGpu), "jpg");
    }

    #[test]
    fn scroll_offset_waits_then_starts_from_zero() {
        let info = ProgressInfo {
            name: "name.zip".to_string(),
            index: 1,
            total: 1,
            term_width: 80,
            start: Instant::now(),
            scroll_wait: Duration::from_secs(2),
            scroll_interval: Duration::from_millis(300),
            base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
        };

        let before = scroll_offset(&info, Duration::from_secs(1));
        assert_eq!(before, 0);

        let at_start = scroll_offset(&info, Duration::from_secs(2));
        assert_eq!(at_start, 0);

        let after = scroll_offset(&info, Duration::from_millis(2300));
        assert_eq!(after, 1);
    }

    #[test]
    fn scroll_offset_advances_by_interval() {
        let info = ProgressInfo {
            name: "name.zip".to_string(),
            index: 1,
            total: 1,
            term_width: 80,
            start: Instant::now(),
            scroll_wait: Duration::from_secs(0),
            scroll_interval: Duration::from_millis(300),
            base_elapsed_ms: AtomicU64::new(SCROLL_BASE_UNSET),
        };
        let first = scroll_offset(&info, Duration::from_millis(300));
        let second = scroll_offset(&info, Duration::from_millis(900));
        assert_eq!(first, 0);
        assert_eq!(second, 2);
    }

    #[cfg(not(windows))]
    #[test]
    fn file_display_name_falls_back_unix_root() {
        let path = Path::new("/");
        assert_eq!(file_display_name(path), "/");
    }

    #[test]
    fn keyword_key_uses_jpg_gpu() {
        let keywords = vec!["cat".to_string()];
        let key = keyword_key(&keywords, Some(ConvertFormat::JpegGpu));
        assert!(key.contains("convert=jpg_gpu"));
    }

    #[cfg(windows)]
    #[test]
    fn file_display_name_falls_back_windows_root() {
        let path = Path::new("C:\\");
        assert_eq!(file_display_name(path), "C:\\");
    }

    #[test]
    fn run_skips_errors_and_reports() {
        let _guard = env_guard();
        let dir = tempdir().unwrap();
        let prev = env::var("DELETE_IMAGES_CACHE_HOME").ok();
        unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", dir.path()) };

        let good = dir.path().join("good.zip");
        let mut zip_file = File::create(&good).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("a.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.finish().unwrap();

        let bad = dir.path().join("bad.zip");
        fs::write(&bad, b"not a zip").unwrap();

        let result = run(dir.path(), "cat", false, None);
        assert!(result.is_err());

        let cache = load_cache(&cache_file_path());
        let keyword_key = keyword_key(&vec!["cat".into()], None);
        let key = good.display().to_string();
        assert!(cache
            .get(&keyword_key)
            .and_then(|m| m.get(&key))
            .is_some());

        match prev {
            Some(v) => unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", v) },
            None => unsafe { env::remove_var("DELETE_IMAGES_CACHE_HOME") },
        }
    }

    #[test]
    fn run_updates_cache_hash_after_write() {
        let _guard = env_guard();
        let dir = tempdir().unwrap();
        let prev = env::var("DELETE_IMAGES_CACHE_HOME").ok();
        unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", dir.path()) };

        let zip_path = dir.path().join("rewrite.zip");
        let mut zip_file = File::create(&zip_path).unwrap();
        let mut writer = ZipWriter::new(&mut zip_file);
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        writer.start_file("a.json", opts).unwrap();
        writer
            .write_all(r#"{"prompt":"cat"}"#.as_bytes())
            .unwrap();
        writer.start_file("a.png", opts).unwrap();
        writer.write_all(&make_png_with_prompt("other")).unwrap();
        writer.finish().unwrap();

        let meta_before = fs::metadata(&zip_path).unwrap();
        let hash_before = zip_hash(&zip_path, &meta_before);

        run(dir.path(), "cat", false, None).unwrap();

        let meta_after = fs::metadata(&zip_path).unwrap();
        let hash_after = zip_hash(&zip_path, &meta_after);

        let cache = load_cache(&cache_file_path());
        let keyword_key = keyword_key(&vec!["cat".into()], None);
        let key = zip_path.display().to_string();
        let cached = cache
            .get(&keyword_key)
            .and_then(|m| m.get(&key))
            .cloned();

        assert_eq!(cached, Some(hash_after));
        assert_ne!(cached, Some(hash_before));

        match prev {
            Some(v) => unsafe { env::set_var("DELETE_IMAGES_CACHE_HOME", v) },
            None => unsafe { env::remove_var("DELETE_IMAGES_CACHE_HOME") },
        }
    }


    #[test]
    fn path_len_warning_returns_none_for_short_path() {
        assert!(path_len_warning(Path::new("C:\\short.zip")).is_none());
    }

    #[cfg(windows)]
    #[test]
    fn path_len_warning_returns_none_for_long_path() {
        let long_name = "C:\\".to_string() + &"a".repeat(250) + ".zip";
        let p = Path::new(&long_name);
        assert!(path_len_warning(p).is_none());
    }

    #[cfg(windows)]
    #[test]
    fn io_path_adds_verbatim_prefix_for_absolute() {
        let path = Path::new(r"C:\temp\sample.zip");
        let long = io_path(path);
        let long_str = long.to_string_lossy();
        assert!(long_str.starts_with(r"\\?\"));
    }

    #[cfg(windows)]
    #[test]
    fn display_path_strips_verbatim_prefix() {
        let path = Path::new(r"\\?\C:\temp\sample.zip");
        assert_eq!(display_path(path), r"C:\temp\sample.zip");
    }

    fn exif_user_comment_map(exif_data: &[u8]) -> serde_json::Value {
        let mut cursor = std::io::Cursor::new(exif_data);
        let exif = exif::Reader::new().read_from_container(&mut cursor).unwrap();
        for f in exif.fields() {
            if f.tag == exif::Tag::UserComment {
                if let exif::Value::Ascii(ref vec) = f.value {
                    if let Some(bytes) = vec.get(0) {
                        let s = String::from_utf8_lossy(bytes).to_string();
                        return serde_json::from_str(&s).unwrap();
                    }
                }
            }
        }
        serde_json::Value::Null
    }

    fn make_png_with_text_chunks(chunks: &[(&str, &str)]) -> Vec<u8> {
        let width = 1;
        let height = 1;
        make_png_with_text_chunks_size(chunks, width, height)
    }

    fn make_png_with_text_chunks_size(chunks: &[(&str, &str)], width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        for (key, value) in chunks {
            encoder
                .add_text_chunk((*key).to_string(), (*value).to_string())
                .unwrap();
        }
        let mut writer = encoder.write_header().unwrap();
        let total = (width * height * 3) as usize;
        writer.write_image_data(&vec![0; total]).unwrap();
        writer.finish().unwrap();
        buf
    }

    fn make_png_with_prompt(prompt: &str) -> Vec<u8> {
        let width = 1;
        let height = 1;
        let mut buf = Vec::new();
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.add_text_chunk("prompt".into(), prompt.into()).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0, 0, 0]).unwrap();
        writer.finish().unwrap();
        buf
    }

    fn make_png_without_prompt() -> Vec<u8> {
        let width = 1;
        let height = 1;
        let mut buf = Vec::new();
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0, 0, 0]).unwrap();
        writer.finish().unwrap();
        buf
    }
}
