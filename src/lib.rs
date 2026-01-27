use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use filetime::{set_file_times, FileTime};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use sha2::{Digest, Sha256};
use signal_hook::consts::SIGINT;
use signal_hook::flag as signal_flag;
use thiserror::Error;
use walkdir::WalkDir;
use zip::read::ZipFile;
use zip::ZipArchive;
use tempfile::Builder;

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
    Other,
}


pub fn run(root: &Path, keywords_csv: &str, progress: bool) -> Result<(), AppError> {
    let keywords = parse_keywords(keywords_csv);
    let keyword_key = keyword_key(&keywords);
    let cancel = Arc::new(AtomicBool::new(false));
    signal_flag::register(SIGINT, cancel.clone()).map_err(|e| AppError::Invalid(e.to_string()))?;
    let cache_path = cache_file_path();
    let mut cache = load_cache(&cache_path);
    let zip_paths: Vec<_> = WalkDir::new(root)
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

    if progress {
        eprintln!("Found {} zip files", zip_paths.len());
    }

    for path in zip_paths.iter() {
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::Interrupted);
        }
        let meta = io_ctx(path, fs::metadata(path))?;
        let zip_hash = zip_hash(path, &meta);
        if cache_hit(&cache, &keyword_key, path, &zip_hash) {
            if progress {
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string());
                eprintln!(
                    "{} {}",
                    color_msg(&name, "1;37"),
                    color_msg("skipped (cached)", "90")
                );
            }
            continue;
        }
        if progress {
            eprintln!();
        }
        process_zip(path, &keywords, progress, &cancel)?;
        cache_insert(&mut cache, &keyword_key, path, zip_hash);
        save_cache(&cache_path, &cache)?;
    }
    Ok(())
}

fn process_zip(
    path: &Path,
    keywords: &[String],
    progress: bool,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    let meta = io_ctx(path, fs::metadata(path))?;
    let atime = FileTime::from_last_access_time(&meta);
    let mtime = FileTime::from_last_modification_time(&meta);
    let dir_times = dir_times(path);

    // 読み取りフェーズ
    let (entries, read_bar) = {
        let file = io_ctx(path, File::open(path))?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
        let _total = archive.len() as u64;

        let read_bar = if progress {
            let b = ProgressBar::new(_total.max(1));
            // WHY: 進捗の視認性を優先
            b.set_style(
                ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());
            b.set_prefix(color_msg(&name, "1;37"));
            b.set_message(color_msg("scanning", "36"));
            Some(b)
        } else {
            None
        };

        let mut list = Vec::new();
        for i in 0..archive.len() {
            let file = archive
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            list.push(entry_meta_from_name(file.name())?);
            if let Some(ref b) = read_bar {
                b.inc(1);
            }
        }
        (list, read_bar)
    };

    let deletions = decide_deletions(
        path,
        &entries,
        entries.len() as u64,
        keywords,
        read_bar.as_ref(),
        cancel,
    )?;
    if deletions.is_empty() {
        if let Some(ref b) = read_bar {
            finalize_no_deletions(b, entries.len() as u64);
        }
    set_times(path, atime, mtime)?;
        return Ok(());
    }
    if let Some(ref b) = read_bar {
        let kept_len = entries.len() as u64 - deletions.len() as u64;
        b.set_length(kept_len.max(1));
        b.set_position(0);
        b.set_message(color_msg("writing", "32"));
    }

    write_filtered_zip(
        path,
        &entries,
        &deletions,
        read_bar.as_ref(),
        cancel,
        atime,
        mtime,
    )?;

    if !deletions.is_empty() {
        // WHY: 書き換えが発生したときだけ元の時刻を復元して変更検知を抑制
        set_times(path, atime, mtime)?;
    }
    if let Some((dir_path, dir_atime, dir_mtime)) = dir_times {
        // WHY: tmp作成やrenameで更新されたディレクトリ日時を元に戻す
        let _ = set_times(&dir_path, dir_atime, dir_mtime);
    }
    Ok(())
}

fn write_filtered_zip(
    path: &Path,
    entries: &[EntryMeta],
    deletions: &HashSet<String>,
    bar: Option<&ProgressBar>,
    cancel: &AtomicBool,
    atime: FileTime,
    mtime: FileTime,
) -> Result<(), AppError> {
    let kept_len = entries.len().saturating_sub(deletions.len());

    let dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let orig_permissions = io_ctx(path, fs::metadata(path))?.permissions();

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
        let mut src = ZipArchive::new(io_ctx(path, File::open(path))?)
            .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            if deletions.contains(file.name()) {
                continue;
            }
            writer.raw_copy_file(file)?;
            if let Some(b) = bar {
                b.inc(1);
            }
        }
        writer.finish()?;
    }

    if let Some(b) = bar {
        // WHY: syncやタイムスタンプ復元など後処理中であることを示すため
        b.set_message(color_msg("finalizing", "90"));
        b.tick();
    }

    // WHY: 書き込み後のfsyncコストを避けるため、renameベースで差し替える
    io_ctx(tmp.path(), fs::set_permissions(tmp.path(), orig_permissions.clone()))?;

    if path.exists() {
        // WHY: Windows上書き問題回避
        io_ctx(path, fs::remove_file(path))?;
    }
    let persist_result = tmp.persist(path);
    if let Err(err) = persist_result {
        return Err(AppError::IoWithPath {
            path: path.display().to_string(),
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
    Ok(())
}

fn decide_deletions(
    path: &Path,
    entries: &[EntryMeta],
    base_len: u64,
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
            EntryKind::Png | EntryKind::Webp => {
                image_name_to_meta.insert(e.name.clone(), (e.stem.clone(), e.kind));
            }
            EntryKind::Other => {}
        }
    }

    let json_count = json_name_to_stem.len() as u64;

    let mut json_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    if json_count > 0 {
        let mut src = ZipArchive::new(io_ctx(path, File::open(path))?)
            .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let mut file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
            if file.name().ends_with('/') {
                continue;
            }
            let stem = match json_name_to_stem.get(file.name()) {
                Some(v) => v.clone(),
                None => continue,
            };
            let prompt = json_prompt_tags_from_reader(&mut file)?;
            json_prompts.insert(stem, prompt);
            if let Some(b) = bar {
                b.inc(1);
            }
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
        let new_len = base_len + json_count + image_names.len() as u64;
        b.set_length(new_len.max(1));
        b.set_style(
            ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        if b.position() < base_len {
            // WHY: 既存の読み取り進捗を維持するため
            b.set_position(base_len);
        }
    }

    let mut image_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    if !image_names.is_empty() {
        let image_name_set: HashSet<String> = image_names.into_iter().collect();
        let mut src = ZipArchive::new(io_ctx(path, File::open(path))?)
            .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
        for i in 0..src.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AppError::Interrupted);
            }
            let mut file = src
                .by_index(i)
                .map_err(|e| AppError::ZipWithPath { path: path.display().to_string(), source: e })?;
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
                _ => None,
            };
            image_prompts.insert(stem, prompt);
            if let Some(b) = bar {
                b.inc(1);
            }
        }
    }

    if let Some(b) = bar {
        // WHY: 解析完了をユーザに即時示すため
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
        Some("png") => EntryKind::Png,
        Some("webp") => EntryKind::Webp,
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

fn extract_exif_prompt(data: &[u8]) -> Result<Option<String>, AppError> {
    let mut cursor = std::io::Cursor::new(data);
    match exif::Reader::new().read_from_container(&mut cursor) {
        Ok(exif) => {
            for f in exif.fields() {
                if f.tag == exif::Tag::UserComment {
                    if let exif::Value::Ascii(ref vec) = f.value {
                        if let Some(bytes) = vec.get(0) {
                            let s = String::from_utf8_lossy(bytes).to_string();
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
    if let Some(start) = content.find("<prompt>") {
        if let Some(end) = content[start + 8..].find("</prompt>") {
            let val = &content[start + 8..start + 8 + end];
            return Some(val.to_string());
        }
    }
    None
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

fn color_msg(msg: &str, code: &str) -> String {
    format!("\x1b[{}m{}\x1b[0m", code, msg)
}

fn keyword_key(keywords: &[String]) -> String {
    let mut list = keywords.to_vec();
    list.sort();
    list.join(",")
}

fn cache_file_path() -> std::path::PathBuf {
    let home = env::var_os("HOME").map(std::path::PathBuf::from);
    let base = home.unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join(".cache").join("delete_images_from_zips").join("processed.json")
}

fn load_cache(path: &std::path::Path) -> CacheMap {
    let data = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
}

fn save_cache(path: &std::path::Path, cache: &CacheMap) -> Result<(), AppError> {
    if let Some(dir) = path.parent() {
        io_ctx(dir, fs::create_dir_all(dir))?;
    }
    let data = serde_json::to_string(cache)?;
    io_ctx(path, fs::write(path, data))?;
    Ok(())
}

fn cache_hit(cache: &CacheMap, keyword: &str, path: &Path, hash: &str) -> bool {
    cache
        .get(keyword)
        .and_then(|m| m.get(&path.display().to_string()))
        .map(|v| v == hash)
        .unwrap_or(false)
}

fn cache_insert(cache: &mut CacheMap, keyword: &str, path: &Path, hash: String) {
    cache
        .entry(keyword.to_string())
        .or_default()
        .insert(path.display().to_string(), hash);
}

fn zip_hash(path: &Path, meta: &fs::Metadata) -> String {
    let mtime = FileTime::from_last_modification_time(meta);
    let mut hasher = Sha256::new();
    hasher.update(path.display().to_string().as_bytes());
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
    let meta = fs::metadata(dir).ok()?;
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

fn finalize_no_deletions(bar: &ProgressBar, scanned_len: u64) {
    bar.set_length(scanned_len.max(1));
    bar.set_position(scanned_len);
    bar.set_message(color_msg("scanned (no changes)", "90"));
    bar.finish_with_message(color_msg("done", "32"));
}

fn io_ctx<T>(path: &Path, res: std::io::Result<T>) -> Result<T, AppError> {
    res.map_err(|e| AppError::IoWithPath {
        path: path.display().to_string(),
        source: e,
    })
}

fn set_times(path: &Path, atime: FileTime, mtime: FileTime) -> Result<(), AppError> {
    set_file_times(path, atime, mtime).map_err(|e| AppError::IoWithPath {
        path: path.display().to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    use zip::CompressionMethod;
    use zip::write::FileOptions;
    use zip::ZipWriter;

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["anthro".into()], false, &cancel).unwrap();

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
        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["cat".into()], false, &cancel).unwrap();

        let meta_after = fs::metadata(&zip_path).unwrap();
        let mtime_after = FileTime::from_last_modification_time(&meta_after);
        let ctime_after = FileTime::from_creation_time(&meta_after).unwrap_or(mtime_after);

        assert_eq!(mtime_before, mtime_after);
        assert_eq!(ctime_before, ctime_after);
    }

    #[test]
    fn run_with_progress_bar_completes() {
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

        run(dir.path(), "dog", true).unwrap();
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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["cat".into()], false, &cancel).unwrap();

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

        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["dog".into()], false, &cancel).unwrap();

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
        let cancel = AtomicBool::new(false);
        process_zip(&zip_path, &vec!["cat".into()], false, &cancel).unwrap();

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
    fn finalize_no_deletions_keeps_length() {
        let bar = ProgressBar::new(5);
        bar.set_position(0);
        finalize_no_deletions(&bar, 5);
        assert_eq!(bar.length(), Some(5));
        assert_eq!(bar.position(), 5);
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
