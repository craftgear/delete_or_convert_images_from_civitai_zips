use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use filetime::{set_file_times, FileTime};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use thiserror::Error;
use walkdir::WalkDir;
use zip::read::ZipFile;
use zip::write::FileOptions;
use zip::ZipArchive;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("EXIF parse error: {0}")]
    Exif(#[from] exif::Error),
    #[error("Invalid data: {0}")]
    Invalid(String),
}

#[derive(Clone)]
struct EntryData {
    name: String,
    data: Vec<u8>,
    options: FileOptions,
}

#[derive(Clone, Copy)]
enum EntryKind {
    Json,
    Png,
    Webp,
    Other,
}

#[derive(Clone)]
struct EntryAnalysis {
    name: String,
    stem: String,
    kind: EntryKind,
    prompt_tags: Option<Vec<String>>, // None means no prompt, Some(vec) means prompt found
}

pub fn run(root: &Path, keywords_csv: &str, progress: bool) -> Result<(), AppError> {
    let keywords = parse_keywords(keywords_csv);
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
        process_zip(path, &keywords, progress)?;
    }
    Ok(())
}

fn process_zip(path: &Path, keywords: &[String], progress: bool) -> Result<(), AppError> {
    let meta = fs::metadata(path)?;
    let atime = FileTime::from_last_access_time(&meta);
    let mtime = FileTime::from_last_modification_time(&meta);

    // 読み取りフェーズ
    let (entries, bar) = {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        let total = archive.len() as u64;

        let bar = if progress {
            let b = ProgressBar::new(total.saturating_mul(2));
            // WHY: 進捗の視認性を優先
            b.set_style(
                ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len} ({percent}%) {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());
            b.set_prefix(name);
            b.set_message("reading");
            Some(b)
        } else {
            None
        };

        let mut list = Vec::new();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            if file.name().ends_with('/') {
                continue;
            }
            list.push(read_entry(&mut file)?);
            if let Some(ref b) = bar {
                b.inc(1);
            }
        }
        if let Some(ref b) = bar {
            b.set_message("writing");
        }
        (list, bar)
    };

    let deletions = decide_deletions(&entries, keywords)?;
    write_filtered_zip(path, &entries, &deletions, bar.as_ref())?;
    set_file_times(path, atime, mtime)?;
    Ok(())
}

fn read_entry(file: &mut ZipFile) -> Result<EntryData, AppError> {
    let mut data = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut data)?;
    let options = FileOptions::default()
        .compression_method(file.compression())
        .last_modified_time(file.last_modified())
        .unix_permissions(file.unix_mode().unwrap_or(0o644));
    Ok(EntryData {
        name: file.name().to_string(),
        data,
        options,
    })
}

fn write_filtered_zip(
    path: &Path,
    entries: &[EntryData],
    deletions: &HashSet<String>,
    bar: Option<&ProgressBar>,
) -> Result<(), AppError> {
    let kept: Vec<&EntryData> = entries
        .iter()
        .filter(|e| !deletions.contains(&e.name))
        .collect();

    let mut buffer: Vec<u8> = Vec::new();
    {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(&mut buffer));
        for entry in kept {
            writer.start_file(&entry.name, entry.options)?;
            writer.write_all(&entry.data)?;
            if let Some(b) = bar {
                b.inc(1);
            }
        }
        writer.finish()?;
    }

    let orig_permissions = fs::metadata(path)?.permissions();
    {
        let mut dst = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(path)?;
        dst.write_all(&buffer)?;
        dst.sync_all()?;
    }
    fs::set_permissions(path, orig_permissions)?;

    if let Some(b) = bar {
        b.set_message("done");
        b.finish_and_clear();
    }
    Ok(())
}

fn decide_deletions(
    entries: &[EntryData],
    keywords: &[String],
) -> Result<HashSet<String>, AppError> {
    let mut deletions = HashSet::new();

    // 解析を並列化
    let analyses: Vec<EntryAnalysis> = entries
        .par_iter()
        .map(analyze_entry)
        .collect::<Result<_, _>>()?;

    let mut json_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    let mut image_prompts: HashMap<String, Option<Vec<String>>> = HashMap::new();
    let mut stem_to_entries: HashMap<String, Vec<String>> = HashMap::new();

    for a in &analyses {
        stem_to_entries
            .entry(a.stem.clone())
            .or_default()
            .push(a.name.clone());
        match a.kind {
            EntryKind::Json => {
                json_prompts.insert(a.stem.clone(), a.prompt_tags.clone());
            }
            EntryKind::Png | EntryKind::Webp => {
                image_prompts.insert(a.stem.clone(), a.prompt_tags.clone());
            }
            EntryKind::Other => {}
        }
    }

    for (stem, prompt_opt) in json_prompts.iter() {
        match prompt_opt {
            Some(tags) => {
                if tags_match(tags, keywords) {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            deletions.insert(name.clone());
                        }
                    }
                }
            }
            None => match image_prompts.get(stem) {
                Some(None) => {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            deletions.insert(name.clone());
                        }
                    }
                }
                Some(Some(_)) => {}
                None => {
                    if let Some(names) = stem_to_entries.get(stem) {
                        for name in names {
                            if Path::new(name).extension().map(|e| e == "json") == Some(true) {
                                deletions.insert(name.clone());
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

fn analyze_entry(entry: &EntryData) -> Result<EntryAnalysis, AppError> {
    let file_name = Path::new(&entry.name)
        .file_name()
        .ok_or_else(|| AppError::Invalid(format!("Invalid file name: {}", entry.name)))?
        .to_string_lossy()
        .to_string();
    let stem = Path::new(&file_name)
        .file_stem()
        .ok_or_else(|| AppError::Invalid(format!("Invalid stem: {}", entry.name)))?
        .to_string_lossy()
        .to_string();
    let ext = Path::new(&file_name)
        .extension()
        .map(|s| s.to_string_lossy().to_ascii_lowercase());

    let (kind, prompt_tags) = match ext.as_deref() {
        Some("json") => (EntryKind::Json, json_prompt_tags(&entry.data)?),
        Some("png") => (EntryKind::Png, png_prompt_tags(&entry.data)),
        Some("webp") => (EntryKind::Webp, webp_prompt_tags(&entry.data)),
        _ => (EntryKind::Other, None),
    };

    Ok(EntryAnalysis {
        name: entry.name.clone(),
        stem,
        kind,
        prompt_tags,
    })
}

fn json_prompt_tags(data: &[u8]) -> Result<Option<Vec<String>>, AppError> {
    let v: serde_json::Value = serde_json::from_slice(data)?;
    let prompt = find_prompt(&v);
    Ok(prompt.map(|s| normalize_prompt(&s)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    use zip::CompressionMethod;
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

        process_zip(&zip_path, &vec!["dog".into()], false).unwrap();

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

        process_zip(&zip_path, &vec!["dog".into()], false).unwrap();

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

        process_zip(&zip_path, &vec!["dog".into()], false).unwrap();

        let file = File::open(&zip_path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 2);
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

        process_zip(&zip_path, &vec!["anthro".into()], false).unwrap();

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
        process_zip(&zip_path, &vec!["cat".into()], false).unwrap();

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
