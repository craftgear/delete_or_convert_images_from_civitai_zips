# delete_images_from_civitai_zips

CLI tool that scans ZIP files under a directory and deletes image and JSON entries when their prompt matches given keywords. It preserves timestamps and avoids recompression by copying raw ZIP entries.

## Features
- Recursively scans for `.zip` files
- Reads prompts from JSON, PNG, and WebP metadata
- Deletes all entries in the same stem group when a match is found
- Skips `model_info.json`
- Caches processed ZIPs per keyword set
- Shows progress by default

## Requirements
- Rust stable

## Install
```bash
cargo build --release
```

The binary will be at `target/release/delete_images_from_zips`.

## Usage
```bash
delete_images_from_zips /path/to/zips --keywords cat,dog
```

### Options
- `DIR` Path to scan for ZIP files
- `--keywords` Comma separated keywords used for matching
- `--progress` Show progress to stderr, default is true

## Behavior
- For each ZIP file, prompts are read from JSON first. If a JSON entry has no prompt, the tool checks the corresponding image metadata.
- If a prompt contains any keyword, all entries with the same stem are deleted.
- If JSON has no prompt and image metadata has no prompt, the group is deleted.
- If JSON has no prompt but image metadata has a prompt, the group is kept.
- If JSON has a prompt that does not match, the group is kept even if image metadata matches.
- `model_info.json` is never deleted.

## Cache
Processed ZIPs are cached by keyword set using a hash of path, size, and mtime.  
Cache file location:
`~/.cache/delete_images_from_zips/processed.json`

## Environment variables
- `BUF_MB` Set write buffer size in MB, default is 16

## Examples
```bash
# delete entries matching either "cat" or "dog"
delete_images_from_zips ./data --keywords cat,dog

# disable progress output
delete_images_from_zips ./data --keywords cat --progress false
```

## Development
```bash
cargo test
```
