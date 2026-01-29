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

Windows binary can be downloaded from the [ GitHub Releases page ](https://github.com/craftgear/delete_images_from_civitai_zips/releases).

## Build
```bash
cargo build --release
```

## Usage
```bash
delete_images_from_zips /path/to/zips --keywords "cat,dog"
```

### Maintenance
- Clear cache: `delete_images_from_zips --clear-cache`

### Options
- `DIR` Path to scan for ZIP files
- `--keywords` Comma separated keywords used for matching
- `--progress` Show progress to stderr, default is true
- `--clear-cache` Delete cache file and exit
- `--convert` Convert png to `webp`, `jpg`, or `jxl` after deletions

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

# clear cache only
delete_images_from_zips --clear-cache

# convert png to webp
delete_images_from_zips ./data --keywords cat --convert webp

# convert png to jpg
delete_images_from_zips ./data --keywords cat --convert jpg

# convert png to jxl
delete_images_from_zips ./data --keywords cat --convert jxl

```

## Development
```bash
cargo test
```

## License
GPL-3.0-or-later. See `LICENSE`.
