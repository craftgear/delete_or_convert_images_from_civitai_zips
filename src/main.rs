use std::path::PathBuf;

use clap::Parser;
use delete_or_convert_images_from_civitai_zips::{run, AppError, ConvertFormat};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Root directory to scan
    #[arg(value_name = "DIR", required_unless_present = "clear_cache")]
    root: Option<PathBuf>,

    /// Keywords separated by comma
    #[arg(long, value_name = "kw1,kw2")]
    keywords: Option<String>,

    /// Show progress to stderr
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    progress: bool,

    /// Clear cache file and exit
    #[arg(long)]
    clear_cache: bool,

    /// Convert png to webp, jpg, or jxl after deletions
    #[arg(long, value_enum)]
    convert: Option<ConvertFormat>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.clear_cache {
        delete_or_convert_images_from_civitai_zips::clear_cache_file()
            .map_err(|e| anyhow::anyhow!(e))?;
        return Ok(());
    }
    let root = cli
        .root
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DIR is required unless --clear-cache"))?;
    let keywords = cli.keywords.as_deref().unwrap_or("");
    match run(root, keywords, cli.progress, cli.convert) {
        Ok(()) => Ok(()),
        Err(AppError::Interrupted) => {
            eprintln!("Interrupted");
            std::process::exit(130);
        }
        Err(e) => Err(anyhow::anyhow!(e)),
    }
}
