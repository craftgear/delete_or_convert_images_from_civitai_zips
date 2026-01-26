use std::path::PathBuf;

use clap::Parser;
use delete_images_from_zips::run;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Root directory to scan
    #[arg(value_name = "DIR")]
    root: PathBuf,

    /// Keywords separated by comma
    #[arg(long, value_name = "kw1,kw2")]
    keywords: String,

    /// Show progress to stderr
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    progress: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run(&cli.root, &cli.keywords, cli.progress).map_err(|e| anyhow::anyhow!(e))
}
