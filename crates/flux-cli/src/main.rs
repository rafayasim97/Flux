use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flux", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Bench,
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Bench => {
            println!("bench not implemented yet");
        }
    }
}
