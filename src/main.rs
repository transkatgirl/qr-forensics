mod app;
mod cli;

use eframe::egui;

fn main() -> eframe::Result {
    // `decode` first on the command line runs the headless reader, with
    // no window, and exits with its status.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|first| first == "decode") {
        std::process::exit(cli::run(&args[1..]));
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("QR Forensics")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    // An image path on the command line is opened at startup.
    let image = args.first().map(std::path::PathBuf::from);
    eframe::run_native("QR Forensics", options, Box::new(|cc| Ok(Box::new(app::App::new(cc, image)))))
}
