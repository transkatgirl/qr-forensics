mod app;
mod cli;

use eframe::egui;

/// What the window takes, for `--help`; the headless reader prints a
/// usage of its own.
const USAGE: &str = "usage: qr-forensics [<image>]
       qr-forensics decode [options] <image>...

Opens the window, with the image given loaded. `decode` runs the headless reader instead, with no window, and
exits with its status: `qr-forensics decode --help` says what it takes.";

fn main() -> eframe::Result {
    // `decode` first on the command line runs the headless reader, with
    // no window, and exits with its status.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|first| first == "decode") {
        std::process::exit(cli::run(&args[1..]));
    }
    // The usage or the version asked for is printed, not opened as an
    // image named `--help`.
    if args
        .first()
        .is_some_and(|first| first == "--help" || first == "-h")
    {
        println!("{USAGE}");
        return Ok(());
    }
    if args.first().is_some_and(|first| first == "--version") {
        println!("qr-forensics {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    // The photo's filters run on the graphics card: with none, the window
    // says so before it would open, as well as the terminal, if there is
    // one to read it.
    if let Err(message) = qr_forensics::imaging::gpu::ready() {
        eprintln!("{message}");
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("QR Forensics")
            .set_description(message)
            .set_buttons(rfd::MessageButtons::Ok)
            .show();
        std::process::exit(1);
    }
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("QR Forensics")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    // The window is drawn by the graphics card the photo is adjusted on,
    // so that what is made there to be shown is laid into the window's
    // textures there, and not taken off the card to be handed it again.
    if let Some((instance, adapter, device, queue)) = qr_forensics::imaging::gpu::shared_device() {
        options.wgpu_options.wgpu_setup = eframe::egui_wgpu::WgpuSetupExisting { instance, adapter, device, queue }.into();
    }
    // An image path on the command line is opened at startup.
    let image = args.first().map(std::path::PathBuf::from);
    eframe::run_native(
        "QR Forensics",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, image)))),
    )
}
