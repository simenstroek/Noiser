// Gives noiser.exe the icon the window already shows (eframe's default), so Explorer and the taskbar show it too.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winresource::WindowsResource::new().set_icon("assets/icon.ico").compile().expect("could not embed the icon");
    }
}
