//! The library: a folder chosen in the preferences, with one subfolder per kind of file. Saving into it also writes a
//! small picture of the active view next to the file (`name.sdf.png`), so the files themselves stay light. The drawer
//! (Tab) lists what is there and lets it be dragged into the open tab.

use eframe::egui;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Sdf,
    Noise,
    MeshGraph,
    Bake,
    Udim,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Sdf, Kind::Noise, Kind::MeshGraph, Kind::Bake, Kind::Udim];
    /// The kinds the drawer lists (the others are only output folders).
    pub const SHOWN: [Kind; 3] = [Kind::Sdf, Kind::Noise, Kind::MeshGraph];

    pub fn folder(self) -> &'static str {
        match self {
            Kind::Sdf => "SDFs",
            Kind::Noise => "Noises",
            Kind::MeshGraph => "Mesh Graphs",
            Kind::Bake => "Bakes",
            Kind::Udim => "UDIM Packs",
        }
    }

    pub fn extension(self) -> Option<&'static str> {
        match self {
            Kind::Sdf => Some("sdf"),
            Kind::Noise => Some("noisegraph"),
            Kind::MeshGraph => Some("meshgraph"),
            Kind::Bake | Kind::Udim => None,
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Kind::Sdf => "shape_cube",
            Kind::Noise => "noise",
            Kind::MeshGraph => "mesh_graph",
            Kind::Bake => "bake",
            Kind::Udim => "udim",
        }
    }
}

static ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);
static THUMBNAILS: Mutex<bool> = Mutex::new(true);
static REQUESTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static VIEW: Mutex<Option<egui::Rect>> = Mutex::new(None);

pub fn set_root(root: Option<PathBuf>) {
    if let Ok(mut r) = ROOT.lock() {
        *r = root;
    }
}

pub fn root() -> Option<PathBuf> {
    ROOT.lock().ok().and_then(|r| r.clone())
}

pub fn set_thumbnails(on: bool) {
    if let Ok(mut t) = THUMBNAILS.lock() {
        *t = on;
    }
}

/// The folder for a kind of file, when a library is set.
pub fn dir(kind: Kind) -> Option<PathBuf> {
    root().map(|r| r.join(kind.folder()))
}

/// Makes the library folder and its subfolders.
pub fn create_folders(root: &Path) -> std::io::Result<()> {
    for kind in Kind::ALL {
        std::fs::create_dir_all(root.join(kind.folder()))?;
    }
    Ok(())
}

/// A file dialog that opens in the library's folder for `kind`, when there is one. A dialog that already has a
/// directory of its own (a file the user is working on) is better left alone, so callers use this for fresh ones.
pub fn dialog(kind: Kind, dialog: rfd::FileDialog) -> rfd::FileDialog {
    match dir(kind) {
        Some(d) if d.is_dir() => dialog.set_directory(d),
        _ => dialog,
    }
}

/// The library folder for the kind of file `path` is: used to put a save dialog where the file already lives.
pub fn is_inside(path: &Path) -> bool {
    let Some(root) = root() else { return false };
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    // The file may not exist yet, but its folder does.
    let path = match (path.parent().and_then(|p| std::fs::canonicalize(p).ok()), path.file_name()) {
        (Some(parent), Some(name)) => parent.join(name),
        _ => path.to_path_buf(),
    };
    path.starts_with(root)
}

pub fn thumbnail_path(file: &Path) -> PathBuf {
    let mut name = file.file_name().map_or_else(Default::default, |n| n.to_os_string());
    name.push(".png");
    file.with_file_name(name)
}

/// Asks for a picture of the active view to be written next to `file`, if it was saved into the library.
pub fn request_thumbnail(file: &Path) {
    let enabled = THUMBNAILS.lock().map(|t| *t).unwrap_or(false);
    if enabled && is_inside(file) && let Ok(mut r) = REQUESTS.lock() {
        r.push(file.to_path_buf());
    }
}

pub fn take_requests() -> Vec<PathBuf> {
    REQUESTS.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default()
}

/// The screen rectangle of the view the picture is cut from; every tab sets it while it is showing.
pub fn set_view_rect(rect: egui::Rect) {
    if let Ok(mut v) = VIEW.lock() {
        *v = Some(rect);
    }
}

pub fn view_rect() -> Option<egui::Rect> {
    VIEW.lock().ok().and_then(|v| *v)
}

/// The longest side of a thumbnail, in pixels.
const THUMBNAIL_SIDE: u32 = 320;

/// Cuts `rect` (in points) out of a screenshot, scales it down and writes it as a PNG.
pub fn save_thumbnail(screenshot: &egui::ColorImage, rect: egui::Rect, pixels_per_point: f32, dest: &Path) -> Result<(), String> {
    let [width, height] = screenshot.size;
    let x0 = ((rect.min.x * pixels_per_point).round().max(0.0) as usize).min(width);
    let y0 = ((rect.min.y * pixels_per_point).round().max(0.0) as usize).min(height);
    let x1 = ((rect.max.x * pixels_per_point).round().max(0.0) as usize).min(width);
    let y1 = ((rect.max.y * pixels_per_point).round().max(0.0) as usize).min(height);
    if x1 <= x0 || y1 <= y0 {
        return Err("the view is not on screen".into());
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in y0..y1 {
        for pixel in &screenshot.pixels[y * width + x0..y * width + x1] {
            rgba.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b(), 255]);
        }
    }
    let image = image::RgbaImage::from_raw(w as u32, h as u32, rgba).ok_or("bad image size")?;
    let longest = w.max(h) as f32;
    let image = if longest > THUMBNAIL_SIDE as f32 {
        let scale = THUMBNAIL_SIDE as f32 / longest;
        image::imageops::resize(&image, ((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1), image::imageops::FilterType::Triangle)
    } else {
        image
    };
    image.save(dest).map_err(|e| e.to_string())
}

// ---- the drawer ---------------------------------------------------------------------------------------------------

/// What a card carries while it is dragged.
#[derive(Clone, Debug)]
pub struct LibraryItem {
    pub path: PathBuf,
    pub kind: Kind,
}

#[derive(Clone, Debug)]
struct Entry {
    path: PathBuf,
    name: String,
    thumbnail: Option<PathBuf>,
}

/// What the drawer asks the app to do.
pub enum Action {
    /// A card was double-clicked.
    Open(PathBuf, Kind),
    /// The "Set a library folder" button.
    ChooseFolder,
}

const RESCAN: Duration = Duration::from_millis(1500);

#[derive(Default)]
pub struct Drawer {
    pub open: bool,
    tab: Option<Kind>,
    filter: String,
    listings: HashMap<Kind, (Instant, Vec<Entry>)>,
    textures: HashMap<PathBuf, (Option<SystemTime>, Option<egui::TextureHandle>)>,
}

fn scan(kind: Kind) -> Vec<Entry> {
    let (Some(dir), Some(extension)) = (dir(kind), kind.extension()) else { return Vec::new() };
    let mut out = Vec::new();
    let mut stack = vec![dir];
    while let Some(folder) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&folder) else { continue };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case(extension)) {
                let thumbnail = Some(thumbnail_path(&path)).filter(|t| t.is_file());
                let name = path.file_stem().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                out.push(Entry { path, name, thumbnail });
            }
        }
    }
    out.sort_by_key(|e| e.name.to_lowercase());
    out
}

fn load_texture(ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
    let image = image::open(path).ok()?.to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    Some(ctx.load_texture(path.to_string_lossy(), egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw()), egui::TextureOptions::LINEAR))
}

impl Drawer {
    fn entries(&mut self, kind: Kind) -> Vec<Entry> {
        let stale = self.listings.get(&kind).is_none_or(|(at, _)| at.elapsed() > RESCAN);
        if stale {
            self.listings.insert(kind, (Instant::now(), scan(kind)));
        }
        self.listings[&kind].1.clone()
    }

    fn texture(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureId> {
        let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let current = self.textures.get(path).is_some_and(|(m, _)| *m == modified);
        if !current {
            self.textures.insert(path.to_path_buf(), (modified, load_texture(ctx, path)));
        }
        self.textures.get(path).and_then(|(_, t)| t.as_ref()).map(|t| t.id())
    }

    /// The drawer as a panel along the bottom. `hint` says what a drag does in the tab that is open.
    pub fn show(&mut self, ui: &mut egui::Ui, hint: &str) -> Option<Action> {
        let mut action = None;
        let ctx = ui.ctx().clone();
        let tab = *self.tab.get_or_insert(Kind::Sdf);
        egui::Panel::bottom("library drawer").resizable(false).exact_size(178.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                for kind in Kind::SHOWN {
                    let on = tab == kind;
                    if crate::icons::button(ui, kind.icon(), kind.folder(), Some(on), true).on_hover_text(format!("Show the {} in the library", kind.folder())).clicked() {
                        self.tab = Some(kind);
                    }
                }
                ui.separator();
                ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Search").desired_width(140.0));
                ui.weak(hint);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::icons::button(ui, "close", "Close the library", None, true).on_hover_text("Close the library (Tab)").clicked() {
                        self.open = false;
                    }
                    if let Some(dir) = dir(tab)
                        && crate::icons::button(ui, "open", "Open the library folder", None, dir.is_dir()).on_hover_text("Show this folder in the file explorer").clicked()
                    {
                        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
                    }
                });
            });
            ui.separator();
            if root().is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.label("No library folder is set.");
                    if ui.button("Set a library folder").clicked() {
                        action = Some(Action::ChooseFolder);
                    }
                });
                return;
            }
            let filter = self.filter.to_lowercase();
            let entries: Vec<Entry> = self.entries(tab).into_iter().filter(|e| filter.is_empty() || e.name.to_lowercase().contains(&filter)).collect();
            if entries.is_empty() {
                ui.weak(format!("Nothing in {} yet. Files saved into this folder show up here with a picture of the view they were saved from.", tab.folder()));
                return;
            }
            egui::ScrollArea::horizontal().auto_shrink([false, false]).show(ui, |ui| {
                ui.horizontal(|ui| {
                    for entry in &entries {
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(124.0, 124.0), egui::Sense::click_and_drag());
                        let hovered = response.hovered();
                        ui.painter().rect_filled(rect, 4.0, if hovered { egui::Color32::from_gray(48) } else { egui::Color32::from_gray(36) });
                        let picture = egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::vec2(116.0, 90.0));
                        match entry.thumbnail.as_ref().and_then(|t| self.texture(&ctx, t)) {
                            Some(texture) => {
                                let size = ctx.tex_manager().read().meta(texture).map_or(egui::vec2(1.0, 1.0), |m| egui::vec2(m.size[0] as f32, m.size[1] as f32));
                                let fit = (picture.width() / size.x).min(picture.height() / size.y);
                                let shown = egui::Rect::from_center_size(picture.center(), size * fit);
                                ui.painter().image(texture, shown, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                            }
                            None => {
                                let side = 36.0;
                                crate::icons::image(tab.icon(), side, egui::Color32::from_gray(110)).paint_at(ui, egui::Rect::from_center_size(picture.center(), egui::vec2(side, side)));
                            }
                        }
                        let name = ui.painter().layout(entry.name.clone(), egui::FontId::proportional(12.0), egui::Color32::from_gray(210), 116.0);
                        ui.painter().with_clip_rect(rect).galley(rect.left_top() + egui::vec2(6.0, 98.0), name, egui::Color32::from_gray(210));
                        response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, entry.name.clone()));
                        let response = response.on_hover_text(format!("{}\nDrag it into the view, or double-click to open it.", entry.path.display()));
                        if response.drag_started() {
                            response.dnd_set_drag_payload(LibraryItem { path: entry.path.clone(), kind: tab });
                        }
                        if response.double_clicked() {
                            action = Some(Action::Open(entry.path.clone(), tab));
                        }
                    }
                });
            });
        });
        // A card that is being dragged follows the pointer.
        if let Some(item) = egui::DragAndDrop::payload::<LibraryItem>(&ctx)
            && let Some(pointer) = ctx.input(|i| i.pointer.interact_pos())
        {
            let name = item.path.file_stem().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            egui::Area::new(egui::Id::new("library drag")).order(egui::Order::Tooltip).fixed_pos(pointer + egui::vec2(14.0, 10.0)).interactable(false).show(&ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| ui.label(name));
            });
        }
        action
    }

    /// The card dropped this frame, if the pointer was let go outside `keep_out` (the drawer itself).
    pub fn dropped(ctx: &egui::Context, keep_out: egui::Rect) -> Option<(LibraryItem, egui::Pos2)> {
        if !ctx.input(|i| i.pointer.any_released()) {
            return None;
        }
        let item = egui::DragAndDrop::payload::<LibraryItem>(ctx)?;
        let at = ctx.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()))?;
        if keep_out.contains(at) {
            return None;
        }
        Some(((*item).clone(), at))
    }
}

/// The library folder is global, so tests that set it take turns.
#[cfg(test)]
pub(crate) static TURN: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    use super::TURN;

    #[test]
    fn thumbnails_sit_next_to_the_file() {
        assert_eq!(thumbnail_path(Path::new("C:/lib/SDFs/arrow.sdf")), PathBuf::from("C:/lib/SDFs/arrow.sdf.png"));
    }

    #[test]
    fn a_screenshot_is_cut_down_and_written() {
        let mut screenshot = egui::ColorImage::filled([800, 600], egui::Color32::from_rgb(10, 20, 30));
        for y in 100..200 {
            for x in 100..300 {
                screenshot.pixels[y * 800 + x] = egui::Color32::from_rgb(200, 0, 0);
            }
        }
        let dir = std::env::temp_dir().join("noiser_library_test");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("view.sdf.png");
        // At 2 pixels per point the points (50,50)-(150,100) are the pixels (100,100)-(300,200): all red.
        save_thumbnail(&screenshot, egui::Rect::from_min_max(egui::pos2(50.0, 50.0), egui::pos2(150.0, 100.0)), 2.0, &dest).unwrap();
        let written = image::open(&dest).unwrap().to_rgba8();
        assert_eq!((written.width(), written.height()), (200, 100));
        assert_eq!(written.get_pixel(100, 50).0, [200, 0, 0, 255]);
        // A view that is larger than the thumbnail is scaled to fit.
        save_thumbnail(&screenshot, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0)), 1.0, &dest).unwrap();
        let written = image::open(&dest).unwrap();
        assert_eq!((written.width(), written.height()), (THUMBNAIL_SIDE, 240));
        // Nothing on screen is an error, not a blank file.
        assert!(save_thumbnail(&screenshot, egui::Rect::from_min_max(egui::pos2(900.0, 0.0), egui::pos2(950.0, 50.0)), 1.0, &dest).is_err());
    }

    #[test]
    fn saving_inside_the_library_asks_for_a_thumbnail_and_outside_does_not() {
        let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join("noiser_library_root");
        create_folders(&root).unwrap();
        set_root(Some(root.clone()));
        set_thumbnails(true);
        take_requests();
        request_thumbnail(&root.join("SDFs").join("a.sdf"));
        request_thumbnail(&std::env::temp_dir().join("elsewhere.sdf"));
        let asked = take_requests();
        set_root(None);
        assert_eq!(asked, vec![root.join("SDFs").join("a.sdf")]);
    }

    #[test]
    fn the_scan_finds_files_of_the_right_kind_and_their_pictures() {
        let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join("noiser_library_scan");
        create_folders(&root).unwrap();
        std::fs::write(root.join("SDFs").join("one.sdf"), "{}").unwrap();
        std::fs::write(root.join("SDFs").join("one.sdf.png"), "").unwrap();
        std::fs::write(root.join("SDFs").join("two.txt"), "").unwrap();
        set_root(Some(root.clone()));
        let found = scan(Kind::Sdf);
        set_root(None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "one");
        assert!(found[0].thumbnail.is_some());
    }

    #[test]
    fn the_drawer_lists_what_is_in_the_library() {
        let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join("noiser_library_drawer");
        create_folders(&root).unwrap();
        std::fs::write(root.join("SDFs").join("arrow.sdf"), "{}").unwrap();
        set_root(Some(root));
        let mut h = Harness::builder().with_size(egui::vec2(700.0, 400.0)).build_ui_state(
            |ui, drawer: &mut Drawer| {
                drawer.show(ui, "Drag it somewhere.");
            },
            Drawer { open: true, ..Default::default() },
        );
        h.run();
        h.get_by_label("arrow");
        set_root(None);
        h.run();
        h.get_by_label("Set a library folder");
    }
}
