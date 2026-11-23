//! Settings that belong to the user, not to a graph, and the Preferences window that edits them.

use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    #[default]
    System,
    Dark,
    Light,
}

impl Theme {
    pub fn preference(self) -> egui::ThemePreference {
        match self {
            Theme::System => egui::ThemePreference::System,
            Theme::Dark => egui::ThemePreference::Dark,
            Theme::Light => egui::ThemePreference::Light,
        }
    }
}

/// The key that opens the library drawer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LibraryKey {
    #[default]
    Tab,
    Backtick,
    F2,
}

impl LibraryKey {
    pub const ALL: [LibraryKey; 3] = [LibraryKey::Tab, LibraryKey::Backtick, LibraryKey::F2];

    pub fn key(self) -> egui::Key {
        match self {
            LibraryKey::Tab => egui::Key::Tab,
            LibraryKey::Backtick => egui::Key::Backtick,
            LibraryKey::F2 => egui::Key::F2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LibraryKey::Tab => "Tab",
            LibraryKey::Backtick => "` (backtick)",
            LibraryKey::F2 => "F2",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefsTab {
    #[default]
    General,
    Keybinds,
    Performance,
    FilePaths,
    Style,
}

impl PrefsTab {
    const ALL: [(PrefsTab, &'static str); 5] = [(PrefsTab::General, "General"), (PrefsTab::Keybinds, "Keybinds"), (PrefsTab::Performance, "Performance"), (PrefsTab::FilePaths, "File paths"), (PrefsTab::Style, "Style")];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Scale of the whole interface (egui zoom factor).
    pub ui_scale: f32,
    /// Rolling crash-recovery backups of the open graph.
    pub backups: bool,
    /// Nodes snap to a grid when released.
    pub snap_nodes: bool,
    /// The About / getting-started window was shown once already.
    pub seen_about: bool,
    /// The library folder, or empty for none.
    pub library_root: String,
    /// Write a picture of the view next to files saved into the library.
    pub thumbnails: bool,
    pub library_key: LibraryKey,
    pub theme: Theme,
    /// Colour of selections and highlights; none keeps egui's blue.
    pub accent: Option<[u8; 3]>,
    /// Starting values of the 3D preview.
    pub render_scale: f32,
    pub max_steps: u32,
    /// Samples along the longest side when a `.sdf` file is turned into a mesh.
    pub sdf_mesh_resolution: u32,
    /// Starting value of the Composer's mesh export.
    pub export_samples: u32,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            ui_scale: 1.0,
            backups: true,
            snap_nodes: false,
            seen_about: false,
            library_root: String::new(),
            thumbnails: true,
            library_key: LibraryKey::Tab,
            theme: Theme::System,
            accent: None,
            render_scale: 1.0,
            max_steps: 300,
            sdf_mesh_resolution: 160,
            export_samples: 224,
        }
    }
}

impl Prefs {
    pub fn sane(mut self) -> Self {
        self.ui_scale = if self.ui_scale.is_finite() { self.ui_scale.clamp(0.75, 2.0) } else { 1.0 };
        self.render_scale = if [1.0, 0.5, 0.25].contains(&self.render_scale) { self.render_scale } else { 1.0 };
        self.max_steps = self.max_steps.clamp(32, 600);
        self.sdf_mesh_resolution = self.sdf_mesh_resolution.clamp(32, 400);
        self.export_samples = self.export_samples.clamp(48, 640);
        self
    }

    pub fn library_path(&self) -> Option<std::path::PathBuf> {
        Some(self.library_root.trim()).filter(|p| !p.is_empty()).map(std::path::PathBuf::from)
    }
}

/// What the window wants the app to do besides editing the values.
#[derive(Default)]
pub struct Effects {
    /// The scale slider was let go: apply the zoom now.
    pub scale_committed: bool,
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new(text).strong());
    ui.separator();
}

/// The window's contents: category tabs along the top and the settings of the chosen one below.
pub fn show(prefs: &mut Prefs, tab: &mut PrefsTab, ui: &mut egui::Ui, shortcuts: &[(&str, &str)]) -> Effects {
    let mut effects = Effects::default();
    ui.set_min_width(470.0);
    ui.horizontal(|ui| {
        for (which, label) in PrefsTab::ALL {
            ui.selectable_value(tab, which, label);
        }
    });
    ui.separator();
    egui::ScrollArea::vertical().max_height(420.0).auto_shrink([false, true]).show(ui, |ui| match *tab {
        PrefsTab::General => {
            heading(ui, "Files");
            ui.checkbox(&mut prefs.backups, "Keep rolling backups for crash recovery");
            heading(ui, "Graph editor");
            ui.checkbox(&mut prefs.snap_nodes, "Snap nodes to a grid when released");
            heading(ui, "Library");
            ui.checkbox(&mut prefs.thumbnails, "Save a picture of the view next to files saved into the library")
                .on_hover_text("Pictures are separate PNG files (name.sdf.png), so the saved files stay small");
        }
        PrefsTab::Keybinds => {
            heading(ui, "Library drawer");
            ui.horizontal(|ui| {
                let label = ui.label("Open the library with");
                egui::ComboBox::from_id_salt("library key").selected_text(prefs.library_key.label()).show_ui(ui, |ui| {
                    for key in LibraryKey::ALL {
                        ui.selectable_value(&mut prefs.library_key, key, key.label());
                    }
                })
                .response
                .labelled_by(label.id);
            });
            heading(ui, "All shortcuts");
            egui::Grid::new("prefs shortcuts").striped(true).show(ui, |ui| {
                for (keys, what) in shortcuts {
                    ui.monospace(*keys);
                    ui.label(*what);
                    ui.end_row();
                }
            });
        }
        PrefsTab::Performance => {
            heading(ui, "3D preview");
            ui.horizontal(|ui| {
                let label = ui.label("Starting render resolution");
                for (scale, name) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                    ui.selectable_value(&mut prefs.render_scale, scale, name);
                }
                let _ = label;
            });
            let steps = ui.add(egui::Slider::new(&mut prefs.max_steps, 32..=600).logarithmic(true).text("Starting max march steps"));
            steps.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Starting max march steps"));
            ui.weak("These are the values the preview starts from and returns to; they also apply to the open preview when changed here.");
            heading(ui, "SDF files");
            let resolution = ui.add(egui::Slider::new(&mut prefs.sdf_mesh_resolution, 32..=400).text("Mesh resolution"));
            resolution.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Mesh resolution"));
            ui.weak("Samples along the longest side when a .sdf file is turned into a mesh. Higher is finer and slower.");
            let samples = ui.add(egui::Slider::new(&mut prefs.export_samples, 48..=640).text("Composer export samples"));
            samples.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Composer export samples"));
        }
        PrefsTab::FilePaths => {
            heading(ui, "Library folder");
            ui.weak("Saving and opening start in the library's folders. Exporters can still be pointed elsewhere.");
            ui.horizontal(|ui| {
                let field = ui.add(egui::TextEdit::singleline(&mut prefs.library_root).hint_text("No library folder").desired_width(280.0));
                field.widget_info(|| egui::WidgetInfo::labeled(egui::Role::TextInput, true, "Library folder"));
                if ui.button("Browse…").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    prefs.library_root = dir.to_string_lossy().into_owned();
                }
            });
            match prefs.library_path() {
                None => {
                    ui.weak("Choose a folder to switch the library on.");
                }
                Some(root) => {
                    egui::Grid::new("library folders").show(ui, |ui| {
                        for kind in crate::library::Kind::ALL {
                            let folder = root.join(kind.folder());
                            ui.label(kind.folder());
                            if folder.is_dir() {
                                ui.weak("ready");
                            } else {
                                ui.colored_label(crate::editor::ORANGE, "missing");
                            }
                            ui.end_row();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Create the folders").on_hover_text("Make the library folder and its subfolders").clicked() {
                            let _ = crate::library::create_folders(&root);
                        }
                        if ui.button("Show in explorer").clicked() {
                            let _ = std::process::Command::new("explorer").arg(&root).spawn();
                        }
                    });
                }
            }
        }
        PrefsTab::Style => {
            heading(ui, "Interface");
            let scale = ui.add(egui::Slider::new(&mut prefs.ui_scale, 0.75..=2.0).text("Interface scale"));
            scale.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, "Interface scale"));
            // Applied on release: rescaling under the dragged slider would move it.
            effects.scale_committed = scale.changed() && !scale.dragged() || scale.drag_stopped();
            ui.horizontal(|ui| {
                let label = ui.label("Theme");
                for (theme, name) in [(Theme::System, "System"), (Theme::Dark, "Dark"), (Theme::Light, "Light")] {
                    ui.selectable_value(&mut prefs.theme, theme, name);
                }
                let _ = label;
            });
            heading(ui, "Accent colour");
            ui.horizontal(|ui| {
                let mut color = prefs.accent.unwrap_or([0, 92, 128]);
                let label = ui.label("Selections and highlights");
                if ui.color_edit_button_srgb(&mut color).labelled_by(label.id).changed() {
                    prefs.accent = Some(color);
                }
                if ui.add_enabled(prefs.accent.is_some(), egui::Button::new("Default")).clicked() {
                    prefs.accent = None;
                }
            });
        }
    });
    ui.separator();
    if ui.button("Reset to defaults").clicked() {
        *prefs = Prefs { seen_about: true, library_root: std::mem::take(&mut prefs.library_root), ..Prefs::default() };
        effects.scale_committed = true;
    }
    effects
}

/// The egui visuals for a theme with the accent colour applied.
pub fn visuals(dark: bool, accent: Option<[u8; 3]>) -> egui::Visuals {
    let mut visuals = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    if let Some([r, g, b]) = accent {
        let color = egui::Color32::from_rgb(r, g, b);
        visuals.selection.bg_fill = color;
        visuals.hyperlink_color = color;
    }
    visuals
}
