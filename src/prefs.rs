//! Settings that belong to the user, not to a graph, and the Preferences window that edits them.

use crate::keymap::{self, Action, Binding, Keymap};
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

/// The single library key older versions saved; it becomes the first library binding when settings are loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum LibraryKey {
    Tab,
    Backtick,
    F2,
}

impl LibraryKey {
    fn key(self) -> egui::Key {
        match self {
            LibraryKey::Tab => egui::Key::Tab,
            LibraryKey::Backtick => egui::Key::Backtick,
            LibraryKey::F2 => egui::Key::F2,
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
    #[serde(rename = "library_key", skip_serializing)]
    pub legacy_library_key: Option<LibraryKey>,
    pub keymap: Keymap,
    pub theme: Theme,
    /// Colour of selections and highlights; none keeps egui's blue.
    pub accent: Option<[u8; 3]>,
    /// Starting values of the 3D preview.
    pub render_scale: f32,
    pub max_steps: u32,
    /// Starting values of the SDF Composer's viewport.
    pub composer_render_scale: f32,
    pub composer_max_steps: u32,
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
            legacy_library_key: None,
            keymap: Keymap::default(),
            theme: Theme::System,
            accent: None,
            render_scale: 1.0,
            max_steps: 300,
            composer_render_scale: 1.0,
            composer_max_steps: 260,
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
        self.composer_render_scale = if [1.0, 0.5, 0.25].contains(&self.composer_render_scale) { self.composer_render_scale } else { 1.0 };
        self.composer_max_steps = self.composer_max_steps.clamp(32, 600);
        self.sdf_mesh_resolution = self.sdf_mesh_resolution.clamp(32, 400);
        self.export_samples = self.export_samples.clamp(48, 640);
        if let Some(old) = self.legacy_library_key.take() {
            self.keymap.set(Action::Library, 0, Some(Binding::plain(old.key())));
        }
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

/// The slot being recorded: an action and which of its two slots.
fn recording_id() -> egui::Id {
    egui::Id::new("keybind being recorded")
}

/// Every action with its two slots, under group headings.
fn keybind_table(keymap: &mut Keymap, ui: &mut egui::Ui) {
    let mut recording: Option<(Action, usize)> = ui.memory(|m| m.data.get_temp(recording_id()));
    let mut group = "";
    for action in Action::ALL {
        if action.group() != group {
            group = action.group();
            heading(ui, group);
        }
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(220.0, 20.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.set_min_width(220.0);
                ui.label(action.label());
            });
            for slot in 0..2 {
                let listening = recording == Some((action, slot));
                let text = if listening { "Press a key…".to_string() } else { keymap.get(action)[slot].map_or("—".to_string(), Binding::text) };
                let button = ui.add_sized([116.0, 20.0], egui::Button::new(egui::RichText::new(text).monospace()).selected(listening));
                button.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, format!("{} binding {}", action.label(), slot + 1)));
                if button.clicked() {
                    recording = Some((action, slot));
                } else if listening {
                    // Holding focus keeps the editors from also acting on the key that is being recorded.
                    button.request_focus();
                    ui.memory_mut(|m| m.set_focus_lock_filter(button.id, egui::EventFilter { tab: true, horizontal_arrows: true, vertical_arrows: true, escape: true }));
                    let pressed = ui.input(|i| {
                        i.events.iter().find_map(|e| match e {
                            egui::Event::Key { key, pressed: true, repeat: false, modifiers, .. } => Some(Binding { key: *key, ctrl: modifiers.command, shift: modifiers.shift }),
                            _ => None,
                        })
                    });
                    if let Some(binding) = pressed {
                        if binding.key != egui::Key::Escape {
                            keymap.set(action, slot, Some(binding));
                        }
                        recording = None;
                        button.surrender_focus();
                    } else if ui.input(|i| i.pointer.any_pressed()) && !button.hovered() {
                        recording = None;
                    }
                }
                // The clear button's place stays reserved so the columns line up.
                let (cell, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
                if keymap.get(action)[slot].is_some() {
                    let clear = ui.put(cell, egui::Button::new("×").small()).on_hover_text("Clear this slot");
                    clear.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, format!("{} clear {}", action.label(), slot + 1)));
                    if clear.clicked() {
                        keymap.set(action, slot, None);
                    }
                }
            }
            let clashes: Vec<&str> = keymap.get(action).into_iter().flatten().filter_map(|b| keymap.conflict(action, b)).map(Action::label).collect();
            if !clashes.is_empty() {
                ui.colored_label(crate::editor::ORANGE, format!("also used by: {}", clashes.join(", ")));
            }
        });
    }
    keymap::set_capturing(recording.is_some());
    ui.memory_mut(|m| match recording {
        Some(r) => {
            m.data.insert_temp(recording_id(), r);
        }
        None => m.data.remove::<(Action, usize)>(recording_id()),
    });
}

/// The window's contents: category tabs along the top and the settings of the chosen one below.
pub fn show(prefs: &mut Prefs, tab: &mut PrefsTab, ui: &mut egui::Ui, shortcuts: &[(&str, &str)]) -> Effects {
    let mut effects = Effects::default();
    if *tab != PrefsTab::Keybinds {
        keymap::set_capturing(false);
    }
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
            ui.horizontal(|ui| {
                ui.weak("Click a slot, then press the key (with Ctrl or Shift if you like). Escape cancels.");
                if ui.add_enabled(!prefs.keymap.is_default(), egui::Button::new("Restore default keybinds")).clicked() {
                    prefs.keymap = Keymap::default();
                }
            });
            keybind_table(&mut prefs.keymap, ui);
            heading(ui, "Fixed shortcuts");
            egui::Grid::new("prefs shortcuts").striped(true).show(ui, |ui| {
                for (keys, what) in shortcuts {
                    ui.monospace(*keys);
                    ui.label(*what);
                    ui.end_row();
                }
            });
        }
        PrefsTab::Performance => {
            heading(ui, "Noise graph 3D preview");
            egui::Grid::new("prefs preview").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                ui.label("Starting render resolution");
                ui.horizontal(|ui| {
                    for (scale, name) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                        ui.selectable_value(&mut prefs.render_scale, scale, name);
                    }
                });
                ui.end_row();
                let label = ui.label("Starting max march steps");
                ui.add(egui::Slider::new(&mut prefs.max_steps, 32..=600).logarithmic(true)).labelled_by(label.id);
                ui.end_row();
            });
            ui.weak("These are the values the preview starts from; changing them here also changes the open preview.");
            heading(ui, "SDF Composer viewport");
            egui::Grid::new("prefs composer preview").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                ui.label("Starting render resolution");
                ui.horizontal(|ui| {
                    for (scale, name) in [(1.0, "Full"), (0.5, "Half"), (0.25, "Quarter")] {
                        ui.selectable_value(&mut prefs.composer_render_scale, scale, name);
                    }
                });
                ui.end_row();
                let label = ui.label("Starting max ray steps");
                ui.add(egui::Slider::new(&mut prefs.composer_max_steps, 32..=600).logarithmic(true)).labelled_by(label.id);
                ui.end_row();
            });
            ui.weak("The Composer ray marches the whole composition every frame: Half or Quarter resolution helps most when it gets sluggish. The viewport's own quality button changes it on the spot.");
            heading(ui, "SDF files");
            egui::Grid::new("prefs sdf").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                let label = ui.label("Mesh resolution");
                ui.add(egui::Slider::new(&mut prefs.sdf_mesh_resolution, 32..=400)).labelled_by(label.id);
                ui.end_row();
                let label = ui.label("Composer export samples");
                ui.add(egui::Slider::new(&mut prefs.export_samples, 48..=640)).labelled_by(label.id);
                ui.end_row();
            });
            ui.weak("Mesh resolution is the number of samples along the longest side when a .sdf file is turned into a mesh: higher is finer and slower.");
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
            egui::Grid::new("prefs style").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                let label = ui.label("Interface scale");
                let scale = ui.add(egui::Slider::new(&mut prefs.ui_scale, 0.75..=2.0));
                scale.clone().labelled_by(label.id);
                // Applied on release: rescaling under the dragged slider would move it.
                effects.scale_committed = scale.changed() && !scale.dragged() || scale.drag_stopped();
                ui.end_row();
                ui.label("Theme");
                ui.horizontal(|ui| {
                    for (theme, name) in [(Theme::System, "System"), (Theme::Dark, "Dark"), (Theme::Light, "Light")] {
                        ui.selectable_value(&mut prefs.theme, theme, name);
                    }
                });
                ui.end_row();
                let label = ui.label("Accent colour");
                ui.horizontal(|ui| {
                    let mut color = prefs.accent.unwrap_or([0, 92, 128]);
                    if ui.color_edit_button_srgb(&mut color).labelled_by(label.id).changed() {
                        prefs.accent = Some(color);
                    }
                    if ui.add_enabled(prefs.accent.is_some(), egui::Button::new("Default")).clicked() {
                        prefs.accent = None;
                    }
                });
                ui.end_row();
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

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    fn harness() -> Harness<'static, Keymap> {
        Harness::builder().with_size(egui::vec2(700.0, 900.0)).with_step_dt(1.0 / 60.0).build_ui_state(|ui, keymap: &mut Keymap| keybind_table(keymap, ui), Keymap::default())
    }

    #[test]
    fn clicking_a_slot_and_pressing_a_key_rebinds_it_and_escape_cancels() {
        let mut h = harness();
        h.run();
        h.get_by_label("Mute (bypass) the selected nodes binding 2").click();
        h.run();
        h.key_press(egui::Key::B);
        h.run();
        assert_eq!(h.state().get(Action::Mute), [Some(Binding::plain(egui::Key::M)), Some(Binding::plain(egui::Key::B))]);

        h.get_by_label("Mute (bypass) the selected nodes binding 1").click();
        h.run();
        h.key_press(egui::Key::Escape);
        h.run();
        assert_eq!(h.state().get(Action::Mute)[0], Some(Binding::plain(egui::Key::M)), "escape leaves it alone");
        keymap::set_capturing(false);
    }

    #[test]
    fn the_clear_button_empties_a_slot() {
        let mut h = harness();
        h.run();
        h.get_by_label("Delete the selection clear 2").click();
        h.run();
        assert_eq!(h.state().get(Action::Delete), [Some(Binding::plain(egui::Key::Delete)), None]);
    }
}
