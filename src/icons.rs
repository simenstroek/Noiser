// Material Symbols icons (Apache 2.0) fetched from Iconify, drawn white and tinted at use.
use eframe::egui;

pub fn source(name: &str) -> Option<egui::ImageSource<'static>> {
    let bytes: &'static [u8] = match name {
        "add" => include_bytes!("../assets/ui_icons/add.svg"),
        "add_layer" => include_bytes!("../assets/ui_icons/add_layer.svg"),
        "arrange" => include_bytes!("../assets/ui_icons/arrange.svg"),
        "avoid" => include_bytes!("../assets/ui_icons/avoid.svg"),
        "bake" => include_bytes!("../assets/ui_icons/bake.svg"),
        "batch" => include_bytes!("../assets/ui_icons/batch.svg"),
        "close" => include_bytes!("../assets/ui_icons/close.svg"),
        "code" => include_bytes!("../assets/ui_icons/code.svg"),
        "collapsed" => include_bytes!("../assets/ui_icons/collapsed.svg"),
        "deboss" => include_bytes!("../assets/ui_icons/deboss.svg"),
        "delete" => include_bytes!("../assets/ui_icons/delete.svg"),
        "delete_layer" => include_bytes!("../assets/ui_icons/delete_layer.svg"),
        "down" => include_bytes!("../assets/ui_icons/down.svg"),
        "duplicate" => include_bytes!("../assets/ui_icons/duplicate.svg"),
        "emboss" => include_bytes!("../assets/ui_icons/emboss.svg"),
        "expanded" => include_bytes!("../assets/ui_icons/expanded.svg"),
        "export" => include_bytes!("../assets/ui_icons/export.svg"),
        "frame" => include_bytes!("../assets/ui_icons/frame.svg"),
        "function" => include_bytes!("../assets/ui_icons/function.svg"),
        "global" => include_bytes!("../assets/ui_icons/global.svg"),
        "group" => include_bytes!("../assets/ui_icons/group.svg"),
        "group_row" => include_bytes!("../assets/ui_icons/group_row.svg"),
        "import_mesh" => include_bytes!("../assets/ui_icons/import_mesh.svg"),
        "info" => include_bytes!("../assets/ui_icons/info.svg"),
        "intersect" => include_bytes!("../assets/ui_icons/intersect.svg"),
        "keyboard" => include_bytes!("../assets/ui_icons/keyboard.svg"),
        "library" => include_bytes!("../assets/ui_icons/library.svg"),
        "lighting" => include_bytes!("../assets/ui_icons/lighting.svg"),
        "local" => include_bytes!("../assets/ui_icons/local.svg"),
        "material" => include_bytes!("../assets/ui_icons/material.svg"),
        "mesh" => include_bytes!("../assets/ui_icons/mesh.svg"),
        "mesh_graph" => include_bytes!("../assets/ui_icons/mesh_graph.svg"),
        "move" => include_bytes!("../assets/ui_icons/move.svg"),
        "new" => include_bytes!("../assets/ui_icons/new.svg"),
        "noise" => include_bytes!("../assets/ui_icons/noise.svg"),
        "open" => include_bytes!("../assets/ui_icons/open.svg"),
        "paint" => include_bytes!("../assets/ui_icons/paint.svg"),
        "play" => include_bytes!("../assets/ui_icons/play.svg"),
        "preview" => include_bytes!("../assets/ui_icons/preview.svg"),
        "push" => include_bytes!("../assets/ui_icons/push.svg"),
        "recent" => include_bytes!("../assets/ui_icons/recent.svg"),
        "redo" => include_bytes!("../assets/ui_icons/redo.svg"),
        "rotate" => include_bytes!("../assets/ui_icons/rotate.svg"),
        "run_files" => include_bytes!("../assets/ui_icons/run_files.svg"),
        "save" => include_bytes!("../assets/ui_icons/save.svg"),
        "save_as" => include_bytes!("../assets/ui_icons/save_as.svg"),
        "scale" => include_bytes!("../assets/ui_icons/scale.svg"),
        "settings" => include_bytes!("../assets/ui_icons/settings.svg"),
        "shape_cube" => include_bytes!("../assets/ui_icons/shape_cube.svg"),
        "shape_cylinder" => include_bytes!("../assets/ui_icons/shape_cylinder.svg"),
        "shape_polygon" => include_bytes!("../assets/ui_icons/shape_polygon.svg"),
        "shape_prism" => include_bytes!("../assets/ui_icons/shape_prism.svg"),
        "shape_sphere" => include_bytes!("../assets/ui_icons/shape_sphere.svg"),
        "shell" => include_bytes!("../assets/ui_icons/shell.svg"),
        "slice" => include_bytes!("../assets/ui_icons/slice.svg"),
        "snap" => include_bytes!("../assets/ui_icons/snap.svg"),
        "stop" => include_bytes!("../assets/ui_icons/stop.svg"),
        "subtract" => include_bytes!("../assets/ui_icons/subtract.svg"),
        "template" => include_bytes!("../assets/ui_icons/template.svg"),
        "tune" => include_bytes!("../assets/ui_icons/tune.svg"),
        "udim" => include_bytes!("../assets/ui_icons/udim.svg"),
        "undo" => include_bytes!("../assets/ui_icons/undo.svg"),
        "up" => include_bytes!("../assets/ui_icons/up.svg"),
        "view_3d" => include_bytes!("../assets/ui_icons/view_3d.svg"),
        _ => return None,
    };
    Some(egui::ImageSource::Bytes { uri: format!("bytes://icons/{name}.svg").into(), bytes: egui::load::Bytes::Static(bytes) })
}

pub fn image(name: &str, size: f32, tint: egui::Color32) -> egui::Image<'static> {
    let source = source(name).unwrap_or_else(|| panic!("no icon called {name}"));
    egui::Image::new(source).fit_to_exact_size(egui::vec2(size, size)).tint(tint)
}

/// A square icon button whose accessible name is `label`; the caller adds the tooltip.
pub fn button(ui: &mut egui::Ui, name: &str, label: &str, selected: Option<bool>, enabled: bool) -> egui::Response {
    tinted_button(ui, name, label, selected, enabled, None)
}

/// Like `button`, with the icon in `color` (when enabled): for actions that should stand out, like Run.
pub fn tinted_button(ui: &mut egui::Ui, name: &str, label: &str, selected: Option<bool>, enabled: bool, color: Option<egui::Color32>) -> egui::Response {
    egui_extras::install_image_loaders(ui.ctx());
    let tint = if !enabled { ui.visuals().weak_text_color() } else { color.unwrap_or(ui.visuals().widgets.inactive.fg_stroke.color) };
    let mut button = egui::Button::image(image(name, 16.0, tint)).min_size(egui::vec2(24.0, 24.0));
    if let Some(on) = selected {
        button = button.selected(on);
    }
    let response = ui.add_enabled(enabled, button);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, enabled, label));
    response
}

/// A menu entry with an icon, its text and an optional shortcut hint.
/// A tab-like toggle with an icon in front of its text.
pub fn text_tab(ui: &mut egui::Ui, name: &str, text: &str, selected: bool) -> egui::Response {
    egui_extras::install_image_loaders(ui.ctx());
    let tint = if selected { ui.visuals().strong_text_color() } else { ui.visuals().text_color() };
    ui.add(egui::Button::image_and_text(image(name, 14.0, tint), text).selected(selected))
}

/// A submenu entry with an icon.
pub fn submenu(ui: &mut egui::Ui, name: &str, text: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui_extras::install_image_loaders(ui.ctx());
    let tint = ui.visuals().text_color();
    egui::containers::menu::SubMenuButton::from_button(egui::Button::image_and_text(image(name, 16.0, tint), text)).ui(ui, add_contents);
}

pub fn menu_item(ui: &mut egui::Ui, name: &str, text: &str, shortcut: &str) -> egui::Response {
    menu_item_enabled(ui, name, text, shortcut, true)
}

pub fn menu_item_enabled(ui: &mut egui::Ui, name: &str, text: &str, shortcut: &str, enabled: bool) -> egui::Response {
    egui_extras::install_image_loaders(ui.ctx());
    let tint = if enabled { ui.visuals().text_color() } else { ui.visuals().weak_text_color() };
    let mut button = egui::Button::image_and_text(image(name, 16.0, tint), text);
    if !shortcut.is_empty() {
        button = button.shortcut_text(shortcut);
    }
    ui.add_enabled(enabled, button)
}

pub fn mode_icon(mode: &str) -> &'static str {
    match mode {
        "sub" => "subtract",
        "intersect" => "intersect",
        "paint" => "paint",
        "push" => "push",
        "avoid" => "avoid",
        "emboss" => "emboss",
        "deboss" => "deboss",
        "shell" => "shell",
        _ => "add",
    }
}

/// A colour for each operation, so rows and buttons can be told apart at a glance.
pub fn mode_color(mode: &str) -> egui::Color32 {
    match mode {
        "sub" => egui::Color32::from_rgb(235, 110, 100),
        "intersect" => egui::Color32::from_rgb(235, 190, 80),
        "paint" => egui::Color32::from_rgb(200, 130, 230),
        "push" | "avoid" => egui::Color32::from_rgb(100, 170, 240),
        "emboss" | "deboss" => egui::Color32::from_rgb(90, 200, 200),
        "shell" => egui::Color32::from_rgb(240, 150, 90),
        _ => egui::Color32::from_rgb(120, 205, 125),
    }
}

pub fn brush_icon(brush: &str) -> &'static str {
    match brush {
        "cube" => "shape_cube",
        "cylinder" => "shape_cylinder",
        "prism" => "shape_prism",
        "polygon" => "shape_polygon",
        _ => "shape_sphere",
    }
}
