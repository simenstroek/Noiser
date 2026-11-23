// Material Symbols icons (Apache 2.0) fetched from Iconify, drawn white and tinted at use.
use eframe::egui;

pub fn source(name: &str) -> Option<egui::ImageSource<'static>> {
    let bytes: &'static [u8] = match name {
        "add" => include_bytes!("../assets/ui_icons/add.svg"),
        "add_layer" => include_bytes!("../assets/ui_icons/add_layer.svg"),
        "avoid" => include_bytes!("../assets/ui_icons/avoid.svg"),
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
        "global" => include_bytes!("../assets/ui_icons/global.svg"),
        "group" => include_bytes!("../assets/ui_icons/group.svg"),
        "group_row" => include_bytes!("../assets/ui_icons/group_row.svg"),
        "intersect" => include_bytes!("../assets/ui_icons/intersect.svg"),
        "local" => include_bytes!("../assets/ui_icons/local.svg"),
        "move" => include_bytes!("../assets/ui_icons/move.svg"),
        "new" => include_bytes!("../assets/ui_icons/new.svg"),
        "open" => include_bytes!("../assets/ui_icons/open.svg"),
        "paint" => include_bytes!("../assets/ui_icons/paint.svg"),
        "push" => include_bytes!("../assets/ui_icons/push.svg"),
        "redo" => include_bytes!("../assets/ui_icons/redo.svg"),
        "rotate" => include_bytes!("../assets/ui_icons/rotate.svg"),
        "save" => include_bytes!("../assets/ui_icons/save.svg"),
        "save_as" => include_bytes!("../assets/ui_icons/save_as.svg"),
        "scale" => include_bytes!("../assets/ui_icons/scale.svg"),
        "shape_cube" => include_bytes!("../assets/ui_icons/shape_cube.svg"),
        "shape_cylinder" => include_bytes!("../assets/ui_icons/shape_cylinder.svg"),
        "shape_polygon" => include_bytes!("../assets/ui_icons/shape_polygon.svg"),
        "shape_prism" => include_bytes!("../assets/ui_icons/shape_prism.svg"),
        "shape_sphere" => include_bytes!("../assets/ui_icons/shape_sphere.svg"),
        "shell" => include_bytes!("../assets/ui_icons/shell.svg"),
        "snap" => include_bytes!("../assets/ui_icons/snap.svg"),
        "subtract" => include_bytes!("../assets/ui_icons/subtract.svg"),
        "undo" => include_bytes!("../assets/ui_icons/undo.svg"),
        "up" => include_bytes!("../assets/ui_icons/up.svg"),
        _ => return None,
    };
    Some(egui::ImageSource::Bytes { uri: format!("bytes://icons/{name}.svg").into(), bytes: egui::load::Bytes::Static(bytes) })
}

pub fn image(name: &str, size: f32, tint: egui::Color32) -> egui::Image<'static> {
    let source = source(name).unwrap_or_else(|| panic!("no icon called {name}"));
    egui::Image::new(source).fit_to_exact_size(egui::vec2(size, size)).tint(tint)
}

/// A square icon button whose accessible name is `label`; the caller adds the tooltip. `selected` makes it a toggle.
pub fn button(ui: &mut egui::Ui, name: &str, label: &str, selected: Option<bool>, enabled: bool) -> egui::Response {
    let tint = if enabled { ui.visuals().widgets.inactive.fg_stroke.color } else { ui.visuals().weak_text_color() };
    let mut button = egui::Button::image(image(name, 16.0, tint)).min_size(egui::vec2(24.0, 24.0));
    if let Some(on) = selected {
        button = button.selected(on);
    }
    let response = ui.add_enabled(enabled, button);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, enabled, label));
    response
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

pub fn brush_icon(brush: &str) -> &'static str {
    match brush {
        "cube" => "shape_cube",
        "cylinder" => "shape_cylinder",
        "prism" => "shape_prism",
        "polygon" => "shape_polygon",
        _ => "shape_sphere",
    }
}
