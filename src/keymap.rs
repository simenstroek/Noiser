//! Keyboard shortcuts the user can remap: every action has two slots, each holding a key with optional Ctrl/Shift.
//! The app keeps the active map here (set from the preferences each frame) so the editors can ask `pressed(i, Action::X)`.

use eframe::egui;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    New,
    Open,
    Save,
    SaveAs,
    Undo,
    Redo,
    Library,
    Delete,
    Duplicate,
    SelectAll,
    Mute,
    CommentFrame,
    NodesFrameSelected,
    NodesFrameAll,
    ToolMove,
    ToolRotate,
    ToolScale,
    Group,
    ViewFront,
    ViewRight,
    ViewTop,
    ComposerFrameAll,
    ComposerFrameSelection,
}

/// Where an action works; two actions can share a key as long as they never work in the same place.
const GRAPH: u8 = 1;
const COMPOSER: u8 = 2;
const EVERYWHERE: u8 = 7;

impl Action {
    pub const ALL: [Action; 23] = [
        Action::New,
        Action::Open,
        Action::Save,
        Action::SaveAs,
        Action::Undo,
        Action::Redo,
        Action::Library,
        Action::Delete,
        Action::Duplicate,
        Action::SelectAll,
        Action::Mute,
        Action::CommentFrame,
        Action::NodesFrameSelected,
        Action::NodesFrameAll,
        Action::ToolMove,
        Action::ToolRotate,
        Action::ToolScale,
        Action::Group,
        Action::ViewFront,
        Action::ViewRight,
        Action::ViewTop,
        Action::ComposerFrameAll,
        Action::ComposerFrameSelection,
    ];

    /// The stable name the action is saved under.
    pub fn id(self) -> &'static str {
        match self {
            Action::New => "new",
            Action::Open => "open",
            Action::Save => "save",
            Action::SaveAs => "save_as",
            Action::Undo => "undo",
            Action::Redo => "redo",
            Action::Library => "library",
            Action::Delete => "delete",
            Action::Duplicate => "duplicate",
            Action::SelectAll => "select_all",
            Action::Mute => "mute",
            Action::CommentFrame => "comment_frame",
            Action::NodesFrameSelected => "nodes_frame_selected",
            Action::NodesFrameAll => "nodes_frame_all",
            Action::ToolMove => "tool_move",
            Action::ToolRotate => "tool_rotate",
            Action::ToolScale => "tool_scale",
            Action::Group => "group",
            Action::ViewFront => "view_front",
            Action::ViewRight => "view_right",
            Action::ViewTop => "view_top",
            Action::ComposerFrameAll => "composer_frame_all",
            Action::ComposerFrameSelection => "composer_frame_selection",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Action::New => "New",
            Action::Open => "Open",
            Action::Save => "Save",
            Action::SaveAs => "Save as",
            Action::Undo => "Undo",
            Action::Redo => "Redo",
            Action::Library => "Open the library drawer",
            Action::Delete => "Delete the selection",
            Action::Duplicate => "Duplicate the selection",
            Action::SelectAll => "Select all nodes",
            Action::Mute => "Mute (bypass) the selected nodes",
            Action::CommentFrame => "Comment frame around the selection",
            Action::NodesFrameSelected => "Frame the selected nodes",
            Action::NodesFrameAll => "Frame all nodes",
            Action::ToolMove => "Move tool",
            Action::ToolRotate => "Rotate tool",
            Action::ToolScale => "Scale tool",
            Action::Group => "Group the selected shapes",
            Action::ViewFront => "Look at the front",
            Action::ViewRight => "Look at the right side",
            Action::ViewTop => "Look from the top",
            Action::ComposerFrameAll => "Frame the whole composition",
            Action::ComposerFrameSelection => "Frame the selected shape",
        }
    }

    fn places(self) -> u8 {
        match self {
            Action::New | Action::Open | Action::Save | Action::SaveAs | Action::Undo | Action::Redo | Action::Library => EVERYWHERE,
            Action::Delete | Action::Duplicate => GRAPH | COMPOSER,
            Action::SelectAll | Action::Mute | Action::CommentFrame | Action::NodesFrameSelected | Action::NodesFrameAll => GRAPH,
            _ => COMPOSER,
        }
    }

    /// The heading the action is listed under.
    pub fn group(self) -> &'static str {
        match self.places() {
            EVERYWHERE => "Files, history and the library",
            GRAPH => "Graph editor",
            COMPOSER => "SDF Composer",
            _ => "Graph editor and SDF Composer",
        }
    }

    fn defaults(self) -> [Option<Binding>; 2] {
        use egui::Key as K;
        match self {
            Action::New => [Some(Binding::ctrl(K::N)), None],
            Action::Open => [Some(Binding::ctrl(K::O)), None],
            Action::Save => [Some(Binding::ctrl(K::S)), None],
            Action::SaveAs => [Some(Binding { key: K::S, ctrl: true, shift: true }), None],
            Action::Undo => [Some(Binding::ctrl(K::Z)), None],
            Action::Redo => [Some(Binding::ctrl(K::Y)), Some(Binding { key: K::Z, ctrl: true, shift: true })],
            Action::Library => [Some(Binding::plain(K::Tab)), None],
            Action::Delete => [Some(Binding::plain(K::Delete)), Some(Binding::plain(K::X))],
            Action::Duplicate => [Some(Binding::ctrl(K::D)), None],
            Action::SelectAll => [Some(Binding::ctrl(K::A)), None],
            Action::Mute => [Some(Binding::plain(K::M)), None],
            Action::CommentFrame => [Some(Binding::plain(K::C)), None],
            Action::NodesFrameSelected => [Some(Binding::plain(K::F)), None],
            Action::NodesFrameAll => [Some(Binding::plain(K::Home)), None],
            Action::ToolMove => [Some(Binding::plain(K::W)), None],
            Action::ToolRotate => [Some(Binding::plain(K::E)), None],
            Action::ToolScale => [Some(Binding::plain(K::R)), None],
            Action::Group => [Some(Binding::ctrl(K::G)), None],
            Action::ViewFront => [Some(Binding::plain(K::Num1)), None],
            Action::ViewRight => [Some(Binding::plain(K::Num3)), None],
            Action::ViewTop => [Some(Binding::plain(K::Num7)), None],
            Action::ComposerFrameAll => [Some(Binding::plain(K::F)), None],
            Action::ComposerFrameSelection => [Some(Binding::plain(K::C)), None],
        }
    }
}

/// A key with optional modifiers (Ctrl is Cmd on a Mac).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub key: egui::Key,
    pub ctrl: bool,
    pub shift: bool,
}

impl Binding {
    pub const fn plain(key: egui::Key) -> Binding {
        Binding { key, ctrl: false, shift: false }
    }

    pub const fn ctrl(key: egui::Key) -> Binding {
        Binding { key, ctrl: true, shift: false }
    }

    pub fn text(self) -> String {
        format!("{}{}{}", if self.ctrl { "Ctrl+" } else { "" }, if self.shift { "Shift+" } else { "" }, self.key.name())
    }

    pub fn parse(text: &str) -> Option<Binding> {
        let (mut rest, mut binding) = (text.trim(), Binding::plain(egui::Key::A));
        loop {
            if let Some(r) = rest.strip_prefix("Ctrl+") {
                (binding.ctrl, rest) = (true, r);
            } else if let Some(r) = rest.strip_prefix("Shift+") {
                (binding.shift, rest) = (true, r);
            } else {
                break;
            }
        }
        binding.key = egui::Key::from_name(rest)?;
        Some(binding)
    }

    fn modifiers_match(self, modifiers: &egui::Modifiers) -> bool {
        modifiers.command == self.ctrl && modifiers.shift == self.shift
    }

    pub fn pressed(self, input: &egui::InputState) -> bool {
        input.key_pressed(self.key) && self.modifiers_match(&input.modifiers)
    }
}

type Slots = [Option<Binding>; 2];

/// What is saved: action name to the two bindings as text, empty for an unused slot.
type Saved = BTreeMap<String, [String; 2]>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "Saved", into = "Saved")]
pub struct Keymap {
    slots: BTreeMap<Action, Slots>,
}

impl Default for Keymap {
    fn default() -> Self {
        Keymap { slots: Action::ALL.iter().map(|&a| (a, a.defaults())).collect() }
    }
}

impl From<Saved> for Keymap {
    fn from(saved: Saved) -> Keymap {
        let mut keymap = Keymap::default();
        for action in Action::ALL {
            if let Some(texts) = saved.get(action.id()) {
                keymap.slots.insert(action, [Binding::parse(&texts[0]), Binding::parse(&texts[1])]);
            }
        }
        keymap
    }
}

impl From<Keymap> for Saved {
    fn from(keymap: Keymap) -> Saved {
        keymap.slots.iter().map(|(action, slots)| (action.id().to_string(), slots.map(|b| b.map_or(String::new(), Binding::text)))).collect()
    }
}

impl Keymap {
    pub fn get(&self, action: Action) -> Slots {
        self.slots.get(&action).copied().unwrap_or_default()
    }

    pub fn set(&mut self, action: Action, slot: usize, binding: Option<Binding>) {
        let mut slots = self.get(action);
        slots[slot] = binding;
        self.slots.insert(action, slots);
    }

    pub fn is_default(&self) -> bool {
        *self == Keymap::default()
    }

    pub fn pressed(&self, input: &egui::InputState, action: Action) -> bool {
        self.get(action).iter().flatten().any(|b| b.pressed(input))
    }

    /// Both bindings as one line, like `F / Home`; `unbound` when neither slot is set.
    pub fn text(&self, action: Action) -> String {
        let texts: Vec<String> = self.get(action).iter().flatten().map(|b| b.text()).collect();
        if texts.is_empty() { "unbound".to_string() } else { texts.join(" / ") }
    }

    /// Another action that works in the same place and has this binding.
    pub fn conflict(&self, action: Action, binding: Binding) -> Option<Action> {
        Action::ALL.into_iter().find(|&other| other != action && other.places() & action.places() != 0 && self.get(other).contains(&Some(binding)))
    }
}

static ACTIVE: LazyLock<RwLock<Keymap>> = LazyLock::new(|| RwLock::new(Keymap::default()));
static CAPTURING: AtomicBool = AtomicBool::new(false);

/// Makes `keymap` the one the editors read.
pub fn activate(keymap: &Keymap) {
    if ACTIVE.read().is_ok_and(|active| *active == *keymap) {
        return;
    }
    if let Ok(mut active) = ACTIVE.write() {
        *active = keymap.clone();
    }
}

/// While a binding is being recorded in the preferences, no shortcut fires.
pub fn set_capturing(capturing: bool) {
    CAPTURING.store(capturing, Ordering::Relaxed);
}

pub fn pressed(input: &egui::InputState, action: Action) -> bool {
    !CAPTURING.load(Ordering::Relaxed) && ACTIVE.read().is_ok_and(|keymap| keymap.pressed(input, action))
}

/// Like `pressed`, and takes the key press so nothing else sees it.
pub fn consume(input: &mut egui::InputState, action: Action) -> bool {
    if CAPTURING.load(Ordering::Relaxed) {
        return false;
    }
    let slots = ACTIVE.read().map(|keymap| keymap.get(action)).unwrap_or_default();
    let mut fired = false;
    for binding in slots.into_iter().flatten() {
        let modifiers = egui::Modifiers { command: binding.ctrl, shift: binding.shift, ..Default::default() };
        fired |= input.consume_key(modifiers, binding.key);
    }
    fired
}

pub fn text(action: Action) -> String {
    ACTIVE.read().map_or(String::new(), |keymap| keymap.text(action))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_name_reads_back_as_the_same_key() {
        for key in egui::Key::ALL {
            assert_eq!(Binding::parse(&Binding::plain(*key).text()).map(|b| b.key), Some(*key), "{key:?} is called {:?}", key.name());
        }
    }

    #[test]
    fn modifiers_survive_the_text_form() {
        let binding = Binding { key: egui::Key::Z, ctrl: true, shift: true };
        assert_eq!(binding.text(), "Ctrl+Shift+Z");
        assert_eq!(Binding::parse("Ctrl+Shift+Z"), Some(binding));
        assert_eq!(Binding::parse("Bogus"), None);
    }

    #[test]
    fn the_defaults_include_x_beside_delete_and_both_frame_keys() {
        let keymap = Keymap::default();
        assert_eq!(keymap.get(Action::Delete), [Some(Binding::plain(egui::Key::Delete)), Some(Binding::plain(egui::Key::X))]);
        assert_eq!(keymap.text(Action::Redo), "Ctrl+Y / Ctrl+Shift+Z");
        assert_eq!(keymap.get(Action::NodesFrameAll)[0], Some(Binding::plain(egui::Key::Home)));
        assert!(keymap.is_default());
    }

    #[test]
    fn the_defaults_do_not_clash_with_themselves() {
        let keymap = Keymap::default();
        for action in Action::ALL {
            for binding in keymap.get(action).into_iter().flatten() {
                assert_eq!(keymap.conflict(action, binding), None, "{action:?} {binding:?}");
            }
        }
    }

    #[test]
    fn a_changed_map_is_saved_and_loaded_and_new_actions_get_their_defaults() {
        let mut keymap = Keymap::default();
        keymap.set(Action::Mute, 0, Some(Binding::plain(egui::Key::B)));
        keymap.set(Action::Mute, 1, None);
        keymap.set(Action::Delete, 1, None);
        let json = serde_json::to_string(&keymap).unwrap();
        assert_eq!(serde_json::from_str::<Keymap>(&json).unwrap(), keymap);
        let partial: Keymap = serde_json::from_str(r#"{"mute":["B",""]}"#).unwrap();
        assert_eq!(partial.get(Action::Mute), [Some(Binding::plain(egui::Key::B)), None]);
        assert_eq!(partial.get(Action::Delete), Action::Delete.defaults());
    }

    #[test]
    fn a_shared_key_conflicts_only_where_both_actions_work() {
        let keymap = Keymap::default();
        // C frames a selection in the composer and makes a comment frame in the graph editor: never both at once.
        assert_eq!(keymap.conflict(Action::CommentFrame, Binding::plain(egui::Key::C)), None);
        assert_eq!(keymap.conflict(Action::Mute, Binding::plain(egui::Key::Delete)), Some(Action::Delete));
        assert_eq!(keymap.conflict(Action::ToolMove, Binding::plain(egui::Key::X)), Some(Action::Delete));
    }

    #[test]
    fn pressing_matches_the_modifiers_exactly() {
        let mut input = egui::InputState::default();
        input.keys_down.insert(egui::Key::Z);
        input.modifiers = egui::Modifiers::COMMAND;
        input.events.push(egui::Event::Key { key: egui::Key::Z, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::COMMAND });
        let keymap = Keymap::default();
        assert!(keymap.pressed(&input, Action::Undo));
        assert!(!keymap.pressed(&input, Action::Redo));
    }
}
