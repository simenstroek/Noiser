//! Reading and writing SDF Modeler (.sdf) files, format 0.6.x.
//!
//! A file holds assemblies (a tree of shapes each) and layers (a placement and material of one assembly). Only what
//! Noiser edits is typed; every other field is kept as it was read, so saving a file does not lose what SDF Modeler
//! stored in it. Files from before 0.6 use another layout and are refused with a message saying so.

#![allow(dead_code)] // until the SDF Composer and the Load SDF node use it

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SdfFile {
    pub version: [u32; 3],
    pub environment: Value,
    pub assemblies: Vec<Assembly>,
    pub layers: Vec<Layer>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assembly {
    pub resolution: u32,
    pub shapes: Vec<Node>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Places an assembly in the scene and gives it a material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub assembly: usize,
    pub hidden: bool,
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One entry of an assembly's tree: a shape, a group, a spline or a spline's control point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub hidden: bool,
    #[serde(rename = "type")]
    pub kind: String,
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
    pub color: [f64; 3],
    /// The shape's own settings (mode, blend, brush, bevel, round, repeats …), as stored.
    #[serde(flatten)]
    pub params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Node>>,
}

impl Node {
    pub fn number(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(Value::as_f64)
    }

    pub fn flag(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(Value::as_bool)
    }

    pub fn text(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(Value::as_str)
    }

    pub fn numbers<const N: usize>(&self, key: &str) -> Option<[f64; N]> {
        let items = self.params.get(key)?.as_array()?;
        let mut out = [0.0; N];
        if items.len() != N {
            return None;
        }
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = item.as_f64()?;
        }
        Some(out)
    }

    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        self.params.insert(key.to_string(), value.into());
    }

    pub fn children(&self) -> &[Node] {
        self.children.as_deref().unwrap_or(&[])
    }
}

impl SdfFile {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text).map_err(|e| format!("not a valid .sdf file: {e}"))?;
        let Some(object) = value.as_object() else {
            return Err(LEGACY.into());
        };
        let Some(version) = object.get("version").and_then(Value::as_array) else {
            return Err(LEGACY.into());
        };
        let major_minor = (version.first().and_then(Value::as_u64), version.get(1).and_then(Value::as_u64));
        if major_minor != (Some(0), Some(6)) {
            return Err(format!("this .sdf file is version {}; Noiser reads SDF Modeler 0.6.x files", Value::Array(version.clone())));
        }
        serde_json::from_value(value).map_err(|e| format!("this .sdf file does not match the 0.6 layout: {e}"))
    }

    pub fn to_json(&self) -> String {
        use serde::Serialize as _;
        let mut bytes = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"\t");
        let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, formatter);
        self.serialize(&mut serializer).expect("a document of plain values serializes");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json()).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

const LEGACY: &str = "this .sdf file is from an SDF Modeler older than 0.6; open it in SDF Modeler 0.6.1 and save it again";

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = r#"{
        "version": [0, 6, 1],
        "environment": {"color": [0.3, 0.3, 0.3], "future_setting": 7},
        "assemblies": [{"resolution": 5, "future_field": "kept", "shapes": [
            {"name": "g", "hidden": false, "type": "group", "position": [0,0,0], "rotation": [0,0,0], "scale": [1,1,1],
             "color": [0.8,0.8,0.8], "mode": "add", "blend": 0.02, "children": [
                {"name": "", "hidden": false, "type": "shape", "position": [1,2,3], "rotation": [0,0,0], "scale": [1,1,1],
                 "color": [1,1,1], "mode": "sub", "brush": "cube", "round": 0.5, "linear_repeats": [1,1,1], "chamfer": true}]}]}],
        "layers": [{"name": "L", "assembly": 0, "hidden": false, "position": [0,0,0], "rotation": [0,0,0], "scale": [1,1,1], "roughness": 0.3}]
    }"#;

    /// The document with every number as a float, so `1` and `1.0` compare equal.
    fn by_value(value: Value) -> Value {
        match value {
            Value::Number(n) => serde_json::Number::from_f64(n.as_f64().unwrap()).map_or(Value::Null, Value::Number),
            Value::Array(items) => Value::Array(items.into_iter().map(by_value).collect()),
            Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, by_value(v))).collect()),
            other => other,
        }
    }

    /// Where two documents first differ, for a readable failure.
    fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                for (key, value) in x {
                    match y.get(key) {
                        Some(other) => {
                            if let Some(found) = first_difference(value, other, &format!("{path}/{key}")) {
                                return Some(found);
                            }
                        }
                        None => return Some(format!("{path}/{key} is missing after the round trip")),
                    }
                }
                y.keys().find(|k| !x.contains_key(*k)).map(|k| format!("{path}/{k} appeared"))
            }
            (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x.iter().zip(y).enumerate().find_map(|(i, (p, q))| first_difference(p, q, &format!("{path}[{i}]"))),
            _ => (a != b).then(|| format!("{path}: {a} became {b}")),
        }
    }

    #[test]
    fn unknown_fields_survive_a_round_trip() {
        let file = SdfFile::from_json(SMALL).unwrap();
        let again = SdfFile::from_json(&file.to_json()).unwrap();
        assert_eq!(file, again);
        let original: Value = serde_json::from_str(SMALL).unwrap();
        let written: Value = serde_json::from_str(&file.to_json()).unwrap();
        assert_eq!(by_value(original), by_value(written));
    }

    #[test]
    fn nodes_expose_their_settings() {
        let file = SdfFile::from_json(SMALL).unwrap();
        let group = &file.assemblies[0].shapes[0];
        assert_eq!(group.kind, "group");
        let cube = &group.children()[0];
        assert_eq!(cube.text("brush"), Some("cube"));
        assert_eq!(cube.number("round"), Some(0.5));
        assert_eq!(cube.flag("chamfer"), Some(true));
        assert_eq!(cube.numbers::<3>("linear_repeats"), Some([1.0; 3]));
        assert_eq!(cube.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn files_from_before_0_6_are_refused_with_an_explanation() {
        let old = r#"[{"name": "Layer", "strokes": []}]"#;
        assert!(SdfFile::from_json(old).unwrap_err().contains("older than 0.6"));
        let no_version = r#"{"environment": {}, "layers": []}"#;
        assert!(SdfFile::from_json(no_version).unwrap_err().contains("older than 0.6"));
        assert!(SdfFile::from_json(r#"{"version": [0, 7, 0]}"#).unwrap_err().contains("0.6.x"));
    }

    /// Every 0.6 file in SDF Modeler's sample folder (and `NOISER_SDF_DIR`, if set) keeps all of its content.
    #[test]
    fn sdf_modeler_sample_files_round_trip() {
        let mut dirs = vec![];
        if let Ok(dir) = std::env::var("NOISER_SDF_DIR") {
            dirs.push(std::path::PathBuf::from(dir));
        }
        if let Ok(home) = std::env::var("USERPROFILE") {
            dirs.push(Path::new(&home).join("Documents").join("SDF Modeler").join("sdfs"));
        }
        let mut checked = 0;
        for dir in dirs {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for path in entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "sdf")) {
                let text = std::fs::read_to_string(&path).unwrap();
                let original: Value = serde_json::from_str(&text).unwrap();
                let Ok(file) = SdfFile::from_json(&text) else {
                    assert!(original.get("version").is_none(), "{} is 0.6 but was refused", path.display());
                    continue;
                };
                let written: Value = serde_json::from_str(&file.to_json()).unwrap();
                if let Some(difference) = first_difference(&by_value(original), &by_value(written), "") {
                    panic!("{} changed in a round trip at {difference}", path.display());
                }
                checked += 1;
            }
        }
        eprintln!("round-tripped {checked} sample file(s)");
    }
}
