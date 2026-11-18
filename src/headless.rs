//! `noiser --headless graph.noisegraph --batch-out DIR (--batch-file F ... | --batch-dir SRC) [options]`:
//! displaces meshes with a graph without opening a window, for scripts and render farms.
//! Progress goes to the console (when started from one) and to `noiser-batch.log` in the output folder.

use crate::graph::Graph;
use eframe::wgpu;
use crate::meshexport::{Displace, FileJob, Format, Progress, SeedMode, Settings};
use crate::meshmask::{MaskKind, MaskSettings, MaskTerm};
use std::io::Write;
use std::path::PathBuf;

pub const USAGE: &str = "usage: noiser --headless GRAPH.noisegraph --batch-out DIR (--batch-file F.obj ... | --batch-dir SRC)
  [--export-noiser OUT.noiser] [--set 3.scale=8 ...] [--amount 0.25] [--amount-var 0.2] [--tile 1.0] [--size 1.0] [--seed 0] [--seed-mode same|increment|random]
  [--res 256] [--grid-res 128] [--format obj|ply] [--cull 0.01] [--mask KIND ...]";

#[derive(Debug)]
struct Options {
    graph: PathBuf,
    out: PathBuf,
    files: Vec<PathBuf>,
    amount: f32,
    tile: f32,
    size: f32,
    seed: u32,
    mode: SeedMode,
    amount_var: f32,
    /// `--export-noiser FILE`: write the graph as a .noiser math function instead of displacing meshes.
    export_noiser: Option<PathBuf>,
    /// `--set NODE.FIELD=VALUE` overrides of node settings in the graph file.
    sets: Vec<(String, String, serde_json::Value)>,
    settings: Settings,
    grid_res: u32,
    mask: MaskSettings,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        graph: PathBuf::new(),
        out: PathBuf::new(),
        files: Vec::new(),
        amount: 0.25,
        tile: 1.0,
        size: 1.0,
        seed: 0,
        mode: SeedMode::Same,
        amount_var: 0.0,
        export_noiser: None,
        sets: Vec::new(),
        settings: Settings::default(),
        grid_res: 128,
        mask: MaskSettings::default(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |what: &str| it.next().cloned().ok_or(format!("{a} needs {what}"));
        let float = |s: String| s.parse::<f32>().map_err(|_| format!("{a}: {s} is not a number"));
        let whole = |s: String| s.parse::<u32>().map_err(|_| format!("{a}: {s} is not a whole number"));
        match a.as_str() {
            "--headless" => {}
            "--batch-out" => o.out = value("a folder")?.into(),
            "--batch-file" => o.files.push(value("a file")?.into()),
            "--batch-dir" => {
                let dir = PathBuf::from(value("a folder")?);
                let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .map_err(|e| format!("{}: {e}", dir.display()))?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| crate::meshformats::is_mesh_file(p))
                    .collect();
                found.sort();
                o.files.extend(found);
            }
            "--amount" => o.amount = float(value("a number")?)?,
            "--tile" => o.tile = float(value("a number")?)?,
            "--size" => o.size = float(value("a number")?)?,
            "--export-noiser" => o.export_noiser = Some(value("a file")?.into()),
            "--set" => {
                let spec = value("NODE.FIELD=VALUE")?;
                let (target, v) = spec.split_once('=').ok_or(format!("--set {spec}: expected NODE.FIELD=VALUE"))?;
                let (node, field) = target.split_once('.').ok_or(format!("--set {spec}: expected NODE.FIELD=VALUE"))?;
                // Numbers, booleans and quoted strings are JSON; anything else is taken as a plain string (an enum name).
                let value = serde_json::from_str(v).unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
                o.sets.push((node.to_string(), field.to_string(), value));
            }
            "--amount-var" => o.amount_var = float(value("a fraction like 0.2")?)?.clamp(0.0, 0.5),
            "--cull" => o.settings.cull = float(value("a number")?)?,
            "--seed" => o.seed = whole(value("a number")?)?,
            "--res" => o.settings.res = whole(value("a number")?)?,
            "--grid-res" => o.grid_res = whole(value("a number")?)?,
            "--seed-mode" => {
                o.mode = match value("same, increment or random")?.as_str() {
                    "same" => SeedMode::Same,
                    "increment" => SeedMode::Increment,
                    "random" => SeedMode::Random,
                    other => return Err(format!("unknown seed mode {other}")),
                }
            }
            "--format" => {
                o.settings.format = match value("obj or ply")?.as_str() {
                    "obj" => Format::Obj,
                    "ply" => Format::Ply,
                    other => return Err(format!("unknown format {other}")),
                }
            }
            "--mask" => {
                let name = value("a term name")?.to_lowercase();
                let kind = MaskKind::ALL.iter().find(|k| k.name().to_lowercase().starts_with(&name)).ok_or(format!("unknown mask term {name}"))?;
                o.mask.terms.push(MaskTerm::new(*kind));
            }
            other if !other.starts_with("--") && o.graph.as_os_str().is_empty() => o.graph = other.into(),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if o.graph.as_os_str().is_empty() {
        return Err("no graph file given".into());
    }
    if o.export_noiser.is_some() {
        return Ok(o);
    }
    if o.out.as_os_str().is_empty() {
        return Err("--batch-out is required".into());
    }
    if o.files.is_empty() {
        return Err("no meshes: use --batch-file or --batch-dir".into());
    }
    Ok(o)
}

/// The graph file with node settings overridden: `--set 3.scale=8` sets the `scale` field of node 3 (the names
/// are those in the .noisegraph file; a Value node has `value`). An unknown node or field is an error.
fn apply_sets(text: &str, sets: &[(String, String, serde_json::Value)]) -> Result<String, String> {
    if sets.is_empty() {
        return Ok(text.to_string());
    }
    let mut doc: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    for (node, field, value) in sets {
        let kind = doc["nodes"].get_mut(node.as_str()).and_then(|n| n.get_mut("kind")).ok_or(format!("--set: there is no node {node}"))?;
        match kind {
            // A Value node is {"Float": 1.0}.
            serde_json::Value::Object(m) if m.len() == 1 && field == "value" && m.values().next().is_some_and(|v| v.is_number()) => {
                *m.values_mut().next().unwrap() = value.clone();
            }
            serde_json::Value::Object(m) => {
                let fields = m.values_mut().next().and_then(|v| v.as_object_mut()).ok_or(format!("--set: node {node} has no settings"))?;
                let slot = fields.get_mut(field.as_str()).ok_or(format!("--set: node {node} has no setting {field}"))?;
                *slot = value.clone();
            }
            _ => return Err(format!("--set: node {node} has no settings")),
        }
    }
    Ok(doc.to_string())
}

/// Runs the batch when `--headless` is among the arguments; returns the process exit code.
pub fn try_run() -> Option<i32> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|a| a == "--headless") {
        return None;
    }
    attach_console();
    let mut log = Log::default();
    let code = match run(&args, &mut log) {
        Ok(failed) => i32::from(failed > 0),
        Err(e) => {
            log.line(&format!("error: {e}\n{USAGE}"));
            2
        }
    };
    Some(code)
}

fn run(args: &[String], log: &mut Log) -> Result<usize, String> {
    let o = parse(args)?;
    let text = std::fs::read_to_string(&o.graph).map_err(|e| format!("{}: {e}", o.graph.display()))?;
    let graph = Graph::from_json(&apply_sets(&text, &o.sets)?)?;
    if let Some(path) = &o.export_noiser {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))?;
        let ev = crate::export::Evaluator::new(&device, &queue, &graph, o.seed)?;
        let text = crate::noiser_file::file(&graph, o.seed, |pts| ev.eval_points(pts))?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        log.line(&format!("wrote {}", path.display()));
        return Ok(0);
    }
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    log.file = std::fs::OpenOptions::new().create(true).append(true).open(o.out.join("noiser-batch.log")).ok();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| format!("no GPU device: {e}"))?;
    let d = Displace { amount: o.amount, tile_size: o.tile, prim_size: o.size, mask: o.mask.pack() };
    let total = o.files.len();
    let mut failed = 0;
    for (i, input) in o.files.iter().enumerate() {
        let stem = input.file_stem().map_or("mesh".into(), |s| s.to_string_lossy().into_owned());
        let seed = o.mode.seed_for(o.seed, i);
        let label = if o.mode == SeedMode::Same { stem.clone() } else { format!("{stem}_seed{seed}") };
        let output = crate::meshapp::unique_path(&o.out, &format!("{label}_displaced"), o.settings.format.extension());
        let d = Displace { amount: d.amount * crate::meshexport::amount_factor(o.seed, i, o.amount_var), ..d };
        let job = FileJob { device: &device, queue: &queue, graph: &graph, seed, d, settings: o.settings.clone(), grid_res: o.grid_res, grid_pad: 0.5 };
        let started = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(input, &output, &Progress::default())));
        let n = i + 1;
        match result {
            Ok(Ok(info)) => log.line(&format!("[{n}/{total}] {} -> {} ({info}, {:.1} s)", input.display(), output.display(), started.elapsed().as_secs_f32())),
            Ok(Err(e)) => {
                failed += 1;
                log.line(&format!("[{n}/{total}] {} FAILED: {e}", input.display()));
            }
            Err(_) => {
                failed += 1;
                log.line(&format!("[{n}/{total}] {} FAILED: the export crashed", input.display()));
            }
        }
    }
    log.line(&format!("done: {} of {total} meshes written to {}", total - failed, o.out.display()));
    Ok(failed)
}

/// Where progress lines go: the console if there is one, and the log file once known.
#[derive(Default)]
struct Log {
    file: Option<std::fs::File>,
}

impl Log {
    fn line(&mut self, s: &str) {
        let console = if cfg!(windows) { "CONOUT$" } else { "/dev/stdout" };
        if let Ok(mut con) = std::fs::OpenOptions::new().write(true).open(console) {
            let _ = writeln!(con, "{s}");
        }
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{s}");
        }
    }
}

/// The release build is a GUI-subsystem program with no console of its own; borrow the one it was started from.
fn attach_console() {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn AttachConsole(process_id: u32) -> i32;
        }
        // SAFETY: a plain Win32 call with the documented "parent process" argument.
        unsafe { AttachConsole(u32::MAX) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn options_are_parsed_and_checked() {
        let o = parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --seed 7 --seed-mode random --format ply --mask thickness")).unwrap();
        assert_eq!((o.seed, o.mode, o.settings.format, o.mask.terms.len(), o.files.len()), (7, SeedMode::Random, Format::Ply, 1, 1));
        assert!(parse(&args("--headless g.noisegraph --batch-file a.obj")).unwrap_err().contains("--batch-out"));
        assert!(parse(&args("--headless --batch-out out --batch-file a.obj")).unwrap_err().contains("graph"));
        assert!(parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --seed x")).is_err());
        assert!(parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --bogus")).is_err());
    }

    #[test]
    fn set_overrides_node_settings_and_rejects_unknown_ones() {
        let mut g = Graph::default_graph(); // 0 Position, 1 Noise, 2 and 3 outputs
        let float = g.add(crate::graph::NodeKind::Float(1.0), [0.0; 2]);
        let text = g.to_json();
        let set = |node: &str, field: &str, v: serde_json::Value| vec![(node.to_string(), field.to_string(), v)];
        let patched = Graph::from_json(&apply_sets(&text, &set("1", "scale", serde_json::json!(8.0))).unwrap()).unwrap();
        assert!(matches!(&patched.nodes[&1].kind, crate::graph::NodeKind::Noise(p) if p.scale == 8.0));
        let patched = Graph::from_json(&apply_sets(&text, &set(&float.to_string(), "value", serde_json::json!(2.5))).unwrap()).unwrap();
        assert_eq!(patched.nodes[&float].kind, crate::graph::NodeKind::Float(2.5));
        assert!(apply_sets(&text, &set("1", "no_such", serde_json::json!(1))).unwrap_err().contains("no setting"));
        assert!(apply_sets(&text, &set("99", "scale", serde_json::json!(1))).unwrap_err().contains("no node"));
        assert_eq!(apply_sets(&text, &[]).unwrap(), text);
        let o = parse(&args("--headless g.noisegraph --batch-out out --batch-file a.obj --set 1.scale=8 --set 1.kind=Simplex")).unwrap();
        assert_eq!(o.sets.len(), 2);
        assert_eq!(o.sets[1].2, serde_json::json!("Simplex"));
    }
}
