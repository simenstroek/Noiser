//! `noiser --headless graph.noisegraph --batch-out DIR (--batch-file F ... | --batch-dir SRC) [options]`:
//! displaces meshes with a graph without opening a window, for scripts and render farms.
//! Progress goes to the console (when started from one) and to `noiser-batch.log` in the output folder.

use crate::graph::Graph;
use eframe::wgpu;
use crate::meshexport::{Displace, FileJob, Format, Progress, SeedMode, Settings};
use crate::meshmask::{MaskKind, MaskSettings, MaskTerm};
use std::io::Write;
use std::path::PathBuf;

pub const USAGE: &str = "usage: noiser --headless --mesh-graph FILE.meshgraph [--seed N] [--batch-dir SRC | --batch-file F ...]   (runs a mesh graph, once per mesh file for a batch: Load Mesh paths of {batch} and output names with {name})
   or: noiser --headless GRAPH.noisegraph --batch-out DIR (--batch-file F.obj ... | --batch-dir SRC)
  [--export-noiser OUT.noiser] [--export-shader OUT.hlsl|glsl|wgsl] [--manifest meshes.json] [--param name=value ...] [--set 3.scale=8 ...] [--amount 0.25] [--amount-var 0.2] [--tile 1.0] [--size 1.0] [--seed 0] [--seed-mode same|increment|random]
  [--res 256] [--grid-res 128] [--format obj|ply] [--cull 0.01] [--mask KIND ...]
  [--combine union|intersect|smooth-union|smooth-intersect] [--combine-smooth 0.1] [--object-units] [--report] [--remesh TRIANGLES] [--uv] [--uv-res 2048] [--uv-cost 4] [--uv-smooth 10] [--uv-merge 3]
  [--bake normal,occlusion,curvature,depth,thickness,position|all] [--bake-size 2048] [--bake-pad 8] [--green-up]";

/// One mesh of a `--manifest`: the file and any settings that differ from the command line for that mesh.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    file: PathBuf,
    amount: Option<f32>,
    tile: Option<f32>,
    size: Option<f32>,
    seed: Option<u32>,
    res: Option<u32>,
    #[serde(rename = "grid-res")]
    grid_res: Option<u32>,
    cull: Option<f32>,
    /// Mask terms for this mesh (kind names as for `--mask`); replaces the command line's.
    mask: Option<Vec<String>>,
    /// Exposed parameters of the graph for this mesh.
    #[serde(default)]
    params: std::collections::BTreeMap<String, serde_json::Value>,
}

/// Reads a manifest: a JSON list of specs, or an object with a `meshes` list. Relative files are relative to it.
fn read_manifest(path: &std::path::Path) -> Result<Vec<Spec>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let list = if value.is_array() { value } else { value.get("meshes").cloned().ok_or("the manifest needs a list of meshes")? };
    let mut specs: Vec<Spec> = serde_json::from_value(list).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    for s in &mut specs {
        if s.file.is_relative() {
            s.file = dir.join(&s.file);
        }
    }
    Ok(specs)
}

#[derive(Debug)]
struct Options {
    graph: PathBuf,
    out: PathBuf,
    files: Vec<PathBuf>,
    /// Meshes from `--manifest`, with their own settings.
    specs: Vec<Spec>,
    amount: f32,
    tile: f32,
    size: f32,
    seed: u32,
    mode: SeedMode,
    amount_var: f32,
    /// `--export-noiser FILE`: write the graph as a .noiser math function instead of displacing meshes.
    export_noiser: Option<PathBuf>,
    /// `--export-shader FILE.hlsl|glsl|wgsl`: write the graph as shader code instead of displacing meshes.
    export_shader: Option<PathBuf>,
    /// `--param name=value`: overrides of the graph's exposed parameters.
    params: Vec<(String, serde_json::Value)>,
    /// `--set NODE.FIELD=VALUE` overrides of node settings in the graph file.
    sets: Vec<(String, String, serde_json::Value)>,
    settings: Settings,
    grid_res: u32,
    combine: crate::meshsdf::Combine,
    /// `--object-units`: --tile and --amount are in the meshes' own units, so the detail is the same size on every mesh.
    object_units: bool,
    mask: MaskSettings,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        graph: PathBuf::new(),
        out: PathBuf::new(),
        files: Vec::new(),
        specs: Vec::new(),
        amount: 0.25,
        tile: 1.0,
        size: 1.0,
        seed: 0,
        mode: SeedMode::Same,
        amount_var: 0.0,
        export_noiser: None,
        export_shader: None,
        params: Vec::new(),
        sets: Vec::new(),
        settings: Settings::default(),
        grid_res: 128,
        combine: Default::default(),
        object_units: false,
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
            "--manifest" => o.specs.extend(read_manifest(std::path::Path::new(&value("a JSON file")?))?),
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
            "--export-shader" => o.export_shader = Some(value("a file")?.into()),
            "--export-noiser" => o.export_noiser = Some(value("a file")?.into()),
            "--param" => {
                let spec = value("name=value")?;
                let (name, v) = spec.split_once('=').ok_or(format!("--param {spec}: expected name=value"))?;
                o.params.push((name.to_string(), serde_json::from_str(v).unwrap_or_else(|_| serde_json::Value::String(v.to_string()))));
            }
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
            "--combine" => {
                let name = value("union, intersect, smooth-union or smooth-intersect")?;
                o.combine.mode = crate::meshsdf::CombineMode::ALL.into_iter().find(|m| m.file_name() == name).ok_or(format!("unknown combine mode {name}"))?;
            }
            "--object-units" => o.object_units = true,
            "--report" => o.settings.report = true,
            "--uv" => o.settings.uv = Some(crate::uvgen::UvParams::default()),
            "--uv-res" => o.settings.uv.get_or_insert_with(Default::default).resolution = whole(value("a texture size like 2048")?)?,
            "--uv-smooth" => o.settings.uv.get_or_insert_with(Default::default).smoothing = whole(value("a number of rounds")?)?,
            "--uv-merge" => o.settings.uv.get_or_insert_with(Default::default).merge_rounds = whole(value("a number of rounds")?)?,
            "--bake" => {
                let list = value("channel names or all")?;
                let channels = if list == "all" {
                    crate::bake::Channel::ALL.to_vec()
                } else {
                    list.split(',').map(|n| crate::bake::Channel::ALL.into_iter().find(|c| c.name().eq_ignore_ascii_case(n.trim())).ok_or(format!("unknown map {n}"))).collect::<Result<Vec<_>, _>>()?
                };
                o.settings.bake.get_or_insert_with(Default::default).channels = channels;
            }
            "--bake-size" => o.settings.bake.get_or_insert_with(Default::default).size = whole(value("a texture size like 2048")?)?,
            "--bake-pad" => o.settings.bake.get_or_insert_with(Default::default).padding = whole(value("texels")?)?,
            "--green-up" => o.settings.bake.get_or_insert_with(Default::default).green_up = true,
            "--uv-brute" => o.settings.uv.get_or_insert_with(Default::default).brute_force = true,
            "--uv-cost" => o.settings.uv.get_or_insert_with(Default::default).max_cost = float(value("a number like 4")?)?,
            "--remesh" => o.settings.remesh = Some(whole(value("a triangle count")?)?.max(100) as usize),
            "--combine-smooth" => o.combine.smooth = float(value("a width like 0.1")?)?.max(0.0),
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
    if o.export_noiser.is_some() || o.export_shader.is_some() {
        return Ok(o);
    }
    if o.out.as_os_str().is_empty() {
        return Err("--batch-out is required".into());
    }
    if o.files.is_empty() && o.specs.is_empty() {
        return Err("no meshes: use --batch-file, --batch-dir or --manifest".into());
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

/// Mask terms from kind names (`thickness`, `position`...), as `--mask` takes them.
fn mask_from_names(names: &[String]) -> Result<MaskSettings, String> {
    let mut m = MaskSettings::default();
    for name in names {
        let lower = name.to_lowercase();
        let kind = MaskKind::ALL.iter().find(|k| k.name().to_lowercase().starts_with(&lower)).ok_or(format!("unknown mask term {name}"))?;
        m.terms.push(MaskTerm::new(*kind));
    }
    Ok(m)
}

/// `--param name=value` turned into `--set` overrides through the graph file's exposed parameters.
fn resolve_params(text: &str, params: &[(String, serde_json::Value)]) -> Result<Vec<(String, String, serde_json::Value)>, String> {
    if params.is_empty() {
        return Ok(Vec::new());
    }
    let graph = Graph::from_json(text)?;
    params
        .iter()
        .map(|(name, v)| {
            let e = graph.exposed.iter().find(|e| e.name == *name).ok_or_else(|| {
                let known: Vec<&str> = graph.exposed.iter().map(|e| e.name.as_str()).collect();
                format!("--param: the graph exposes no parameter named {name} (it exposes: {})", if known.is_empty() { "none".into() } else { known.join(", ") })
            })?;
            Ok((e.node.to_string(), e.field.clone(), v.clone()))
        })
        .collect()
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

/// `noiser --headless --mesh-graph FILE.meshgraph [--seed N]`: runs a mesh graph (see meshgraph.rs).
fn run_mesh_graph(args: &[String], log: &mut Log) -> Result<usize, String> {
    let at = args.iter().position(|a| a == "--mesh-graph").unwrap();
    let file = PathBuf::from(args.get(at + 1).ok_or("--mesh-graph needs a file")?);
    log.file = std::fs::OpenOptions::new().create(true).append(true).open(file.with_extension("log")).ok();
    let seed = args.iter().position(|a| a == "--seed").and_then(|i| args.get(i + 1)).map_or(Ok(0), |v| v.parse::<u32>().map_err(|_| format!("--seed {v}: not a whole number")))?;
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let graph = Graph::from_json(&text)?;
    let (device, queue) = crate::meshgraph::headless_gpu()?;
    let mut cache = crate::meshgraph::MeshCache::default();
    let base = file.parent().map_or(PathBuf::from("."), |p| p.to_path_buf());
    // Batch runs: the graph once per mesh file; without any, once.
    let mut batch: Vec<Option<PathBuf>> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--batch-file" => batch.extend(it.next().map(|f| Some(PathBuf::from(f)))),
            "--batch-dir" => {
                let dir = PathBuf::from(it.next().ok_or("--batch-dir needs a folder")?);
                let mut found: Vec<PathBuf> = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| crate::meshformats::is_mesh_file(p)).collect();
                found.sort();
                batch.extend(found.into_iter().map(Some));
            }
            _ => {}
        }
    }
    if batch.is_empty() {
        batch.push(None);
    }
    let started = std::time::Instant::now();
    let (mut failed, mut written) = (0usize, 0usize);
    for (i, file) in batch.iter().enumerate() {
        let lines = std::cell::RefCell::new(Vec::<String>::new());
        let mut note = |s: &str| lines.borrow_mut().push(s.to_string());
        let result = {
            let mut runner = crate::meshgraph::Runner { device: &device, queue: &queue, cache: &mut cache, base: base.clone(), seed, progress: Default::default(), log: &mut note, batch_file: file.clone(), max_triangles: None, numbering: Default::default() };
            runner.run(&graph)
        };
        let label = file.as_ref().map_or(String::new(), |f| format!("[{}/{}] {}: ", i + 1, batch.len(), f.display()));
        for l in lines.borrow().iter() {
            log.line(&format!("{label}{l}"));
        }
        match result {
            Ok(files) => written += files.len(),
            Err(e) => {
                failed += 1;
                log.line(&format!("{label}FAILED: {e}"));
            }
        }
    }
    log.line(&format!("done: {written} file(s) written, {failed} run(s) failed, in {:.1} s", started.elapsed().as_secs_f32()));
    Ok(failed)
}

fn run(args: &[String], log: &mut Log) -> Result<usize, String> {
    if args.iter().any(|a| a == "--mesh-graph") {
        return run_mesh_graph(args, log);
    }
    let o = parse(args)?;
    let text = std::fs::read_to_string(&o.graph).map_err(|e| format!("{}: {e}", o.graph.display()))?;
    let mut sets = o.sets.clone();
    sets.extend(resolve_params(&text, &o.params)?);
    let graph = Graph::from_json(&apply_sets(&text, &sets)?)?;
    if let Some(path) = &o.export_shader {
        let lang = path.extension().and_then(|e| e.to_str()).and_then(crate::shaderexport::Lang::from_extension).ok_or("--export-shader needs a .hlsl, .glsl or .wgsl file name")?;
        std::fs::write(path, crate::shaderexport::export(&graph, o.seed, lang)?).map_err(|e| format!("{}: {e}", path.display()))?;
        log.line(&format!("wrote {}", path.display()));
        return Ok(0);
    }
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
    // Plain files first, then the manifest's meshes, each with the overrides it names.
    let specs: Vec<Spec> = o.files.iter().map(|f| Spec { file: f.clone(), ..Default::default() }).chain(o.specs.iter().cloned()).collect();
    let total = specs.len();
    let mut failed = 0;
    for (i, spec) in specs.iter().enumerate() {
        let input = &spec.file;
        let d = Displace {
            amount: spec.amount.unwrap_or(d.amount),
            tile_size: spec.tile.unwrap_or(d.tile_size),
            prim_size: spec.size.unwrap_or(d.prim_size),
            mask: match &spec.mask {
                Some(names) => mask_from_names(names)?.pack(),
                None => d.mask,
            },
        };
        let mut settings = o.settings.clone();
        settings.res = spec.res.unwrap_or(settings.res);
        settings.cull = spec.cull.unwrap_or(settings.cull);
        // A mesh with its own parameters gets its own copy of the graph.
        let own_graph;
        let graph = if spec.params.is_empty() {
            &graph
        } else {
            let mut sets = o.sets.clone();
            sets.extend(resolve_params(&text, &spec.params.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<Vec<_>>())?);
            own_graph = Graph::from_json(&apply_sets(&text, &sets)?)?;
            &own_graph
        };
        let stem = input.file_stem().map_or("mesh".into(), |s| s.to_string_lossy().into_owned());
        let seed = spec.seed.unwrap_or_else(|| o.mode.seed_for(o.seed, i));
        let label = if o.mode == SeedMode::Same && spec.seed.is_none() { stem.clone() } else { format!("{stem}_seed{seed}") };
        let output = crate::meshapp::unique_path(&o.out, &format!("{label}_displaced"), settings.format.extension());
        let d = Displace { amount: d.amount * crate::meshexport::amount_factor(o.seed, i, o.amount_var), ..d };
        let job = FileJob { device: &device, queue: &queue, graph, seed, d, settings, grid_res: spec.grid_res.unwrap_or(o.grid_res), grid_pad: 0.5, combine: o.combine, object_scale: o.object_units.then_some(1.0) };
        let started = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(input, &output, &Progress::default())));
        let n = i + 1;
        match result {
            Ok(Ok(info)) => {
                log.line(&format!("[{n}/{total}] {} -> {} ({info}, {:.1} s)", input.display(), output.display(), started.elapsed().as_secs_f32()));
            }
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

    #[test]
    fn exposed_parameters_resolve_to_node_settings() {
        let mut g = Graph::default_graph();
        g.exposed.push(crate::graph::Exposed { name: "roughness".into(), node: 1, field: "scale".into() });
        let text = g.to_json();
        let sets = resolve_params(&text, &[("roughness".to_string(), serde_json::json!(6.0))]).unwrap();
        let patched = Graph::from_json(&apply_sets(&text, &sets).unwrap()).unwrap();
        assert!(matches!(&patched.nodes[&1].kind, crate::graph::NodeKind::Noise(p) if p.scale == 6.0));
        assert_eq!(patched.exposed.len(), 1, "the exposed list survives");
        let err = resolve_params(&text, &[("nope".to_string(), serde_json::json!(1))]).unwrap_err();
        assert!(err.contains("roughness"), "the error lists what is exposed: {err}");
        assert_eq!(g.exposed_value(&g.exposed[0]).unwrap(), serde_json::json!(4.0));
        assert!(g.exposed_value(&crate::graph::Exposed { name: "x".into(), node: 1, field: "bogus".into() }).is_err());
    }

    #[test]
    fn a_manifest_gives_each_mesh_its_own_settings() {
        let dir = std::env::temp_dir().join("noiser-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.json");
        std::fs::write(&path, r#"{"meshes": [{"file": "a.obj", "amount": 0.4, "seed": 9, "mask": ["thickness"], "params": {"size": 6}}, {"file": "sub/b.obj"}]}"#).unwrap();
        let specs = read_manifest(&path).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!((specs[0].amount, specs[0].seed), (Some(0.4), Some(9)));
        assert!(specs[0].file.starts_with(&dir) && specs[1].file.ends_with("sub/b.obj") || specs[1].file.ends_with("sub\\b.obj"));
        assert_eq!(mask_from_names(specs[0].mask.as_ref().unwrap()).unwrap().terms.len(), 1);
        std::fs::write(&path, r#"[{"file": "a.obj", "bogus": 1}]"#).unwrap();
        assert!(read_manifest(&path).is_err(), "unknown keys are rejected");
        let o = parse(&args("--headless g.noisegraph --batch-out out --manifest ".to_string().as_str()).into_iter().chain([path.to_string_lossy().into_owned()]).collect::<Vec<_>>());
        assert!(o.is_err());
    }
}
