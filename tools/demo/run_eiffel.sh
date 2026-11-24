#!/bin/bash
# Regenerates the Eiffel / sphinx demo, exports it, imports it into the Unreal project and photographs it.
#   bash tools/demo/run_eiffel.sh [triangles per chunk] [map size] [shots.json]
# Needs: the Noiser library at LIB, the Unreal project at PROJ, UnrealEditor.exe at UE. Shots go to target/ue/shots.
set -e
cd "$(dirname "$0")/../.."
TRIS=${1:-250000}
MAP=${2:-1024}
SHOTS=${3:-target/ue/shots_eiffel.json}
LIB="H:/My Drive/NoiserLibrary"
PROJ="E:/Unreal Projects/FAB_StagingZone/FAB_StagingZone.uproject"
UE="/e/Unreal Engine/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"
OUT="E:/Unreal Projects/FAB_StagingZone/NoiserExport_Eiffel"
mkdir -p "$OUT" target/ue
cargo build --release 2>&1 | grep -E "^error|Finished"
python tools/demo/make_shapes.py "$LIB/SDFs" > /dev/null
python tools/demo/make_landmarks.py "$LIB/SDFs" > /dev/null
rm -rf tools/demo/__pycache__
rm -f "$OUT"/*.obj "$OUT"/*.png "$OUT"/*.json "$OUT"/*.mtl 2>/dev/null || true
NOISER_DEMO_LIB="$LIB" NOISER_DEMO_OUT="$OUT" NOISER_DEMO_TRIS=$TRIS NOISER_DEMO_MAP=$MAP cargo test --release write_eiffel_scene -- --ignored 2>&1 | grep -E "^test "
rm -f "$LIB/Mesh Graphs/EiffelTerrain.log"
start=$(date +%s)
./target/release/noiser.exe --headless --mesh-graph "$LIB/Mesh Graphs/EiffelTerrain.meshgraph" > /dev/null 2>&1
echo "export took $(( $(date +%s) - start )) s"
grep -E "FAILED|done" "$LIB/Mesh Graphs/EiffelTerrain.log" | cut -c1-160
export MSYS_NO_PATHCONV=1 NOISER_UE_QUIT=1 NOISER_MANIFEST="$OUT/terrain_chunks.json" NOISER_DESTINATION=/Game/NoiserEiffel NOISER_LEVEL=/Game/NoiserEiffel/EiffelParadise
"$UE" "$PROJ" -ExecutePythonScript="E:/_Noiser/tools/unreal_import_terrain.py" -unattended -nosplash -stdout -FullStdOutLogOutput > target/ue/import_stdout.log 2>&1 || true
grep -E "Noiser terrain|LogPython: Error" target/ue/import_stdout.log | head -5 | cut -c1-250
rm -rf target/ue/shots; mkdir -p target/ue/shots
unset NOISER_UE_QUIT
"$UE" "$PROJ" -ExecutePythonScript="E:/_Noiser/tools/unreal_shots.py /Game/NoiserEiffel/EiffelParadise E:/_Noiser/$SHOTS E:/_Noiser/target/ue/shots" -unattended -nosplash -stdout -FullStdOutLogOutput > target/ue/shots_stdout.log 2>&1 || true
ls target/ue/shots
echo chain finished
