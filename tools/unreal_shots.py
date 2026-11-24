"""Takes screenshots of the editor's level viewport from a list of cameras and quits (for checking a generated terrain).

    UnrealEditor.exe Project.uproject -ExecutePythonScript="unreal_shots.py <level> <shots.json> <output folder>" -unattended

`shots.json`: {"size": [1920, 1080], "shots": [{"name": "aerial", "from": [x, y, z], "to": [x, y, z]}, ...]} in centimetres.
Pictures are saved to <output folder>/<name>.png.
"""

import json
import math
import sys
import time

import unreal

SECONDS_TO_SETTLE = 8.0  # Nanite streaming, shadow maps and the exposure need a moment after the camera moves
SECONDS_TO_SAVE = 4.0


def look_at(src, dst):
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    return unreal.Rotator(0.0, math.degrees(math.atan2(dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx)))


def run(level, shots_path, out_dir):
    with open(shots_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    width, height = spec.get("size", [1920, 1080])
    shots = spec["shots"]
    if level:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(level)
    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    # The editor closes when a start-up script returns, unless the script says it is still working.
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    state = {"shot": 0, "start": None, "taken": False}
    handle = {}
    log = open(out_dir.rstrip("/") + "/shots.log", "w")

    def note(text):
        log.write(text + chr(10))
        log.flush()

    def tick(delta):
        i = state["shot"]
        if i >= len(shots):
            unreal.unregister_slate_post_tick_callback(handle["h"])
            note("done")
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
            unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
            return
        shot = shots[i]
        if state["start"] is None:
            state["start"] = time.time()
            editor.set_level_viewport_camera_info(unreal.Vector(*shot["from"]), look_at(shot["from"], shot["to"]))
            note("camera for " + shot["name"])
        elapsed = time.time() - state["start"]
        if elapsed > SECONDS_TO_SETTLE and not state["taken"]:
            path = out_dir.rstrip("/") + "/" + shot["name"] + ".png"
            unreal.SystemLibrary.execute_console_command(editor.get_editor_world(), "HighResShot {}x{} filename=\"{}\"".format(width, height, path))
            note("shot " + path)
            state["taken"] = True
        if elapsed > SECONDS_TO_SETTLE + SECONDS_TO_SAVE:
            state.update(shot=i + 1, start=None, taken=False)

    handle["h"] = unreal.register_slate_post_tick_callback(tick)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        unreal.log_error("usage: py unreal_shots.py <level> <shots.json> <output folder>")
    else:
        run(args[0], args[1], args[2])
