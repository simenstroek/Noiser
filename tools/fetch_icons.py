"""Fetches Material icons from Iconify into assets/ui_icons (drawn white, tinted by the app) and regenerates the
lookup table in src/icons.rs. Usage: python tools/fetch_icons.py name=prefix/icon-name ...
Example: python tools/fetch_icons.py noise=material-symbols/grain-rounded"""
import os
import re
import sys
import urllib.request

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
folder = os.path.join(root, "assets", "ui_icons")

for arg in sys.argv[1:]:
    name, source = arg.split("=", 1)
    request = urllib.request.Request(f"https://api.iconify.design/{source}.svg", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request) as r:
        svg = r.read().decode("utf-8")
    if "<path" not in svg:
        sys.exit(f"{source}: not an icon")
    with open(os.path.join(folder, f"{name}.svg"), "w", encoding="utf-8") as f:
        f.write(svg.replace("currentColor", "#ffffff"))
    print("fetched", name)

names = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".svg") and re.fullmatch(r"[a-z0-9_]+\.svg", f))
path = os.path.join(root, "src", "icons.rs")
text = open(path, encoding="utf-8").read()
start = text.index('        "')
end = text.index("        _ => return None")
table = "".join(f'        "{n}" => include_bytes!("../assets/ui_icons/{n}.svg"),\n' for n in names)
open(path, "w", encoding="utf-8").write(text[:start] + table + text[end:])
print(len(names), "icons in the table")
