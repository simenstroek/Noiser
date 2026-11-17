# Usage: cargo test write_check_vdb -- --ignored
#        blender -b --factory-startup --python tools/check_vdb.py -- target/check.vdb
# Reads target/check.vdb with Blender's OpenVDB module and compares every voxel to the formula
# the export used: density = mean of the texel centre, ((i, j, k) + 0.5) / 20.
import sys, openvdb
n = 20
path = sys.argv[sys.argv.index("--") + 1]
grids, meta = openvdb.readAll(path)
g = grids[0]
print("CHECK name", g.name, "type", type(g).__name__, "active", g.activeVoxelCount(), "bbox", g.evalActiveVoxelBoundingBox(), "voxel", g.transform.voxelSize())
acc = g.getConstAccessor()
worst = 0.0
for i in range(n):
    for j in range(n):
        for k in range(n):
            want = ((i + 0.5) + (j + 0.5) + (k + 0.5)) / (3 * n)
            worst = max(worst, abs(acc.getValue((i, j, k)) - want))
print("CHECK worst error", worst, "outside value", acc.getValue((n, 0, 0)), "active outside", acc.isValueOn((n, 0, 0)))

