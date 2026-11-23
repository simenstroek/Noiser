import sys, os; sys.path.insert(0,'.')
import numpy as np
from sdfcal import *
from sdftree import *

def layer_ply(name, i):
    return os.path.join(SDFS, f"{name}.ply" if i == 0 else f"{name} - {i}.ply")

def run(name, only=None, n=20000, layer_mode='A'):
    d=load_sdf(name)
    for i,l in enumerate(d['layers']):
        if only is not None and i not in only: continue
        a=d['assemblies'][l['assembly']]
        v,_=load_ply(layer_ply(name,i),n)
        f=np.stack([v[:,0], v[:,2], -v[:,1]],1)
        if layer_mode=='B':
            f=((f-np.array(l['position']))@rot_xyz(np.array(l['rotation'])))/np.array(l['scale'])
        try:
            dd=eval_nodes(f,a['shapes'],{})
        except Exception as e:
            print(f'{name} layer {i+1} {l["name"]!r}: ERROR {e!r}'); continue
        if dd is None: print(f'{name} layer {i+1} {l["name"]!r}: empty'); continue
        ad=np.abs(dd)
        print(f'{name:10s} layer {i+1:2d} {l["name"]!r:14s} res {a["resolution"]} median {np.median(ad):.5f} p95 {np.percentile(ad,95):.5f} max {ad.max():.4f}')

if __name__=='__main__':
    run(sys.argv[1], None if len(sys.argv)<3 else [int(x) for x in sys.argv[2].split(',')], layer_mode=os.environ.get('LM','B'))
