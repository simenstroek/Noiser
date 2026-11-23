import sys, os; sys.path.insert(0,'.')
import numpy as np
from sdfcal import *
import sdftree
from runlayers import layer_ply

def load_gallery(n=400000):
    d=load_sdf('shapes')
    v,_=load_ply(layer_ply('shapes',0),n)
    f=np.stack([v[:,0], v[:,2], -v[:,1]],1)
    return d['assemblies'][0]['shapes'], f

def cell_mask(f,node,pad=0.15):
    pos=np.array(node['position']); half=np.array(node['scale'])/2
    ext=np.maximum(half,half[[0,2,1]]) .max()
    m=(np.abs(f[:,0]-pos[0])<half[0]+pad)&(np.abs(f[:,2]-pos[2])<half[2]+pad)&(np.abs(f[:,1]-pos[1])<max(half[1],0.2)+pad)
    return m

def evaluate(node,pts,prim=None):
    q=(pts-np.array(node['position']))@rot_xyz(np.array(node['rotation']))
    fn=prim or sdftree.PRIMS[node['brush']]
    return fn(q,node)

if __name__=='__main__':
    shapes,f=load_gallery()
    for i,s in enumerate(shapes):
        m=cell_mask(f,s); pts=f[m]
        if len(pts)<50: print(i,s['brush'],'no vertices'); continue
        if s['brush'] not in sdftree.PRIMS: print(f"{i:2d} {s['brush']:8s} (not implemented) n={len(pts)}"); continue
        d=np.abs(evaluate(s,pts))
        print(f"{i:2d} {s['brush']:8s} t {s['thickness']:.3f} bevel {s['bevel']:.2f} round {s['round']:.2f} cone {s['cone']:.3f}  n={len(pts):6d} median {np.median(d):.5f} p95 {np.percentile(d,95):.5f} max {d.max():.4f}")
