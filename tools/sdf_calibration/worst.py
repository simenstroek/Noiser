import sys, os; sys.path.insert(0,'.')
import numpy as np
from sdfcal import *
import sdftree
from runlayers import layer_ply

def flat(nodes, tf=None, path=''):
    """Yield (path, node, world->local function) for every shape, composing group frames."""
    for i,n in enumerate(nodes):
        here=f'{path}/{i}:{n["type"]}'
        if n['type']=='shape': yield here,n,tf
        elif n['type']=='group':
            t=np.array(n['position']); M=rot_xyz(np.array(n['rotation'])); s=n['scale'][0]
            sub=(lambda p,tf=tf,t=t,M=M,s=s: ((tf(p) if tf else p)-t)@M/s)
            yield from flat(n['children'],sub,here)

def worst(name,li,n=30000,top=8):
    d=load_sdf(name); l=d['layers'][li]; a=d['assemblies'][l['assembly']]
    v,_=load_ply(layer_ply(name,li),n)
    f=np.stack([v[:,0], v[:,2], -v[:,1]],1)
    f=((f-np.array(l['position']))@rot_xyz(np.array(l['rotation'])))/np.array(l['scale'])
    dd=sdftree.eval_nodes(f,a['shapes'],{}); ad=np.abs(dd)
    bad=np.where(ad>max(0.01,np.percentile(ad,90)))[0]
    print(f'{name} layer {li+1} {l["name"]}: {len(bad)} vertices above threshold of {len(f)}')
    # which shapes are closest (by their own solid-ish distance) to the bad vertices
    shapes=list(flat(a['shapes']))
    score={}
    for path,node,tf in shapes:
        if node['brush'] not in sdftree.PRIMS: continue
        p=f[bad] if tf is None else tf(f[bad])
        if node.get('mirror_x'): p=p.copy(); p[:,0]=np.abs(p[:,0])
        q=(p-np.array(node['position']))@rot_xyz(np.array(node['rotation']))
        dist=np.abs(sdftree.PRIMS[node['brush']](q,node))
        score[path]=(np.median(dist),node)
    for path,(m,node) in sorted(score.items(),key=lambda kv:kv[1][0])[:top]:
        keys=('brush','mode','scale','round','bevel','thickness','cone','mirror_x','repeat_mode','radial_repeats','linear_repeats','blend','blend_strength','chamfer')
        print(f'  {path:34s} median dist to bad {m:.4f}  ',{k:(np.round(node[k],3).tolist() if isinstance(node[k],(list,float)) else node[k]) for k in keys if k in node and not (k in("radial_repeats",) and node[k]==1)})
if __name__=='__main__':
    worst(sys.argv[1],int(sys.argv[2]))
