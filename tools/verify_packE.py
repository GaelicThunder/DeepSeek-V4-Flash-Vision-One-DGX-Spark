#!/usr/bin/env python3
"""Byte-verify every expert tensor of pack E (216 slots) against Vis expert kept[j], all layers."""
import json, struct, hashlib, os, sys, time
V=os.path.expanduser('~/models/deepseek-v4-flash-vision-spark/tp1'); E=os.path.expanduser('~/models/dsvision3e-splice')
PLAN=os.path.expanduser('~/models/deepseek-v4-flash-vision-k216-spark/tp1/REAP_K216_PLAN.json')
keep=json.load(open(PLAN))['keep_maps']['keep_by_layer']
class ST:
    def __init__(s,p):
        s.f=open(p,'rb'); n=struct.unpack('<Q',s.f.read(8))[0]; s.h=json.loads(s.f.read(n)); s.h.pop('__metadata__',None); s.base=8+n
    def raw(s,k):
        a,b=s.h[k]['data_offsets']; s.f.seek(s.base+a); return s.f.read(b-a)
bad=0; t0=time.time()
for L in range(43):
    fe=f'{E}/exl3-layer-{L:03d}-tp1-rank0.safetensors'
    if not os.path.exists(fe): print(f'L{L}: MISSING in E'); bad+=1; continue
    se=ST(fe); sv=ST(f'{V}/exl3-layer-{L:03d}-tp1-rank0.safetensors'); kept=keep[str(L)]
    nE=sum(1 for k in se.h if k.endswith('.w1.rank0.trellis')); extra=[k for k in se.h if '.experts.' not in k]
    mism=0; kinds=set()
    for j,e in enumerate(kept):
        for w in ('w1','w2','w3'):
            for t in ('trellis','suh','svh','mcg'):
                ke=f'layers.{L}.ffn.experts.{j}.{w}.rank0.{t}'; kv=f'layers.{L}.ffn.experts.{e}.{w}.rank0.{t}'
                if ke not in se.h or kv not in sv.h: mism+=1; kinds.add(t+'-missing'); continue
                if se.h[ke]['shape']!=sv.h[kv]['shape'] or se.h[ke]['dtype']!=sv.h[kv]['dtype']: mism+=1; kinds.add(t+'-shape'); continue
                if se.raw(ke)!=sv.raw(kv): mism+=1; kinds.add(t)
    K=se.h[f'layers.{L}.ffn.experts.0.w1.rank0.trellis']['shape'][2]//16
    print(f'L{L:2d}: E slots {nE} K{K} extra-tensors {len(extra)} mismatches {mism} {sorted(kinds) if kinds else ""}', flush=True)
    bad+=mism
print(f'TOTAL mismatches {bad}  ({time.time()-t0:.0f}s)')
