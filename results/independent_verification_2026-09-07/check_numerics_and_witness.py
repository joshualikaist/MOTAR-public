#!/usr/bin/env python3
"""Optional independent SciPy cross-check and archived Torch CPU counterexample.
Does not run Isaac Gym, train a policy, or modify the archived implementation.
"""
import json,importlib.util,sys
from pathlib import Path
from scipy.stats import chi2,norm,t
import torch
P=Path(__file__).resolve().parent;BASE=P.parents[1]
j=json.loads((P/'recomputed.json').read_text())
for c in j['contrasts'].values():
    for r in [c['pooled']]+c['per_density']+c['per_seed']:
        assert abs(r['p']-2*norm.sf(abs(r['delta']/r['se'])))<1e-12
        assert abs(r['Q_p']-chi2.sf(r['Q'],r['df']))<1e-12
    r=c['seed_t']
    assert abs(r['p']-2*t.sf(abs(r['mean']/r['se']),2))<1e-12
p=BASE/'results/navrl_grid_r1_seedrep_ep25000_s527/source_bundle/source_snapshot/aerial_gym/task/navrl_task/speed_governor.py'
spec=importlib.util.spec_from_file_location('audit_archived_governor',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
scan=torch.full((1,4,1),12.);scan[0,2,0]=.5
args=(scan,torch.tensor([0.]),torch.tensor([[2.,0.]]))
straight=m.directional_lidar_clearance(*args,max_range_m=12.,path_half_width_m=.45)
arc=m.arc_clearance(*args,torch.tensor([1.]),max_range_m=12.,path_half_width_m=.45)
assert straight.item()==.5 and arc.item()==12.
print('PASS: normal/chi-square/t cross-checks agree within 1e-12.')
print('Archived arc sensor-input counterexample (CPU): straight=0.5 m, arc=12.0 m.')
print('This is a function-level input, not a replay of a measured episode.')
