#!/usr/bin/env python3
"""Read-only source, receipt timing and condition audit; standard library only."""
import csv,json,hashlib,statistics,datetime,difflib,math
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((BASE/'results/verification_export_2026-09-07/governor_cells.csv').open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
roots=['D1p_ep25000_s523','L1_width_s523','RB_seedrep_s527','RB_seedrep_s531','RB_crosstree_s523']
evidence={};old=None
for root in roots:
 r=next(r for r in rows if r['root']==root);raw=json.loads((BASE/r['result_path']).read_text());p=Path(raw['runtime_source_manifest']);m=json.loads(p.read_text());files={f['path']:f['sha256'] for f in m['runtime_files']}
 issues=[]
 for f in m['runtime_files']:
  q=p.parent/f['snapshot']
  if not q.exists():issues.append([f['path'],'missing_snapshot'])
  elif sha(q)!=f['sha256']:issues.append([f['path'],'snapshot_hash'])
 env=Path(raw['python_environment_manifest'])
 evidence[root]=dict(file_count=len(files),snapshot_issues=issues,manifest_hash_matches=sha(p)==raw['runtime_source_manifest_sha256'],env_hash_matches=sha(env)==raw['python_environment_manifest_sha256'],checkpoint=raw['checkpoint_sha256'],robot=raw['condition']['robot_name'],goal_min=raw['condition']['goal_dist_min_m'],goal_max=raw['condition']['goal_dist_max_m'])
 if old is None:old=files;oldenv=env.read_text();oldmanifest=p
 else:
  evidence[root]['changed_files']=[f for f in sorted(set(old)|set(files)) if old.get(f)!=files.get(f)]
  evidence[root]['environment_diff']=''.join(difflib.unified_diff(oldenv.splitlines(True),env.read_text().splitlines(True)))
 if root=='RB_seedrep_s527':
  for f in evidence[root]['changed_files']:
   a=oldmanifest.parent/'source_snapshot'/f;b=p.parent/'source_snapshot'/f
   (OUT/('source_diff_'+Path(f).name+'.patch')).write_text(''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile='old/'+f,tofile='new/'+f)))
timing=[]
for r in rows:
 if r['root'] not in ['RB_seedrep_s527','RB_seedrep_s531']:continue
 raw=json.loads((BASE/r['result_path']).read_text());receipt=json.loads(Path(raw['evaluation_receipt']).read_text())
 a=datetime.datetime.fromisoformat(receipt['started_at_utc']);b=datetime.datetime.fromisoformat(receipt['completed_at_utc'])
 timing.append(dict(root=r['root'],cell=r['cell'],mode=r['cond_speed_governor_mode'],width=r['cond_speed_governor_half_width_m'],minutes=(b-a).total_seconds()/60,start=a.isoformat(),end=b.isoformat()))
grouped=[]
for mode,w in [('riskcap','0.45'),('stopcap','0.45'),('dwa_arc','0.45'),('dwa_arc','1.2'),('stopcap','1.2')]:
 vals=[r['minutes'] for r in timing if r['mode']==mode and r['width']==w];grouped.append(dict(mode=mode,width=w,n=len(vals),mean=statistics.mean(vals),median=statistics.median(vals),min=min(vals),max=max(vals)))
# Boundary witness for the published arc formula. This is a mathematical counterexample,
# not an estimate of how often that state occurred in any measured episode.
R=2.;px=.5;py=0.;w=.45
perp=abs(math.hypot(px,py-R)-R);along=R*2*abs(math.atan2(py,px))
x,y=R*math.sin(.1/R),R*(1-math.cos(.1/R))
witness=dict(speed_mps=2.,yaw_rate_radps=1.,radius_m=R,return_xy=[px,py],tube_half_width_m=w,perp_m=perp,implemented_along_m=along,implemented_on_arc=(along>0 and perp<=w),implemented_clearance_m=12.,centerline_at_s_0p1=[x,y],distance_to_return_at_s_0p1=math.hypot(px-x,py-y),initial_distance_to_return=math.hypot(px,py))
result=dict(sources=evidence,timing=timing,timing_by_arm=grouped,arc_boundary_witness=witness)
(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('Source snapshots:',[(k,v['file_count'],len(v['snapshot_issues'])) for k,v in evidence.items()]);print('Timing',grouped);print('Arc witness',witness)
