#!/usr/bin/env python3
"""Independent count-level audit. Standard library only; no project analysis imports.
Inputs are read only. Recomputes contrasts and audits linked evidence when present.
"""
import csv, json, math, statistics, hashlib, collections
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
INPUT=BASE/'results/verification_export_2026-09-07'
Z=1.959963984540054
rows=list(csv.DictReader((INPUT/'governor_cells.csv').open()))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gamma_sf(a,x):
    if x<=0:return 1.0
    gl=math.lgamma(a)
    if x<a+1:
        term=total=1/a
        for n in range(1,10000):
            term*=x/(a+n);total+=term
            if abs(term)<abs(total)*1e-14:break
        return max(0,1-total*math.exp(-x+a*math.log(x)-gl))
    b=x+1-a;c=1e300;d=1/b;h=d
    for i in range(1,10000):
        an=-i*(i-a);b+=2;d=an*d+b
        if abs(d)<1e-300:d=1e-300
        c=b+an/c
        if abs(c)<1e-300:c=1e-300
        d=1/d;delta=d*c;h*=delta
        if abs(delta-1)<1e-14:break
    return h*math.exp(-x+a*math.log(x)-gl)
def stat(d,v):
    se=math.sqrt(v);z=d/se
    return dict(delta=d,se=se,lo=d-Z*se,hi=d+Z*se,p=math.erfc(abs(z)/math.sqrt(2)),variance=v)
def diff(a,b,metric='crash'):
    na,nb=int(a['actual_episodes']),int(b['actual_episodes'])
    pa,pb=int(a['n_'+metric])/na,int(b['n_'+metric])/nb
    return stat(100*(pa-pb),10000*(pa*(1-pa)/na+pb*(1-pb)/nb))
def pool(ds):
    w=[1/d['variance'] for d in ds];sw=sum(w)
    m=sum(x*d['delta'] for x,d in zip(w,ds))/sw
    r=stat(m,1/sw);q=sum(x*(d['delta']-m)**2 for x,d in zip(w,ds));df=len(ds)-1
    r.update(k=len(ds),Q=q,df=df,I2=max(0,(q-df)/q) if q else 0,Q_p=gamma_sf(df/2,q/2) if df else None)
    return r
def holm(ds):
    order=sorted(range(len(ds)),key=lambda i:ds[i]['p']);last=0
    for rank,i in enumerate(order):
        last=max(last,min(1,(len(ds)-rank)*ds[i]['p']));ds[i]['p_holm']=last

def equal_seed(ds):
    vals=[d['delta'] for d in ds];mean=statistics.mean(vals);se=statistics.stdev(vals)/math.sqrt(3)
    # Student t with df=2 has an elementary CDF; three observed seeds only.
    t=mean/se;p=1-abs(t)/math.sqrt(t*t+2)
    return dict(mean=mean,se=se,lo=mean-4.30265272975*se,hi=mean+4.30265272975*se,p=p,df=2)
def quantile(v,p):
    v=sorted(v);i=(len(v)-1)*p;j=int(i)
    return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(i-j)
def desc(v):return dict(n=len(v),q25=quantile(v,.25),median=quantile(v,.5),q75=quantile(v,.75),min=min(v),max=max(v)) if v else None

integrity=dict(rows=len(rows),columns=len(rows[0]),roots=len(set(r['root'] for r in rows)),episodes=sum(int(r['actual_episodes']) for r in rows),n_range=[min(int(r['actual_episodes']) for r in rows),max(int(r['actual_episodes']) for r in rows)],issues=[],root_counts=dict(collections.Counter(r['root'] for r in rows)))
raw={};contact={}
for r in rows:
    n=int(r['actual_episodes']);key=r['root']+'/'+r['cell']
    if sum(int(r['n_'+m]) for m in ['crash','captured','timeout'])!=n:integrity['issues'].append([key,'sum'])
    for m in ['crash','captured','timeout']:
        if abs(float(r['rate_'+m])-int(r['n_'+m])/n)>1e-12:integrity['issues'].append([key,'rate_'+m])
    p=BASE/r['result_path']
    if not p.exists():integrity['issues'].append([key,'missing_result']);continue
    if sha(p)!=r['result_sha256']:integrity['issues'].append([key,'result_hash'])
    d=raw[key]=json.loads(p.read_text())
    for m in ['crash','captured','timeout']:
        if d['outcome'][m]!=int(r['n_'+m]):integrity['issues'].append([key,'raw_count_'+m])
    for k in [k for k in r if k.startswith('cond_') and r[k]!='']:
        v=d['condition'].get(k[5:]);target=r[k]
        if str(v)!=target:
            try:equal=float(v)==float(target)
            except (ValueError,TypeError):equal=False
            if not equal:integrity['issues'].append([key,'condition_'+k])
    cg=d.get('contact_geometry') or {}
    if r['contact_records_path']:
        p=Path(r['contact_records_path'])
        if not p.exists():integrity['issues'].append([key,'missing_contacts']);continue
        cr=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];contact[key]=cr
        if len(cr)!=int(r['contact_records_rows']):integrity['issues'].append([key,'contact_rows'])
        if cg.get('contact_records_sha256') and sha(p)!=cg['contact_records_sha256']:integrity['issues'].append([key,'contact_hash'])
manifest=json.loads((INPUT/'roots_manifest.json').read_text())
integrity['manifest_entries']=len(manifest)
for m in manifest:
    p=BASE/m['root']/'cells.json'
    if not p.exists() or sha(p)!=m['cells_json_sha256']:integrity['issues'].append([m['root'],'manifest_hash'])
integrity['roots_without_manifest']=sorted(set(r['root'] for r in rows)-{r['root'] for r in rows if str(Path(r['result_path']).parents[1]) in {m['root'] for m in manifest}})
integrity['contact_files']=len(contact);integrity['contact_rows']=sum(map(len,contact.values()))

# Explicit inclusion: narrow arms from D1-prime and each R-B root; widths from L1 for seed 523.
rootmap={523:'D1p_ep25000_s523',527:'RB_seedrep_s527',531:'RB_seedrep_s531'}
seeds=list(rootmap);dens=[70,100,130,160,205];arms=['dwa_arc','riskcap','stopcap']
cells={}
for s,root in rootmap.items():
    for r in rows:
        if r['root']==root and r['cond_speed_governor_mode'] in arms and float(r['cond_speed_governor_half_width_m'])==.45:
            k=(s,int(r['cond_bars']),r['cond_speed_governor_mode']);assert k not in cells;cells[k]=r
assert len(cells)==45
with (OUT/'primary_cells.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(cells.values())
contrasts={}
for a,b in [('dwa_arc','riskcap'),('dwa_arc','stopcap'),('stopcap','riskcap')]:
    name=a+'-'+b;dd={(s,d):diff(cells[s,d,a],cells[s,d,b]) for s in seeds for d in dens}
    perden=[dict(density=d,**pool([dd[s,d] for s in seeds])) for d in dens];holm(perden)
    perseed=[dict(seed=s,**pool([dd[s,d] for d in dens])) for s in seeds]
    equal=stat(statistics.mean(v['delta'] for v in dd.values()),sum(v['variance'] for v in dd.values())/225)
    qbetween=pool(list(dd.values()))['Q']-sum(v['Q'] for v in perden)
    qseed=pool(list(dd.values()))['Q']-sum(v['Q'] for v in perseed)
    contrasts[name]=dict(pooled=pool(list(dd.values())),per_density=perden,per_seed=perseed,cell=[dict(seed=s,density=d,**v) for (s,d),v in dd.items()],equal_cell=equal,seed_t=equal_seed(perseed),new_seeds_only=pool([dd[s,d] for s in [527,531] for d in dens]),leave_one_seed_out={s:pool([v for (ss,d),v in dd.items() if ss!=s]) for s in seeds},density_heterogeneity=dict(Q=qbetween,df=4,p=gamma_sf(2,qbetween/2)),seed_heterogeneity=dict(Q=qseed,df=2,p=gamma_sf(1,qseed/2)))
    contrasts[name]['design_effect_to_lose_95CI']=(contrasts[name]['pooled']['delta']/(Z*contrasts[name]['pooled']['se']))**2
    contrasts[name]['density_weight_share']={d:sum(1/v['variance'] for (s,dden),v in dd.items() if dden==d)/sum(1/v['variance'] for v in dd.values()) for d in dens}
# Explicit interaction for P5: low vs each high, and low vs pooled high.
stop=contrasts['stopcap-riskcap'];low=stop['per_density'][0]
stop['low_vs_high']=[dict(density=r['density'],**stat(low['delta']-r['delta'],low['variance']+r['variance'])) for r in stop['per_density'][1:]]
holm(stop['low_vs_high'])
high=pool([r for r in stop['cell'] if r['density']!=70]);stop['low_vs_high_pooled']=stat(low['delta']-high['delta'],low['variance']+high['variance'])
stop['70_seed_t']=equal_seed([r for r in stop['cell'] if r['density']==70])
stop['70_design_effect']=(low['delta']/(Z*low['se']))**2
all15=[dict(r) for r in stop['cell']];holm(all15);stop['15_cell_holm']=all15
# Equal-density seed sensitivity uses the same five-density target in every seed.
for c in contrasts.values():
    c['equal_density_seed_t']=equal_seed([dict(delta=statistics.mean(r['delta'] for r in c['cell'] if r['seed']==seed)) for seed in seeds])
# A7 factorial controls separate law from geometry on the same checkpoint within each root.
a7=[]
for root in ['A7_ref5in_s509','A7_ref5in_s491','A7_ep25000_205_s49']:
    r={r['cond_speed_governor_mode']:r for r in rows if r['root']==root}
    g_stop=diff(r['dwa_arc'],r['stopcap']);g_risk=diff(r['riskcap_arc'],r['riskcap'])
    a7.append(dict(root=root,arc_minus_straight_stoplaw=g_stop,arc_minus_straight_risklaw=g_risk,interaction=stat(g_stop['delta']-g_risk['delta'],g_stop['variance']+g_risk['variance'])))
# Raw total rates by arm and outcome; aggregate descriptive totals are NOT IVW weights.
totals={}
for arm in arms:
    rr=[r for (s,d,a),r in cells.items() if a==arm];n=sum(int(r['actual_episodes']) for r in rr)
    totals[arm]=dict(n=n,**{m:dict(count=sum(int(r['n_'+m]) for r in rr),rate=100*sum(int(r['n_'+m]) for r in rr)/n) for m in ['crash','captured','timeout']})
# Widening uses paired contrasts inside each root, including L1 baseline for seed 523.
width=[]
for d in [70,205]:
    for arm in ['dwa_arc','stopcap']:
        byseed=[]
        for s in seeds:
            root='L1_width_s523' if s==523 else rootmap[s]
            matches=[r for r in rows if r['root']==root and int(r['cond_bars'])==d and r['cond_speed_governor_mode']==arm]
            narrow=next(r for r in matches if float(r['cond_speed_governor_half_width_m'])==.45)
            wide=next(r for r in matches if float(r['cond_speed_governor_half_width_m'])==1.2)
            byseed.append(dict(seed=s,crash=diff(wide,narrow),captured=diff(wide,narrow,'captured'),timeout=diff(wide,narrow,'timeout'),wide_capture=100*int(wide['n_captured'])/int(wide['actual_episodes']),counts={w:dict(n=int(r['actual_episodes']),**{m:int(r['n_'+m]) for m in ['crash','captured','timeout']}) for w,r in [('narrow',narrow),('wide',wide)]}))
        width.append(dict(density=d,arm=arm,seeds=byseed,**{m:pool([r[m] for r in byseed]) for m in ['crash','captured','timeout']}))
# D4 and C6 baseline interaction.
d4=[r for r in rows if r['root']=='D4_coadapt_s521'];d4out=[]
for pol in ['T0','T1']:
    r={r['cond_speed_governor_mode']:r for r in d4 if r['cell'].startswith(pol)}
    d4out.append(dict(policy=pol,**diff(r['stopcap'],r['riskcap'])))
d4interaction=stat(d4out[1]['delta']-d4out[0]['delta'],sum(r['variance'] for r in d4out))
c6=[]
for d in dens:
    rr={r['cond_speed_governor_mode']:r for r in rows if r['root']=='D1p_ep25000_s523' and int(r['cond_bars'])==d}
    c6.append(dict(density=d,**diff(rr['riskcap'],rr['off'])))
c6interaction=stat(c6[-1]['delta']-c6[0]['delta'],c6[-1]['variance']+c6[0]['variance'])
# Contact geometry: descriptive case-only summaries (do NOT assume independent episodes).
cs=[]
for r in rows:
    k=r['root']+'/'+r['cell'];cr=contact.get(k,[])
    if not cr:continue
    cs.append(dict(root=r['root'],cell=r['cell'],seed=int(r['cond_seed']),density=int(r['cond_bars']),mode=r['cond_speed_governor_mode'],width=float(r['cond_speed_governor_half_width_m']),n=len(cr),lateral=desc([v['hit_lateral_cmd'] for v in cr]),fixed_lateral_count=sum(v['hit_lateral_cmd']>.45 for v in cr),pinch_n=sum('pinch_t0' in v for v in cr),pinch=sum(bool(v.get('pinch_t0')) for v in cr),opposite_gap=desc([max(v['gap_left_t0'],v['gap_right_t0']) for v in cr if 'gap_left_t0' in v]),categories={cat:dict(n=len(vv),yaw=desc([abs(v['yaw_rate_t1']) for v in vv]),slip=desc([v['cmd_vs_actual_deg'] for v in vv])) for cat in sorted(set(v['category_cmd'] for v in cr)) if (vv:=[v for v in cr if v['category_cmd']==cat])}))
rbcontacts=[]
for mode,w in [('riskcap',.45),('stopcap',.45),('dwa_arc',.45),('dwa_arc',1.2),('stopcap',1.2)]:
    cr=[v for r in rows if r['root'] in ['RB_seedrep_s527','RB_seedrep_s531'] and r['cond_speed_governor_mode']==mode and float(r['cond_speed_governor_half_width_m'])==w for v in contact.get(r['root']+'/'+r['cell'],[])]
    rbcontacts.append(dict(mode=mode,width=w,n=len(cr),pinch=sum(bool(v['pinch_t0']) for v in cr),pinch_rate=100*sum(bool(v['pinch_t0']) for v in cr)/len(cr),lateral=desc([v['hit_lateral_cmd'] for v in cr]),opposite_gap=desc([max(v['gap_left_t0'],v['gap_right_t0']) for v in cr])))
# Expose duplicate treatment rows that a first-root-wins selector silently collapses.
groups=collections.defaultdict(list)
for r in rows:
    k=(r['checkpoint_sha256'],r['cond_seed'],r['cond_bars'],r['cond_speed_governor_mode'],r['cond_speed_governor_half_width_m'])
    groups[k].append(r)
dup=[]
for k,rr in groups.items():
    if len(rr)<2:continue
    sets={tuple(r[x] for x in ['actual_episodes','n_crash','n_captured','n_timeout']) for r in rr}
    dup.append(dict(key=k,rows=[r['root']+'/'+r['cell'] for r in rr],count_identical=len(sets)==1,counts=list(sets)))
report=dict(A7=a7,input_hashes={p.name:sha(p) for p in INPUT.iterdir() if p.is_file()},integrity=integrity,primary_n=sum(int(r['actual_episodes']) for r in cells.values()),contrasts=contrasts,totals=totals,width=width,D4=d4out,D4_interaction=d4interaction,C6=c6,C6_density_interaction=c6interaction,contacts=cs,RB_contacts=rbcontacts,duplicates=dup)
(OUT/'recomputed.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:report[k] for k in ['integrity','primary_n','totals','D4_interaction','C6_density_interaction','RB_contacts']},indent=2,ensure_ascii=False))
for name,c in contrasts.items():
 print(name,json.dumps({k:c[k] for k in ['pooled','per_density','per_seed','seed_t','new_seeds_only','density_heterogeneity','seed_heterogeneity']},ensure_ascii=False))
print('P5 details',json.dumps({k:stop[k] for k in ['70_seed_t','70_design_effect','low_vs_high_pooled','low_vs_high','15_cell_holm']},ensure_ascii=False))
