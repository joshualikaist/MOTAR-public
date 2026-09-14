import csv, json, math, statistics, collections, itertools
from pathlib import Path
REPO=Path("/home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator")
rows=list(csv.DictReader(open(REPO/"results/verification_export_2026-09-06/governor_cells.csv")))
def I(x): return int(x) if x not in ("",None) else None
def F(x): return float(x) if x not in ("",None) else None
def wald(a,b,key="n_crash"):
    pa,na=I(a[key])/I(a["actual_episodes"]),I(a["actual_episodes"]); pb,nb=I(b[key])/I(b["actual_episodes"]),I(b["actual_episodes"])
    d=100*(pa-pb); se=100*math.sqrt(pa*(1-pa)/na+pb*(1-pb)/nb); return d,d-1.96*se,d+1.96*se,se
def recs(r):
    p=r["contact_records_path"]; return [json.loads(l) for l in open(p)] if p and Path(p).is_file() else []
def frames(r):
    p=r["contact_records_path"]
    if not p: return []
    q=Path(p).with_name(Path(p).name.replace(".contact_records.jsonl",".frame_samples.jsonl"))
    return [json.loads(l) for l in open(q)] if q.is_file() else []
byname={(r["root"],r["cell"]):r for r in rows}

print("="*78); print("0. 비율 재계산 vs JSON 기록 비율 (83 cell)")
bad=0
for r in rows:
    j=json.loads((REPO/r["result_path"]).read_text()); n=I(r["actual_episodes"])
    for k,jk in (("n_crash","crash_rate"),("n_captured","capture_rate"),("n_timeout","timeout_rate")):
        if abs(I(r[k])/n-j["outcome"][jk])>1e-9: bad+=1; print("  MISMATCH",r["root"],r["cell"],k)
    if I(r["n_crash"])+I(r["n_captured"])+I(r["n_timeout"])!=n: bad+=1; print("  SUM!=n",r["root"],r["cell"])
print("  불일치:",bad)

print("="*78); print("1. cond_* vs req_* 어긋남 (cells.json이 있는 루트만)")
mism=0
for r in rows:
    for k,v in r.items():
        if not k.startswith("req_gov_") or v in ("",None): continue
        suffix=k[len("req_gov_"):]; ck="cond_speed_governor_"+suffix
        if ck in r and r[ck] not in ("",None) and abs(float(r[ck])-float(v))>1e-9:
            mism+=1; print("  ",r["root"],r["cell"],suffix,"cond",r[ck],"req",v)
        if ck in r and r[ck] in ("",None): mism+=1; print("  기록누락",r["root"],r["cell"],suffix,"req",v)
    if r.get("req_gov")=="" : pass
    m=r.get("req_gov"); 
    if m and m!=r["cond_speed_governor_mode"]: mism+=1; print("  MODE",r["root"],r["cell"],m,r["cond_speed_governor_mode"])
print("  어긋남/누락 건수:",mism)

print("="*78); print("2. 교차 루트 동일 조건 재현성 (같은 정책·bars·mode·폭·시드·계약)")
key=lambda r:(r["checkpoint_sha256"],r["cond_bars"],r["cond_speed_governor_mode"],r["cond_speed_governor_half_width_m"],r["cond_seed"],r["cond_goal_dist_min_m"],r.get("req_gov_width_per_open_m") or "")
groups=collections.defaultdict(list)
for r in rows: groups[key(r)].append(r)
for k,g in sorted(groups.items()):
    if len(g)<2: continue
    outs=[(x["root"],x["runtime_git_commit"],I(x["n_crash"]),I(x["n_captured"]),I(x["n_timeout"]),I(x["actual_episodes"])) for x in g]
    same=len({o[2:] for o in outs})==1
    print("  %s | bars %s %s w=%s | %s"%("동일" if same else "다름",k[1],k[2],k[3]," ; ".join("%s@%s crash %d cap %d to %d n %d"%o for o in outs)))

print("="*78); print("3. 주장1 순환성: 범주 대신 고정 0.45 m 문턱의 측면 오프셋 분포 (접촉별 hit_lateral_cmd, 1 s 전)")
for r in rows:
    if r["root"] not in ("L1_width_s523","L1b_width_adaptive_s523"): continue
    if r["cond_speed_governor_mode"]!="dwa_arc" or (r.get("req_gov_width_per_open_m") or ""): continue
    rs=recs(r); 
    if not rs: continue
    lat=[x["hit_lateral_cmd"] for x in rs]; fixed=sum(l>0.45 for l in lat)
    print("  bars %s w=%.2f  접촉 %3d  범주lateral %3d  고정문턱>0.45m %3d  측면오프셋 중앙값 %.2f m"%(r["cond_bars"],F(r["cond_speed_governor_half_width_m"]),len(rs),I(r["contact_lateral"]),fixed,statistics.median(lat)))

print("="*78); print("4. 주장2: 접촉 1 s 전 거리·속도·요레이트·슬립 — 측면 vs 회랑안 (D1p, dwa_arc·riskcap·stopcap 합산)")
pool=collections.defaultdict(list)
for r in rows:
    if r["root"]!="D1p_ep25000_s523": continue
    for x in recs(r): pool[(r["cond_bars"],x["category_cmd"])].append(x)
med=lambda xs,k: statistics.median([x[k] for x in xs if x.get(k) is not None]) if xs else float("nan")
print("  %-5s %-9s %5s %8s %8s %8s %8s %8s %8s %9s"%("bars","cat","n","surf_t1","gap_min","v_t1","v_t05","|yaw|","slip°","seen&lost"))
for bars in ("70","130","205"):
    for cat,name in ((2,"lateral"),(4,"in_corr")):
        xs=pool[(bars,cat)]
        if not xs: continue
        print("  %-5s %-9s %5d %8.2f %8.2f %8.2f %8.2f %8.2f %8.1f %8.0f%%"%(bars,name,len(xs),med(xs,"hit_surface"),statistics.median([min(x["gap_left"],x["gap_right"]) for x in xs]),med(xs,"speed_act_t1"),med(xs,"speed_act_t05"),statistics.median([abs(x["yaw_rate_t1"]) for x in xs]),med(xs,"cmd_vs_actual_deg"),100*sum(x["seen_then_lost"] for x in xs if x["memory_window_valid"])/max(1,sum(x["memory_window_valid"] for x in xs))))
# closing test: could the drone reach the struck bar's surface in 1 s at v_t1?
xs=[x for b in ("70","130","205") for x in pool[(b,2)]]
reach=sum(x["hit_surface"]<=x["speed_act_t1"]*1.0 for x in xs)/len(xs)
print("  측면 접촉 중 '1 s 전 속도로 1 s 안에 그 막대 표면에 닿을 수 있었던' 비율: %.0f%%"%(100*reach))

print("="*78); print("5. 주장3 D4: 재계산 + 대안(T1이 그냥 전반적으로 나쁜 정책인가?)")
d4={r["cell"]:r for r in rows if r["root"]=="D4_coadapt_s521"}; a8={r["cell"]:r for r in rows if r["root"]=="A8_readapt_s521"}
for pol in ("T0","T1"):
    d,lo,hi,se=wald(d4[f"{pol}_d070_stopcap"],d4[f"{pol}_d070_riskcap"]); print("  %s stopcap-riskcap %+.2f [%+.2f,%+.2f]"%(pol,d,lo,hi))
for pol in ("T0","T1"):
    for m in ("riskcap","stopcap"):
        r=d4[f"{pol}_d070_{m}"]; print("  %s/%-8s crash %5.2f%% interv %5.1f%% exec %.2f m/s req %.2f"%(pol,m,100*I(r["n_crash"])/I(r["actual_episodes"]),100*F(r["gov_intervention_rate"]),F(r["gov_mean_executed_mps"]),F(r["gov_mean_requested_mps"])))
print("  A8 off cells: T0_off %.2f%%  T1_off %.2f%%  (T1이 필터 없이도 나쁨)"%(100*I(a8["T0_off"]["n_crash"])/I(a8["T0_off"]["actual_episodes"]),100*I(a8["T1_off"]["n_crash"])/I(a8["T1_off"]["actual_episodes"])))
print("  법칙별 이득(off 대비): T0 riskcap %+.2f stopcap %+.2f | T1 riskcap %+.2f stopcap %+.2f"%(
    wald(d4["T0_d070_riskcap"],a8["T0_off"])[0],wald(d4["T0_d070_stopcap"],a8["T0_off"])[0],wald(d4["T1_d070_riskcap"],a8["T1_off"])[0],wald(d4["T1_d070_stopcap"],a8["T1_off"])[0]))
print("  (주의: off는 A8 루트, 다른 커밋 — 교차루트 비교)")

print("="*78); print("6. 주장4: dwa_arc − riskcap 다섯 밀도, 부호검정 + 역분산 가중 결합")
d1p={(r["cond_bars"],r["cond_speed_governor_mode"]):r for r in rows if r["root"]=="D1p_ep25000_s523"}
ds=[];ws=[]
for b in ("70","100","130","160","205"):
    d,lo,hi,se=wald(d1p[(b,"dwa_arc")],d1p[(b,"riskcap")]); ds.append(d); ws.append(1/se**2); print("  bars %3s  %+.2f [%+.2f,%+.2f]"%(b,d,lo,hi))
pooled=sum(d*w for d,w in zip(ds,ws))/sum(ws); pse=1/math.sqrt(sum(ws)); z=pooled/pse
print("  부호검정: 5/5 음수, 양측 p = %.3f (=2·0.5^5)"%(2*0.5**5))
print("  역분산 결합: %+.2f pp, SE %.2f, z=%.2f, 95%% CI [%+.2f, %+.2f]"%(pooled,pse,z,pooled-1.96*pse,pooled+1.96*pse))
ds2=[];ws2=[]
for b in ("70","100","130","160","205"):
    d,lo,hi,se=wald(d1p[(b,"dwa_arc")],d1p[(b,"stopcap")]); ds2.append(d); ws2.append(1/se**2)
p2=sum(d*w for d,w in zip(ds2,ws2))/sum(ws2); s2=1/math.sqrt(sum(ws2)); print("  dwa_arc − stopcap 결합: %+.2f pp [%+.2f, %+.2f]"%(p2,p2-1.96*s2,p2+1.96*s2))
# within-root replication in L1 root (w=0.45) and D1p root: same numbers?
for b in ("70","205"):
    l1=byname[("L1_width_s523",f"ep25000_d{int(b):03d}_dwa_arc_w0p45")]; print("  L1 루트 재현 bars %s dwa_arc w0.45: crash %s vs D1p %s"%(b,l1["n_crash"],d1p[(b,"dwa_arc")]["n_crash"]))

print("="*78); print("7. 주장5: 폭 곡선 U자 유의성 + 정의독립 지표(총 접촉 수, 고정문턱 측면 수)")
for bars in ("70","205"):
    for mode in ("dwa_arc","stopcap"):
        cells=[r for r in rows if r["root"] in ("L1_width_s523","L1b_width_adaptive_s523") and r["cond_bars"]==bars and r["cond_speed_governor_mode"]==mode and not (r.get("req_gov_width_per_open_m") or "")]
        cells.sort(key=lambda r:F(r["cond_speed_governor_half_width_m"]))
        best=min(cells,key=lambda r:I(r["n_crash"]))
        line=[]
        for r in cells:
            rs=recs(r); fixed=sum(x["hit_lateral_cmd"]>0.45 for x in rs)
            line.append("w%.2f:crash%d/lat>0.45:%d/tot:%d"%(F(r["cond_speed_governor_half_width_m"]),I(r["n_crash"]),fixed,len(rs)))
        print("  bars %s %-8s"%(bars,mode)); print("    "+" | ".join(line))
        ends=[c for c in cells if c is not best]
        w0=cells[0]; wmax=cells[-1]
        for tag,c in (("최소폭",w0),("최대폭",wmax)):
            d,lo,hi,se=wald(c,best); print("    best w=%.2f vs %s w=%.2f: %+.2f [%+.2f,%+.2f]"%(F(best["cond_speed_governor_half_width_m"]),tag,F(c["cond_speed_governor_half_width_m"]),d,lo,hi))

print("="*78); print("8. L8: 프레임 표본으로 실제 적용된 폭 분포 복원 (w = 0.45 + k(nearest−1.2), [0.1,3.0])")
for r in rows:
    k=r.get("req_gov_width_per_open_m")
    if not k: continue
    k=float(k); fr=frames(r)
    if not fr: print("  프레임 표본 없음",r["cell"]); continue
    w=[min(3.0,max(0.1,0.45+k*(x["nearest_surface"]-1.2))) for x in fr]
    below=sum(x<0.45 for x in w)/len(w)
    print("  %-30s k=%.1f  w 중앙값 %.2f  p10 %.2f  p90 %.2f  기본폭 0.45 미만 프레임 %.0f%%"%(r["cell"],k,statistics.median(w),sorted(w)[len(w)//10],sorted(w)[9*len(w)//10],100*below))
