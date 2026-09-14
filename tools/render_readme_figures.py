#!/usr/bin/env python3
"""Compatibility entry point for the current documentation figure build.

Run: python3 tools/render_readme_figures.py --raster
Historical drawing functions below are retained as source history, not called: some
panels were subsequently corrected by hand. The current builder preserves those
panels and adds evidence captions instead of overwriting the corrections.
"""
from pathlib import Path
from html import escape
import math

OUT = Path(__file__).resolve().parents[1] / 'docs/assets'
INK, MUTED, TEAL, BLUE, AMBER = '#142D3B', '#536B7A', '#007F73', '#3067C8', '#B66318'
BORDER, BG = '#DCE5EA', '#F4F7FA'

class Figure:
    def __init__(self, name, h, number, title, subtitle, status):
        self.name, self.h = name, h
        self.s = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{h}" viewBox="0 0 1440 {h}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#8395A1"/></marker></defs>
<g font-family="Arial, 'Noto Sans CJK KR', sans-serif">
<rect width="1440" height="{h}" rx="24" fill="{BG}"/>
<path d="M24 1H1416" stroke="{BORDER}"/>
''']
        self.text(40, 38, f'MOTAR   /   {number}', 13, TEAL, 700)
        self.text(1400, 38, status, 12, MUTED, 700, 'end')
        self.text(40, 86, title, 32, INK, 700)
        self.text(40, 119, subtitle, 16)
    def raw(self, s): self.s.append(s)
    def text(self,x,y,t,size=17,color=MUTED,weight=400,anchor='start'):
        self.raw(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(t)}</text>')
    def rect(self,x,y,w,h,fill='white',stroke=BORDER,r=16):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x,y,xx,yy,color=BORDER,dash=False,arrow=False):
        self.raw(f'<path d="M{x} {y}L{xx} {yy}" stroke="{color}" stroke-width="2" fill="none"'+(' stroke-dasharray="6 6"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def path(self,d,color=TEAL,fill='none',dash=False):
        self.raw(f'<path d="{d}" stroke="{color}" stroke-width="2" fill="{fill}"'+(' stroke-dasharray="7 6"' if dash else '')+'/>')
    def circle(self,x,y,r,fill,stroke='none'):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def card(self,x,y,w,h,tag,title,rows,accent=TEAL,tint='white'):
        self.rect(x,y,w,h,tint)
        self.rect(x+20,y+24,4,20,accent,accent,2)
        self.text(x+36,y+40,tag,12,accent,700)
        self.text(x+22,y+77,title,22,INK,700)
        for i,row in enumerate(rows): self.text(x+22,y+112+i*29,row,16)
    def footer(self,t):
        self.line(40,self.h-51,1400,self.h-51)
        self.text(40,self.h-23,t,14)
    def save(self):
        (OUT / f'motar-{self.name}.svg').write_text('\n'.join(self.s)+ '\n</g></svg>\n')

def overview():
    f=Figure('system-overview',650,'01 / SYSTEM','From perception to physically bounded flight','카메라·LiDAR·기체 상태 → 구조화된 관측 → 학습 정책 → 고정 비행 제어','CURRENT SOFTWARE PATH')
    xs=[40,394,748,1102]
    for a,b in zip(xs,xs[1:]): f.line(a+300,343,b-12,343,'#8395A1',arrow=True)
    f.card(40,158,300,346,'01  SENSING','Sensor-only inputs',[])
    for y,title,body in [(284,'RGB-D','1 detector → 1 KF track'),(370,'LiDAR','4 × 72 rays · 12 m'),(456,'Ego-state','velocity · yaw · height')]:
        f.circle(78,y-8,17,'#E3F2EF'); f.circle(78,y-8,6,'none',TEAL)
        f.text(108,y-7,title,22,INK,700);f.text(62,y+25,body,16)
    f.card(394,158,300,346,'02  STRUCTURED HISTORY','898-D observation',['17 tokens · 5 time steps'])
    for i,(label,n) in enumerate([('Static scan','288'),('Obstacles','480'),('Robot history','50'),('Target history','80')]):
        y=296+i*46; f.rect(416,y,256,36,'#EEF4F7','none',7); f.text(429,y+24,label,16); f.text(658,y+24,n,17,INK,700,'end')
    f.rect(748,158,300,346,'#086E66','#086E66')
    f.text(772,198,'03  LEARNED POLICY',12,'#B8E8DE',700)
    f.text(772,235,'Transformer',28,'white',700);f.text(772,270,'4 layers · 4 heads',18,'#D8EEE9')
    for i in range(4):
        f.rect(774+i*64,307,48,48,'#D5ECE6','none',9)
        for j in range(3): f.line(786+i*64,319+j*11,810+i*64,319+j*11,'#82BDB0')
    f.text(772,407,'PPO actor + critic',20,'white',700)
    f.text(772,439,'Bounded 4-D actor output',16,'#D8EEE9')
    f.text(772,475,'z channel → altitude PI',16,'#D8EEE9')
    f.card(1102,158,298,346,'04  FLIGHT','Fixed controller',['velocity + yaw · altitude hold'])
    for dx,dy in [(-48,-32),(48,-32),(-48,32),(48,32)]:
        f.line(1250,366,1250+dx,366+dy,TEAL); f.circle(1250+dx,366+dy,14,'white',TEAL)
    f.circle(1250,366,20,'#E3F2EF',TEAL)
    f.text(1124,448,'100 Hz physics',22,INK,700); f.text(1124,480,'10 Hz policy · moving target',16)
    f.rect(394,530,1006,45,'#E8F1EF','none',10)
    f.text(414,559,'TRAINING ONLY',12,TEAL,700);f.text(553,559,'Ground-truth reward + asymmetric critic · actor에는 직접 제공하지 않음',16)
    f.footer('학습 대상: navigation policy와 critic weight  /  센서·명령 범위·제어기·물리 파라미터는 고정된 실험 계약')
    f.save()

def arena():
    f=Figure('arena-geometry',950,'02 / ENVIRONMENT','Arena geometry & sensor coverage','40 × 40 × 3 m 아레나 · 막대 위치는 예시 · 센서 범위와 장애물 크기는 축척 적용','SIMULATION GEOMETRY')
    f.rect(40,156,716,724);f.text(64,191,'01  TOP VIEW',12,TEAL,700);f.text(730,191,'40 × 40 m',18,INK,700,'end')
    x,y,s=78,230,16
    f.rect(x,y,640,640,'#FCFDFE',BORDER,0)
    for i in range(1,8):
        f.line(x+i*80,y,x+i*80,y+640,'#ECF1F4');f.line(x,y+i*80,x+640,y+i*80,'#ECF1F4')
    # Clip the sensor footprints to the physical arena, rather than the image boundary.
    f.raw(f'<defs><clipPath id="arena"><rect x="{x}" y="{y}" width="640" height="640"/></clipPath></defs><g clip-path="url(#arena)">')
    px,py=x+6*s,y+34*s
    f.circle(px,py,12*s,'#EAF1FC',BLUE)
    def pt(r,a):return px+r*s*math.cos(math.radians(a)),py+r*s*math.sin(math.radians(a))
    def point(p):return f'{p[0]:.3f} {p[1]:.3f}'
    a,b=pt(20,-88.5),pt(20,-1.5)
    f.path(f'M{px} {py}L{point(a)}A320 320 0 0 1 {point(b)}Z',AMBER,'#FCF0DF')
    for r in (22.5,28):
        a,b=pt(r,-90),pt(r,0)
        f.path(f'M{point(a)}A{r*s} {r*s} 0 0 1 {point(b)}',TEAL,dash=True)
    for ox,oy in [(7,29),(12,27),(9.5,21.5),(16,24),(14.5,18),(20.5,20.5),(18.5,14),(25,15.5),(23,10.5),(29,13),(11,33.5),(17.5,31),(26.5,23),(31,19.5),(6,24.5),(13.5,12)]:
        f.rect(x+ox*s,y+oy*s,.7096*s,.5756*s,'#8FA2B2','#718799',1)
    tx,ty=pt(26,-43)
    f.line(px,py,tx,ty,'#8395A1',True,True)
    f.circle(px,py,7,BLUE);f.circle(tx,ty,7,'#C14E4E')
    f.raw('</g>')
    f.text(px-25,py+30,'Pursuer',17,BLUE,700)
    f.text(tx+15,ty-4,'Target',17,'#B24444',700);f.text(tx+15,ty+22,'0.3–1.5 m/s',15)
    f.text(203,499,'RGB-D · 87°',18,AMBER,700);f.text(203,525,'20 m range',16,AMBER)
    f.text(91,643,'LiDAR',17,BLUE,700);f.text(91,667,'12 m',16,BLUE)
    f.rect(526,644,178,66,'white','none',8)
    f.text(536,672,'Spawn goal band',16,TEAL,700);f.text(536,698,'22.5–28 m',18,TEAL)
    f.line(518,828,678,828,INK);f.line(518,822,518,834,INK);f.line(678,822,678,834,INK);f.text(598,853,'10 m',14,INK,400,'middle')
    f.rect(780,156,620,372);f.text(804,191,'02  SIDE VIEW',12,TEAL,700);f.text(804,225,'순항 고도와 LiDAR 수직 빔',22,INK,700)
    # Equal horizontal and vertical scale in the side inset: 30 units per metre.
    sx,ground=838,411
    f.rect(sx,ground-90,520,90,'#FCFDFE',BORDER,0)
    for bx in [938,1108,1288]:f.rect(bx,ground-60,.7096*30,60,'#8FA2B2','#718799',1)
    f.line(sx,ground-30,1358,ground-30,BLUE,True)
    f.text(825,ground+5,'0 m',13,MUTED,400,'end');f.text(825,ground-85,'3 m',13,MUTED,400,'end')
    cx,cy=990,ground-30
    for angle in [-10,0,10,20]:
        ex=cx+360*math.cos(math.radians(angle));ey=cy-360*math.sin(math.radians(angle)); f.line(cx,cy,ex,ey,BLUE)
    f.circle(cx,cy,6,BLUE);f.text(1328,252,'+20°',14,BLUE);f.text(1349,455,'−10°',14,BLUE)
    f.text(804,471,'1.0 m altitude hold · 4 beams · 10° spacing',16,BLUE)
    f.text(804,503,'2.0 m 막대 · 측정된 VERTICAL_OUT = 0.0%',16)
    f.card(780,548,620,190,'03  DENSITY LINEAGE','70 / 115 / 160 / 205 bars',[
        '70 bars     4.38 / 100 m²     바닥면적 1.8%',
        '205 bars   12.81 / 100 m²   바닥면적 5.2%',
        '300 bars: disconnected stress'],accent=TEAL)
    f.rect(780,758,620,112,'#E8F1EF','none');f.text(804,789,'OBSTACLE CONTRACT',12,TEAL,700)
    f.text(804,820,'0.7096 × 0.5756 × 2.0 m · bar_000.urdf',18,INK,700)
    f.text(804,848,'표면 여유 0.45 m · 중첩 금지',16)
    f.footer('축척: 평면 16 SVG units/m · 측면 30 SVG units/m  /  센서 영역은 아레나 경계에서 잘라 표시')
    f.save()

def platform():
    f=Figure('platform-hardware',1040,'03 / PLATFORM','Platform & sensing','navrl_ref5in_quad_v2 · hardware-informed simulation candidate','SIMULATION CANDIDATE · UNASSEMBLED')
    f.rect(40,155,440,489);f.text(64,191,'01  AIRFRAME / TOP VIEW',12,TEAL,700)
    cx,cy=260,402;half=220/math.sqrt(2)/2;scale=1.2
    f.rect(cx-169.8,cy-169.8,339.6,339.6,'none',AMBER,0)
    for dx,dy in [(-half,-half),(half,-half),(-half,half),(half,half)]:
        xx,yy=cx+dx*scale,cy+dy*scale
        f.circle(xx,yy,63.5*scale,'#EAF1FC',BLUE);f.line(cx,cy,xx,yy,INK);f.circle(xx,yy,6,INK)
    f.circle(cx,cy,24,INK);f.text(cx,cy+5,'FCU',13,'white',700,'middle')
    f.text(260,611,'127 mm propellers × 4 · 220 mm diagonal',16,INK,700,'middle')
    f.card(504,155,418,305,'02  MASS & THRUST','Thrust-to-weight 3.26',[])
    for i,(k,v) in enumerate([('질량','1.20 kg'),('모터당 최대추력','9.6 N'),('총 추력','38.4 N'),('중량','11.77 N')]):
        yy=278+i*36; f.text(528,yy,k,17);f.text(898,yy,v,19,INK,700,'end')
    f.rect(528,411,370,13,'#E3ECEF','none',5);f.rect(528,411,370*.306,13,TEAL,'none',5)
    f.text(528,448,'Hover 30.6%  /  Reserve 69.4%',16,TEAL)
    f.card(946,155,454,489,'03  SENSOR SUITE','LiDAR + RGB-D',[])
    sensor=[('LiDAR',BLUE),('72 × 4 beams · 360° × (+20°/−10°)',MUTED),('12 m range · 5° horizontal spacing',MUTED),('12 m에서 빔 간격 1.05 m',BLUE),('',MUTED),('RGB-D camera',AMBER),('160 × 90 · 87° × 58° (D455급)',MUTED),('Detector range 20 m',MUTED),('0.25 m 표적 @20 m → 1.05 px',AMBER),('Background depth 40 × 24 @10 m',MUTED)]
    for i,(t,c) in enumerate(sensor):f.text(970,280+i*34,t,17,c,700 if i in (0,5) else 400)
    f.rect(504,480,418,164,'#E8F1EF','none');f.text(528,516,'COLLISION PROXY',12,TEAL,700)
    f.text(528,553,'0.283 × 0.283 × 0.12 m',23,INK,700)
    f.text(528,588,'기체 그림: 1.2 SVG units/mm',16);f.text(528,617,'외곽선은 충돌 프록시의 평면 투영',16)
    f.rect(40,664,1360,179);f.text(64,699,'04  MOTION ENVELOPE',12,TEAL,700)
    values=[('수평 속도 / 축별','2.5 m/s','틸트 상한 45°'),('고도 / PI 고정','1.0 m','Yaw: config 2.5 rad/s'),('수평가속 / 45°','9.81 m/s²','정지거리 @2.5 m/s: 1.81 m'),('선회반경 @2.5 m/s','3.12 m','제동 2.0 m/s² · 반응 0.1 s')]
    for i,(a,b,c) in enumerate(values):
        xx=64+i*337
        if i:f.line(xx-16,718,xx-16,821)
        f.text(xx,737,a,15);f.text(xx,779,b,29,INK,700);f.text(xx,813,c,15)
    labels=[('Policy / PPO','vx · vy · yaw-rate'),('Altitude PI','policy z 덮어씀'),('Lee controller','Kv = 2.5'),('Motor allocation','lag = 0.04 s'),('Rigid-body physics','100 Hz')]
    for i,(a,b) in enumerate(labels):
        xx=40+i*278;f.rect(xx,866,248,88,'#E8F1EF' if i==0 else 'white');f.text(xx+18,900,a,19,TEAL if i==0 else INK,700);f.text(xx+18,930,b,16)
        if i<4:f.line(xx+253,910,xx+271,910,'#8395A1',arrow=True)
    f.footer('실기 미조립 · BOM/관성/추력/열/전원 실측값 아님  /  Yaw 상한: 기본 설정 2.5, canonical corrected-v2 실행은 3.0 rad/s')
    f.save()

def control():
    f=Figure('control-stack',1010,'04 / CONTROL','Policy action → force & torque → motor dynamics','학습하는 항법 정책과 고정된 비행 제어기 · corrected-v2 canonical 실행 계약','10 Hz POLICY / 100 Hz PHYSICS')
    data=[('01  LEARNED POLICY','Transformer actor',['Squashed Gaussian action','a = [aₓ, aᵧ, a_z, aψ]','aᵢ ∈ [−1, 1]','a_z is observed, not executed']),('02  COMMAND MAPPING','Body-frame setpoints',['v*xy = 2.5 · [aₓ, aᵧ]','ω*ψ = 3.0 · aψ','Per-axis clamp · m/s, rad/s','Baseline governor: OFF']),('03  ALTITUDE SUPERVISOR','Independent PI hold',['e_z = 1.0 m − z','I_z ← clip(I_z + e_zΔt, ±2.5)','v*z = clip(4e_z + I_z, ±2.5)','Overwrites the policy z channel']),('04  VELOCITY LOOP','Lee velocity control',['e_v = R_yaw v* − v_world','a* = K_v e_v','K_v = [2.5, 2.5, 2.5]','f* = m(a* − g)','Position term = 0 in velocity mode']),('05  FORCE → ATTITUDE','Tilt-limited orientation',['‖f*xy‖ ≤ f*z tan(45°)','b*₃ = f* / ‖f*‖ → R*','T = f*z / b₃z if b₃z > 0.5','Otherwise: T = f* · b₃','Altitude-priority compensation']),('06  ATTITUDE / RATE','Body torque command',['τ* = −K_R e_R − K_ω e_ω','       + ω × Jω','K_R = [1.0, 1.0, 0.5]','K_ω = [0.15, 0.15, 0.15]','Desired yaw rate from ω*ψ','Fixed gains; randomization off']),('07  ALLOCATION','Wrench → 4 motors',['w* = [0, 0, T, τx, τy, τz]','220 mm motor diagonal','9.6 N max per motor','Fixed allocation matrix','ref5in candidate values']),('08  PLANT','Rigid-body dynamics',['τ_motor = 0.04 s','100 Hz physics','10 physics steps / policy action','Mass · inertia · contact'])]
    for i,(tag,title,rows) in enumerate(data):
        col=i if i<4 else 7-i; yy=162 if i<4 else 540;xx=40+col*348
        f.card(xx,yy,316,310,tag,title,rows,tint='#E8F1EF' if i==0 else 'white')
        if i<3:f.line(xx+321,317,xx+338,317,'#8395A1',arrow=True)
        if 4<=i<7:f.line(xx-5,695,xx-23,695,'#8395A1',arrow=True)
    f.line(1242,480,1242,529,'#8395A1',arrow=True)
    f.raw('<path d="M198 862V903H1242V862" stroke="#8395A1" stroke-width="2" stroke-dasharray="6 6" fill="none" marker-end="url(#arrow)"/>')
    f.text(720,933,'STATE FEEDBACK · position / world velocity / attitude / body angular velocity',16,MUTED,400,'middle')
    f.footer('초록색: 학습 정책 출력  /  흰색: 고정된 task·controller·vehicle 계약  /  actor의 z 출력은 1 m altitude PI가 덮어씀')
    f.save()

def candidate():
    f=Figure('perception-candidate',1210,'05 / PERCEPTION ARCHIVE','Instance-preserving perception','대체된 SAM 설계 후보 · 제어루프 연결 및 성능 측정 없음 · 현재 경로는 160 × 90 / single KF','SUPERSEDED DESIGN / OFFLINE CPU ADAPTER ONLY')
    f.rect(40,149,1360,64,'#FFF1E3','#EDD5BC',10)
    f.text(64,176,'IMPLEMENTED: INSTANCE BOUNDARY ONLY',12,TEAL,700)
    f.text(64,200,'초록색 블록만 오프라인 CPU 구현. 주황색은 미구현 계획이며, 아래 안전 경로 역시 향후 계약입니다.',16,AMBER)
    xs=[40,388,736,1084]
    cards=[('A  SENSOR PACKET','동일 capture time 입력',['RGB · 해상도 미동결','Aligned depth + intrinsics','Camera extrinsics','pose(t_capture)','frame_id · timestamp','해상도/Hz는 측정 후 동결']),('B  DISCOVERY / PLANNED','SAM 3.1 worker',['Text + positive/negative','exemplar · 별도 환경 · 비동기','','Short-horizon tracker','Mask propagation + local IDs','SAM 응답 사이 프레임 담당']),('C  IMPLEMENTED / CPU','K개 instance 보존',['mask[K,H,W] · score[K]','bbox · uv · depth_median','Optional embedding','','K > 1: union centroid 금지','현재 CC stub · SAM worker 없음']),('D  ASSOCIATION / PLAN','3D gate + 다중가설 트랙',['Semantic / exemplar appearance','Bearing-range / motion consistency','Capture-time pose compensation','track_id · state · covariance · age','','단일 KF state를 바로 대체하지 않음'])]
    for i,(tag,title,rows) in enumerate(cards):
        f.card(xs[i],237,316,326,tag,title,rows,TEAL if i==2 else BLUE if i==0 else AMBER,'#E8F1EF' if i==2 else '#FFFAF4' if i in (1,3) else 'white')
        if i<3:f.line(xs[i]+321,399,xs[i]+338,399,'#8395A1',arrow=True)
    f.line(1242,573,1242,590,'#8395A1',arrow=True)
    f.card(40,606,432,210,'READING THE DESIGN','구현 범위와 남은 작업',['완료: K개 mask / 측정값 보존, union 금지','미완료: SAM worker, transport, association','Track bank, policy·safety 연결'],accent=MUTED)
    f.card(502,606,362,210,'POLICY BOUNDARY','기존 target-history token',['confidence · covariance · age','모든 gate 통과 후에만','898-D actor 관측 갱신'])
    f.line(896,709,876,709,'#8395A1',arrow=True)
    f.rect(908,606,492,210,'#FFFAF4');f.text(930,641,'E  FAIL-CLOSED DECISION / PLANNED',12,AMBER,700)
    for i,(a,b,c) in enumerate([('TARGET','유일한 gate 통과 track',TEAL),('AMBIGUOUS','선택 없이 예측 / 탐색 유지',AMBER),('REJECT','유효 후보 없음',INK)]):
        yy=682+i*38;f.text(930,yy,a,17,c,700);f.text(1060,yy,b,16)
    f.text(930,795,'항상 하나를 lock하는 fallback 금지',15)
    f.text(40,867,'INDEPENDENT SAFETY PATH / REQUIRED BEFORE CLOSED LOOP',13,BLUE,700)
    safety=[('Raw LiDAR + obstacle depth',['Semantic target mask와 독립','No-return = unknown, not free','동일 raw range → 동일 안전 출력']),('Occupancy / free-space',['표적 오인으로 bar return 삭제 금지','Collision avoidance · emergency stop','현재 governor와 다른 미래 계약']),('Controller safety input',['표적 인식은 mission input에만 영향','SAM timeout/crash가 안전 주기를 막지 않음','Semantic 변화 → safety output 동일 gate'])]
    for i,(title,rows) in enumerate(safety):
        xx=40+i*462;f.card(xx,890,436,201,f'0{i+1}  SAFETY CONTRACT',title,rows,BLUE)
        if i<2:f.line(xx+441,991,xx+453,991,'#8395A1',True,True)
    f.text(40,1130,'성능·실시간성·sim-to-real 주장은 단계별 held-out gate를 모두 통과한 뒤에만 가능합니다.',17)
    f.footer('ARCHIVED CANDIDATE  /  후속 방향: 실제 공대공 데이터로 인지 학습 → 측정된 detector 오차 모델을 시뮬레이션에 주입')
    f.save()

if __name__ == '__main__':
    import runpy
    runpy.run_path(str(Path(__file__).with_name('render_presentation_figures.py')), run_name='__main__')
