#!/usr/bin/env python3
"""Plot audited statistics; requires matplotlib, never imports project analysis tools."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
j=json.loads((P/'recomputed.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'axes.titleweight':'bold','pdf.fonttype':42})
colors=['#007F73','#3067C8','#B66318']
fig,axes=plt.subplots(1,3,figsize=(13,4.1),sharey=True)
for ax,(key,c) in zip(axes,zip(['dwa_arc-riskcap','dwa_arc-stopcap','stopcap-riskcap'],colors)):
 rs=j['contrasts'][key]['per_density'];y=list(range(5))
 ax.errorbar([r['delta'] for r in rs],y,xerr=[1.959963984540054*r['se'] for r in rs],fmt='o',color=c,capsize=4,lw=1.7)
 ax.axvline(0,color='#8A9CA7',ls='--',lw=1);ax.set_yticks(y,[str(r['density']) for r in rs]);ax.set_title(key.replace('dwa_arc','arc').replace('-',' − '),fontsize=12);ax.set_xlabel('Crash-rate difference (pp)');ax.grid(axis='x',alpha=.15)
 for yy,r in zip(y,rs):ax.annotate(f"Holm p={r['p_holm']:.3g}",(r['hi'],yy),xytext=(5,5),textcoords='offset points',fontsize=8,color='#536B7A')
 ax.set_xlim(-3.8,3.4)
axes[0].set_ylabel('Obstacle density (bars)');axes[0].invert_yaxis()
fig.suptitle('Fixed-effect inverse-variance pooling across evaluation seeds 523, 527, 531',fontsize=12)
fig.text(.5,.01,'Intervals: unadjusted 95% Wald. Holm correction is within each five-density contrast family. Episode independence assumed.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.93])
for ext in ['png','pdf']:fig.savefig(P/f'figure_density_contrasts.{ext}',bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,metric in zip(axes,['crash','captured']):
 for i,arm in enumerate(['dwa_arc','stopcap']):
  rs=[r for r in j['width'] if r['arm']==arm];x=[r['density']+(-3 if i==0 else 3) for r in rs]
  ax.errorbar(x,[r[metric]['delta'] for r in rs],yerr=[1.959963984540054*r[metric]['se'] for r in rs],fmt='o-',color=colors[i],capsize=4,label=arm.replace('dwa_arc','arc'))
 ax.axhline(0,color='#8A9CA7',ls='--',lw=1);ax.set_xticks([70,205]);ax.set_xlabel('Obstacle density (bars)');ax.set_ylabel('Rate difference (pp)');ax.set_title(metric.capitalize()+' · width 1.20 − 0.45 m');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
fig.text(.5,.005,'Three evaluation seeds. Negative crash difference does not imply mission success: wide stopcap mainly increases timeout.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,1])
for ext in ['png','pdf']:fig.savefig(P/f'figure_width_tradeoff.{ext}',bbox_inches='tight')
plt.close(fig)
