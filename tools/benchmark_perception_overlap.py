"""S1: three serial/overlap validation pairs, exact parity and GPU contention audit."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from perception_candidates import sha256_file
from runtime_fingerprint import runtime_fingerprint


def gpu_snapshot(allowed_pid=None):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    unexpected=[]
    for line in processes.splitlines():
        pid,name=line.split(',',1)
        if int(pid.strip())!=allowed_pid and name.strip()!='/usr/share/rustdesk/rustdesk':
            unexpected.append(line)
    if unexpected:
        raise RuntimeError('Other GPU compute jobs: '+repr(unexpected))
    state=subprocess.check_output(['nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'],text=True).strip()
    return {'time_unix':time.time(),'compute_processes':processes,'gpu_util_memory_temp_power':state}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--yolov5',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'runtime':runtime_fingerprint(),'test_used':False,'runs':[],
             'source_sha256':sha256_file(__file__),
             'desktop_exception':'Existing rustdesk and graphical desktop remain active; no training allowed.',
             'order':['serial','overlap','overlap','serial','serial','overlap']}
    counts={'serial':0,'overlap':0}
    for mode in receipt['order']:
        counts[mode]+=1
        name=f'{mode}_{counts[mode]}'
        output=args.output/(name+'.json')
        snapshots=[gpu_snapshot()]
        cmd=[sys.executable,str(Path(__file__).with_name('verify_perception_streaming.py')),
             '--mode','rgb','--data-root',str(args.data_root),'--run',str(args.run),
             '--weights',str(args.weights),'--yolov5',str(args.yolov5),'--output',str(output)]
        if mode=='serial': cmd.append('--serial-flow')
        print('Starting',name,flush=True)
        with (args.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        snapshots.append(gpu_snapshot(child.pid))
            except BaseException:
                child.terminate()
                child.wait()
                raise
        run={'name':name,'returncode':child.returncode,'gpu_snapshots':snapshots,
             'report_sha256':sha256_file(output) if output.exists() else None}
        receipt['runs'].append(run)
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if child.returncode: raise RuntimeError(name+' failed; inspect preserved log/report')
        report=json.loads(output.read_text())
        print(name,'PASS',report['fps_with_decode'],'FPS',flush=True)
    reports=[json.loads((args.output/(r['name']+'.json')).read_text()) for r in receipt['runs']]
    receipt['all_motion_hashes_equal']=len({r['motion_float32_sha256'] for r in reports})==1
    receipt['all_metrics_equal']=all(r['metrics']==reports[0]['metrics'] for r in reports)
    receipt['all_runtimes_equal']=all(r['runtime']==reports[0]['runtime'] for r in reports)
    receipt['pass']=all(receipt[k] for k in ('all_motion_hashes_equal','all_metrics_equal','all_runtimes_equal'))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not receipt['pass']: raise RuntimeError('Cross-run parity failed')


if __name__=='__main__': main()
