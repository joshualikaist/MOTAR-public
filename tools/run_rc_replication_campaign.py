"""Run the authorized four R-C continuations, verify terminals, then evaluate eight cells."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'aerial_gym/rl_training/rl_games/runs'
PYTHON = '/home/fair/miniconda3/envs/aerialgym/bin/python'


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def run_checked(command, env, log):
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
        print('START pid=%d log=%s' % (child.pid, log), flush=True)
        while child.poll() is None:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                with log.open('rb') as reader:
                    reader.seek(max(0, log.stat().st_size - 2000))
                    tail = reader.read().decode(errors='replace').splitlines()[-2:]
                print('MONITOR pid=%d bytes=%d %s' % (
                    child.pid, log.stat().st_size, ' | '.join(tail)), flush=True)
        if child.returncode:
            raise RuntimeError('process failed exit=%d: %s' % (child.returncode, log))


def main():
    root = REPO / 'results/rc_replication_2026-09-09'
    if root.exists():
        raise SystemExit('refusing existing campaign directory: %s' % root)
    root.mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    env = dict(os.environ, PYTHON=PYTHON, PYTHONNOUSERSITE='1')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    checkpoints = []
    try:
        for seed in (233, 239):
            for arm in ('off', 'riskcap'):
                pattern = '*a8-readapt-%s-s%d' % (arm, seed)
                if list(RUNS.glob(pattern)):
                    raise RuntimeError('existing run requires inspection: %s' % pattern)
                current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
                if current != commit:
                    raise RuntimeError('HEAD changed during R-C training')
                run_checked(['bash', 'aerial_gym/rl_training/rl_games/train_navrl_v2_ref5in_a8_readapt.sh'],
                            dict(env, A8_MODE=arm, A8_SEED=str(seed)),
                            root / ('train_%s_s%d.log' % (arm, seed)))
                runs = list(RUNS.glob(pattern))
                if len(runs) != 1 or not (runs[0] / '.aerial_training_finished').is_file():
                    raise RuntimeError('missing unique finished run: %s' % pattern)
                paths = list((runs[0] / 'nn').glob('last_gen_ppo_ep_2900_*.pth'))
                if len(paths) != 1:
                    raise RuntimeError('missing unique terminal ep2900 checkpoint')
                checkpoint = paths[0]
                code = ('import torch,json,sys; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); '
                        's=c["env_state"]; print(json.dumps({"epoch":c["epoch"],'
                        '"seed":s["cfg_training_seed"],"governor":s["cfg_speed_governor_mode"]}))')
                metadata = json.loads(subprocess.check_output(
                    [PYTHON, '-c', code, str(checkpoint)], env=env, text=True))
                if metadata != {'epoch': 2900, 'seed': seed, 'governor': arm}:
                    raise RuntimeError('terminal contract mismatch: %r' % metadata)
                checkpoints.append(dict(metadata, path=str(checkpoint),
                                        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
                write_json(root / 'training_receipt.json', {'commit': commit, 'checkpoints': checkpoints})
                print('VERIFIED', metadata, flush=True)
        cells = []
        for checkpoint in checkpoints:
            group = 'T0' if checkpoint['governor'] == 'off' else 'T1'
            for governor in ('riskcap', 'stopcap'):
                cells.append({'name': '%ss%d_d070_%s' % (group, checkpoint['seed'], governor),
                              'policy': checkpoint['path'], 'bars': 70,
                              'env': {'NAVRL_SPEED_GOVERNOR': governor}})
        spec = REPO / 'docs/specs/grid_r2_d4_trainseed_rep.json'
        if spec.exists():
            raise RuntimeError('refusing existing R-C evaluation spec')
        write_json(spec, {'prereg': 'docs/plans/confirmation_phase_plan_2026-09-06.md',
                          'contract': 'ref5in', 'seed': 521, 'episodes': 2049,
                          'frame_sample_every': 100, 'cells': cells})
        run_checked([PYTHON, 'tools/run_navrl_filter_grid.py', str(spec),
                     str(REPO / 'results/navrl_grid_r2_d4_trainseed_rep')], env, root / 'evaluation.log')
        write_json(root / 'status.json', {'status': 'EVALUATION_PROCESS_COMPLETE',
                                         'training_runs': 4, 'evaluation_cells': 8, 'commit': commit})
    except Exception as exc:
        write_json(root / 'status.json', {'status': 'FAILED', 'error': str(exc),
                                         'completed_training_runs': len(checkpoints)})
        raise


if __name__ == '__main__':
    main()
