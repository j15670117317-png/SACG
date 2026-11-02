"""Publication config launcher. No changes to the supplied model or data loader."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rekognition_online_action_detection.config.defaults import get_cfg
from rekognition_online_action_detection.utils.parser import assert_and_infer_cfg


def load_config(path):
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # same as the original parser
    cfg.merge_from_file(str(path))
    return cfg


def data_check(cfg):
    p = Path(cfg.DATA.DATA_INFO)
    if not p.is_file():
        raise ValueError('Missing real data_info.json: {}. See data/README.md; do not use invented splits.'.format(p))
    info = json.loads(p.read_text(encoding='utf-8-sig'))
    if cfg.DATA.DATA_NAME not in info:
        raise ValueError('data_info.json has no dataset key ' + cfg.DATA.DATA_NAME)
    d = info[cfg.DATA.DATA_NAME]
    keys = ['data_root', 'class_names', 'num_classes', 'ignore_index', 'metrics', 'fps', 'train_session_set', 'test_session_set']
    if not isinstance(d, dict) or any(k not in d for k in keys):
        raise ValueError('Dataset entry requires: ' + ', '.join(keys))
    if not isinstance(d['class_names'], list) or not d['class_names'] or len(d['class_names']) != d['num_classes']:
        raise ValueError('class_names/num_classes are empty or inconsistent')
    for phase in ['train', 'test']:
        a = d[phase + '_session_set']
        if not isinstance(a, list) or not a or len(set(a)) != len(a):
            raise ValueError(phase + ' session list is empty, invalid or duplicated')
        for sid in a:
            if not isinstance(sid, str) or sid in ('.', '..') or any(c in sid for c in '/\\'):
                raise ValueError('Session IDs must be plain filename stems')
    if set(d['train_session_set']) & set(d['test_session_set']):
        raise ValueError('Training/evaluation session lists overlap')
    if not isinstance(d['fps'], (int, float)) or d['fps'] <= 0:
        raise ValueError('fps must be positive')
    return d


def prepare(args):
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = ROOT / config
    config = config.resolve()
    try:
        rel = config.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError('Keep the configuration under this release directory.')
    cfg = load_config(config)
    data_info = Path(args.data_info).expanduser().resolve() if args.data_info else ROOT / 'data/data_info.json'
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else (ROOT / cfg.DATA.DATA_ROOT).resolve()
    output = Path(args.output).expanduser().resolve() if args.output else ROOT / 'outputs'
    seed = args.seed if args.seed is not None else cfg.SEED
    session = args.session or ('eval_seed' if args.action == 'eval' else 'seed') + str(seed)
    if session in ('.', '..') or any(c in session for c in '/\\'):
        raise ValueError('SESSION must be a single folder name')
    overrides = ['DATA.DATA_INFO', str(data_info), 'DATA.DATA_ROOT', str(data_root), 'OUTPUT_DIR', str(output), 'SESSION', session, 'SEED', str(seed)]
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError('workers must be nonnegative')
        overrides += ['DATA_LOADER.NUM_WORKERS', str(args.workers)]
    if args.action == 'eval':
        if not args.checkpoint:
            raise ValueError('Evaluation requires --checkpoint; no automatic checkpoint selection.')
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError('Checkpoint not found: ' + str(checkpoint))
        overrides += ['MODEL.CHECKPOINT', str(checkpoint)]
    elif args.checkpoint:
        raise ValueError('--checkpoint is only accepted for eval; training here starts a new run.')
    elif cfg.MODEL.CHECKPOINT:
        raise ValueError('New training/check configs must not set MODEL.CHECKPOINT.')
    cfg.merge_from_list(overrides)
    data_check(cfg)
    assert_and_infer_cfg(cfg, SimpleNamespace(gpu=args.gpu, config_file=rel))
    cfg.TEST = int(args.action == 'eval')
    root = Path(cfg.DATA.DATA_ROOT)
    for subdir in [cfg.INPUT.VISUAL_FEATURE, cfg.INPUT.MOTION_FEATURE, cfg.INPUT.TARGET_PERFRAME]:
        if not (root / subdir).is_dir():
            raise ValueError('Missing feature/target directory: ' + str(root / subdir))
        for sid in list(cfg.DATA.TRAIN_SESSION_SET) + list(cfg.DATA.TEST_SESSION_SET):
            if not (root / subdir / (sid + '.npy')).is_file():
                raise ValueError('Missing .npy: ' + str(root / subdir / (sid + '.npy')))
    if cfg.DATA.DATA_NAME == 'TVSeries':
        if cfg.DATA.METRICS != 'cAP' or cfg.DATA.NUM_CLASSES != 31 or cfg.DATA.FPS != 4:
            raise ValueError('TVSeries requires 31 columns, 4 FPS and cAP.')
        if cfg.INPUT.TARGET_PERFRAME != 'target_perframe_drop_oob':
            raise ValueError('Publication TVSeries configs require target_perframe_drop_oob.')
    cmd = [sys.executable, '-u', 'main.py', '--test', str(cfg.TEST), '--config_file', rel, '--gpu', args.gpu] + overrides
    return cfg, cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'train', 'eval'])
    p.add_argument('--config')
    p.add_argument('--data-info')
    p.add_argument('--data-root')
    p.add_argument('--checkpoint')
    p.add_argument('--output')
    p.add_argument('--session')
    p.add_argument('--seed', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--gpu', default='0')
    args = p.parse_args(argv)
    if args.action == 'plan':
        index = json.loads((ROOT / 'configs/INDEX.json').read_text())
        for path, record in index.items():
            c = load_config(ROOT / path)
            m = c.MODEL.LSTR
            print('{} | GEN cross={} | CFN={} ({}, s={}) | {}'.format(path, m.GEN_CROSS_ATTN_TYPE, m.GEN_CONTRANORM, m.GEN_CONTRANORM_POS, m.GEN_CONTRANORM_SCALE, c.DATA.METRICS))
        return 0
    if not args.config:
        p.error('--config is required for check/train/eval')
    try:
        cfg, cmd = prepare(args)
        print('CONFIG/DATA PATH PASS (array contents and historical label provenance not audited)')
        print('Command arguments:', json.dumps(cmd, ensure_ascii=False))
        print('Output:', cfg.OUTPUT_DIR)
        if args.action == 'check':
            print(cfg.dump())
            return 0
        # Avoid creating an output directory for an unavailable training environment.
        subprocess.run([sys.executable, '-c', 'import torch,numpy,pandas,scipy,sklearn,einops,tqdm,yacs,matplotlib'], check=True)
        dest = Path(cfg.OUTPUT_DIR)
        dest.mkdir(parents=True, exist_ok=False)
        (dest / 'resolved_config.yaml').write_text(cfg.dump(), encoding='utf-8')
        versions = {}
        for name in ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'einops', 'tqdm', 'yacs', 'PyYAML', 'matplotlib']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        (dest / 'run_command.json').write_text(json.dumps({'argv': cmd, 'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}, indent=2), encoding='utf-8')
        result = subprocess.run(cmd, cwd=str(ROOT))
        (dest / 'exit_status.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result.returncode
    except (ValueError, OSError, AssertionError, KeyError, subprocess.CalledProcessError) as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
