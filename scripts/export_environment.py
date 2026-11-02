"""Run in the actual experiment environment; does not install or change packages."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    dest = Path(args.output)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True).stdout
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'requirements-frozen.txt').write_text(freeze, encoding='utf-8')
    record = {'python': sys.version, 'platform': platform.platform()}
    try:
        import torch
        record.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            record['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        record['torch'] = 'not installed'
    (dest / 'environment.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('Saved', dest, '(review local paths in pip freeze before public upload)')


if __name__ == '__main__':
    main()
