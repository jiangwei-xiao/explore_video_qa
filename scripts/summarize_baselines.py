import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.baseline_report import summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    print({m: report['methods'][m]['by_stratum']['all']['accuracy'] for m in report['methods']})
