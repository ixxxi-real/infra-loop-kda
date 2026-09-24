#!/usr/bin/env python3
"""Parse every captured action remotely with the matching Nsight Compute API."""
import argparse
import json
import re
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expect-kernel', required=True, help='Verified target regular expression')
    parser.add_argument('--ncu-python', default='/opt/nvidia/nsight-compute/2025.3.1/extras/python')
    args = parser.parse_args()
    sys.path.insert(0, args.ncu_python)
    import ncu_report
    report = ncu_report.load_report(args.report)
    actions = []
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            metrics = {}
            for name in action.metric_names():
                try:
                    metric = action[name]
                    metrics[name] = {'value': metric.value(), 'unit': metric.unit()}
                except Exception as error:
                    metrics[name] = {'unavailable': str(error)}
            actions.append({'range': ri, 'action': ai, 'kernel': action.name(), 'metrics': metrics})
    if len(actions) != 1 or not re.search(args.expect_kernel, actions[0]['kernel']):
        raise RuntimeError('NCU report must contain exactly one matching target action: ' + str([a['kernel'] for a in actions]))
    Path(args.output).write_text(json.dumps({'report': args.report, 'actions': actions}, indent=2, default=str) + '\n')
    print('Parsed', len(actions), 'actions')


if __name__ == '__main__':
    main()
