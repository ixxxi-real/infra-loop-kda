#!/usr/bin/env python3
"""Parse every captured action remotely with the matching Nsight Compute API."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnosis-output', help='optional normalized diagnosis.json path')
    parser.add_argument('--phase', default='prefill')
    parser.add_argument('--limiter-class', default='unknown', choices=(
        'memory_bandwidth', 'memory_latency', 'register_pressure',
        'occupancy', 'shared_memory', 'instruction_throughput',
        'launch_overhead', 'unknown'))
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
    parsed = {'report': args.report, 'actions': actions}
    Path(args.output).write_text(json.dumps(parsed, indent=2, default=str) + '\n')
    if args.diagnosis_output:
        report_hash = hashlib.sha256(Path(args.report).read_bytes()).hexdigest()
        diagnosis = {
            'schema_version': 1,
            'diagnosis_id': Path(args.diagnosis_output).stem,
            'kernel': actions[0]['kernel'],
            'phase': args.phase,
            'limiter_class': args.limiter_class,
            'profile_hash': report_hash,
            'source': 'ncu',
            'metrics': actions[0]['metrics'],
            'hypothesis': None,
        }
        Path(args.diagnosis_output).write_text(
            json.dumps(diagnosis, indent=2, default=str) + '\n'
        )
    print('Parsed', len(actions), 'actions')


if __name__ == '__main__':
    main()
