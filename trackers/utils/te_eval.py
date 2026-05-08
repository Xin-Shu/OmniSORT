"""
TrackEval wrapper for OmniSORT.

Expects the OmniSORT dataset layout:
    fp_dataset/
        seq_name/
            gt.txt              (MOT-format ground truth)
            result_{algo}.txt   (MOT-format tracker output)

Builds a temporary MOT-format directory tree, runs TrackEval's
MotChallenge2DBox evaluator with DO_PREPROC=False (to skip class
filtering that is specific to MOT17), and writes a dataset-level
summary to:
    fp_dataset/{algo_name}_trackeval.txt
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np

TRACKEVAL_PATH = os.path.expanduser('~/TrackEval')


def _import_trackeval():
    """Load the TrackEval package from TRACKEVAL_PATH without polluting sys.path."""
    pkg_path = os.path.join(TRACKEVAL_PATH, 'trackeval', '__init__.py')
    if not os.path.exists(pkg_path):
        raise ImportError(f'TrackEval package not found at {TRACKEVAL_PATH}')
    if 'trackeval' in sys.modules:
        return sys.modules['trackeval']
    if TRACKEVAL_PATH not in sys.path:
        sys.path.insert(0, TRACKEVAL_PATH)
    import trackeval as _te  # noqa: PLC0415
    return _te


def _count_frames(label_path):
    max_f = 0
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if parts and parts[0].strip():
                max_f = max(max_f, int(parts[0]))
    return max_f


def run_trackeval(fp_dataset, algo_name, gt_name='gt.txt'):
    """
    fp_dataset : absolute path to dataset folder (e.g. .../OmniSmall)
    algo_name  : algorithm name — looks for result_{algo_name}.txt per clip
    gt_name    : ground truth filename inside each clip folder

    Returns the combined-sequence results dict (indexed by metric name).
    Writes fp_dataset/{algo_name}_trackeval.txt.
    """
    _te = _import_trackeval()

    # --- Collect sequences that have both gt and result files ---
    seq_info = {}
    for name in sorted(os.listdir(fp_dataset)):
        clip_dir  = os.path.join(fp_dataset, name)
        ip_gt     = os.path.join(clip_dir, gt_name)
        ip_result = os.path.join(clip_dir, f'result_{algo_name}.txt')
        if os.path.isdir(clip_dir) and os.path.exists(ip_gt) and os.path.exists(ip_result):
            seq_info[name] = _count_frames(ip_gt)

    if not seq_info:
        raise FileNotFoundError(
            f'No sequences with {gt_name} + result_{algo_name}.txt found in {fp_dataset}')

    print(f'[TrackEval] Evaluating {algo_name} on {len(seq_info)} sequence(s): '
          f'{", ".join(sorted(seq_info))}')

    with tempfile.TemporaryDirectory() as tmp:
        gt_root  = os.path.join(tmp, 'gt')
        trk_root = os.path.join(tmp, 'trackers')
        out_root = os.path.join(tmp, 'output')

        for seq in seq_info:
            # GT: tmp/gt/seq/gt/gt.txt
            gt_dest = os.path.join(gt_root, seq, 'gt')
            os.makedirs(gt_dest, exist_ok=True)
            os.symlink(
                os.path.abspath(os.path.join(fp_dataset, seq, gt_name)),
                os.path.join(gt_dest, 'gt.txt'),
            )
            # Tracker: tmp/trackers/algo_name/data/seq.txt
            trk_dest = os.path.join(trk_root, algo_name, 'data')
            os.makedirs(trk_dest, exist_ok=True)
            os.symlink(
                os.path.abspath(os.path.join(fp_dataset, seq, f'result_{algo_name}.txt')),
                os.path.join(trk_dest, f'{seq}.txt'),
            )

        eval_config = {
            **_te.Evaluator.get_default_eval_config(),
            'PRINT_RESULTS':         True,
            'PRINT_CONFIG':          False,
            'TIME_PROGRESS':         False,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY':        False,
            'OUTPUT_DETAILED':       False,
            'PLOT_CURVES':           False,
        }
        dataset_config = {
            **_te.datasets.MotChallenge2DBox.get_default_dataset_config(),
            'GT_FOLDER':        gt_root,
            'TRACKERS_FOLDER':  trk_root,
            'OUTPUT_FOLDER':    out_root,
            'TRACKERS_TO_EVAL': [algo_name],
            'CLASSES_TO_EVAL':  ['pedestrian'],
            'BENCHMARK':        'OmniSORT',
            'SPLIT_TO_EVAL':    'all',
            'SKIP_SPLIT_FOL':   True,
            'SEQ_INFO':         seq_info,
            # DO_PREPROC=False: skip MOT17-specific class filtering so our
            # gt with class=-1 is not rejected as invalid.
            'DO_PREPROC':       False,
            'PRINT_CONFIG':     False,
        }

        evaluator    = _te.Evaluator(eval_config)
        dataset      = _te.datasets.MotChallenge2DBox(dataset_config)
        metrics_list = [
            _te.metrics.HOTA({}),
            _te.metrics.CLEAR({}),
            _te.metrics.Identity({}),
        ]

        output_res, _ = evaluator.evaluate([dataset], metrics_list)

    all_res  = output_res['MotChallenge2DBox'][algo_name]
    combined = all_res['COMBINED_SEQ']['pedestrian']
    _write_summary(fp_dataset, algo_name, seq_info, all_res)
    return combined


# ---------------------------------------------------------------------------
# Table writer
# ---------------------------------------------------------------------------

# Columns: (header, metric_group, field_key, is_pct_array)
# is_pct_array=True  → value is an ndarray, report mean*100 as percentage
# is_pct_array=False → scalar float (*100 for %) or int (reported as-is)
_COLUMNS = [
    ('HOTA',  'HOTA',     'HOTA',   True),
    ('DetA',  'HOTA',     'DetA',   True),
    ('AssA',  'HOTA',     'AssA',   True),
    ('LocA',  'HOTA',     'LocA',   True),
    ('MOTA',  'CLEAR',    'MOTA',   False),
    ('MOTP',  'CLEAR',    'MOTP',   False),
    ('IDF1',  'Identity', 'IDF1',   False),
    ('MT',    'CLEAR',    'MT',     None),
    ('ML',    'CLEAR',    'ML',     None),
    ('FP',    'CLEAR',    'CLR_FP', None),
    ('FN',    'CLEAR',    'CLR_FN', None),
    ('IDSW',  'CLEAR',    'IDSW',   None),
]


def _cell(res_group, field, is_pct_array):
    """Format one metric cell from a per-class result dict."""
    val = res_group.get(field)
    if val is None:
        return 'N/A'
    if is_pct_array is None:
        # integer count
        return str(int(round(float(np.sum(val)) if hasattr(val, '__len__') else float(val))))
    if is_pct_array:
        return f'{float(np.mean(val)) * 100:.2f}'
    return f'{float(val) * 100:.2f}'


def _row_values(per_cls, frames):
    """Return list of cell strings for one sequence or the combined row."""
    cells = [str(frames)]
    for _, group, field, mode in _COLUMNS:
        cells.append(_cell(per_cls.get(group, {}), field, mode))
    return cells


def _write_summary(fp_dataset, algo_name, seq_info, all_res):
    seqs = sorted(seq_info.keys())

    # Build all data rows first so we can compute column widths
    headers = ['Sequence', 'Frames'] + [c[0] for c in _COLUMNS]
    rows = []
    for seq in seqs:
        per_cls = all_res[seq]['pedestrian']
        rows.append([seq] + _row_values(per_cls, seq_info[seq]))
    combined_row = ['OVERALL', str(sum(seq_info.values()))] + \
                   _row_values(all_res['COMBINED_SEQ']['pedestrian'],
                               sum(seq_info.values()))[1:]

    # Column widths: max of header and all cell values
    widths = [max(len(h), max(len(r[i]) for r in rows + [combined_row]))
              for i, h in enumerate(headers)]

    def fmt_row(cells):
        return '  '.join(c.ljust(w) for c, w in zip(cells, widths))

    sep = '  '.join('-' * w for w in widths)

    lines = [
        f'Algorithm : {algo_name}',
        f'Date      : {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Sequences : {len(seqs)}  ({", ".join(seqs)})',
        '',
        '(Percentage metrics are ×100; integer metrics are raw counts)',
        '',
        fmt_row(headers),
        sep,
    ]
    for r in rows:
        lines.append(fmt_row(r))
    lines += [sep, fmt_row(combined_row)]

    ip_out = os.path.join(fp_dataset, f'{algo_name}_trackeval.txt')
    with open(ip_out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[TrackEval] Results written to {ip_out}')
