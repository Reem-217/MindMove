"""merge_calibration.py — Append extra single-class trials to an existing calibration file.

USAGE
-----
    python merge_calibration.py \
        --base saed-data/saed_cal.npz \
        --extra saed-data/saed_extra_feet.npz \
        --out saed-data/saed_merged.npz

The --extra file may contain a subset of the base classes (e.g. feet only).
Each extra class is matched by name to the base class index and appended.
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base',  required=True, help='Original calibration .npz')
    p.add_argument('--extra', required=True, help='Extra trials .npz (subset of classes)')
    p.add_argument('--out',   required=True, help='Output merged .npz path')
    args = p.parse_args()

    base  = np.load(args.base,  allow_pickle=True)
    extra = np.load(args.extra, allow_pickle=True)

    base_names  = list(base['class_names'])
    extra_names = list(extra['class_names'])

    unknown = [n for n in extra_names if n not in base_names]
    if unknown:
        raise ValueError(f'Extra file has classes not in base: {unknown}\n'
                         f'  Base classes : {base_names}')

    X_base = base['X']
    y_base = base['y'].astype(np.int64)
    X_extra = extra['X']
    y_extra = extra['y'].astype(np.int64)

    # Remap extra labels to base indices
    remap = {i: base_names.index(extra_names[i]) for i in range(len(extra_names))}
    y_extra_remapped = np.array([remap[v] for v in y_extra], dtype=np.int64)

    X_merged = np.concatenate([X_base, X_extra], axis=0)
    y_merged = np.concatenate([y_base, y_extra_remapped], axis=0)

    np.savez(args.out, X=X_merged, y=y_merged,
             class_names=np.array(base_names))

    print(f'Merged -> {args.out}')
    print(f'  Total trials : {len(X_merged)}')
    for i, name in enumerate(base_names):
        print(f'  class {i} = {name:8s}  {int((y_merged == i).sum())} trials')


if __name__ == '__main__':
    main()
