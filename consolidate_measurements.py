import csv
import os
import shutil
from pathlib import Path

HEADERS = (
    ('image_measurements', ['path', 'channel', 'px_min', 'px_max', 'px_mean', 'px_std', 'px_var']),
    ('crop_measurements', ['path', 'cell_idx', 'x', 'y', 'channel', 'px_min', 'px_max', 'px_mean', 'px_std', 'px_var'])
)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root-folder", help="Output path of crop script", default=os.getcwd())
    args = parser.parse_args()

    root_folder = Path(args.root_folder)

    for m in root_folder.glob('*_measurements'):
        if m.is_dir():
            with open(str(m.with_suffix('.csv')), 'w', newline='') as output_file:
                for typ, header in HEADERS:
                    if typ in m.name:
                        w = csv.writer(output_file)
                        w.writerow(header)

                for meas in m.glob('*'):
                    with open(str(meas), 'r') as input_file:
                        shutil.copyfileobj(input_file, output_file)


if __name__ == '__main__':
    main()
