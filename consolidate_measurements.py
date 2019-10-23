import csv
import os
import shutil
from pathlib import Path

HEADERS = (
    ('image_measurements', ['path', 'channel', 'px_min', 'px_max', 'px_mean', 'px_std', 'px_var']),
    ('crop_measurements',
     ['path', 'cell_index', 'row', 'column', 'channel', 'crop_min', 'crop_max', 'crop_mean', 'crop_std', 'crop_var',
      'area', 'centroid_row', 'centroid_column', 'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col',
      'bbox_area', 'eccentricity', 'extent', 'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity',
      'minor_axis_length', 'perimeter', 'solidity', 'area_perimeter_ratio', 'axis_length_major_minor_ratio'])
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
