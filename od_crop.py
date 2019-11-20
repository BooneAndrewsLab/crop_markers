import logging
import pathlib

import numpy as np
import pandas as p
from skimage.io import imread

log_fmt = "[%(asctime)-15s] %(name)s %(message)s"
log = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("coordinates", help="File with cell coordinates", type=argparse.FileType('r'))
    parser.add_argument("base_folder", help="Root folder for images", type=lambda x: pathlib.Path(x))
    parser.add_argument("output_folder", help="Recreate input structure in this folder", type=lambda x: pathlib.Path(x))
    parser.add_argument("-f", "--field", choices=['single', 'multi'], default='single')
    parser.add_argument("-c", "--channels", default=2)
    parser.add_argument("-s", "--crop-size", help="Size of cropped cells", default=64)

    args = parser.parse_args()

    log.info("Reading coordinates")
    locs = p.read_csv(args.coordinates)
    half_crop = args.crop_size // 2
    log.info("Half crop is %d", half_crop)

    # Filter border cells
    log.info("Finding border cells, %d pixels from border", half_crop)
    img = imread(str(args.base_folder / locs.loc[0, 'path']), plugin='tifffile')
    dim_y, dim_x = img.shape[-2:]

    good_y = locs.loc[:, 'center_y'].apply(lambda y: (y >= half_crop) & (y < dim_y - half_crop))
    good_x = locs.loc[:, 'center_x'].apply(lambda x: (x >= half_crop) & (x < dim_x - half_crop))
    mask = good_x & good_y

    log.info("Removing %d border cells", (~mask).sum())
    locs = locs.loc[mask]

    args.output_folder.mkdir(exist_ok=True, parents=True)

    cell_id_map = []

    count = 1
    all_images = locs['path'].unique().shape[0]
    for image_path, cell_data in locs.groupby('path'):
        log.info("Processing %d/%d", count, all_images)
        img = imread(str(args.base_folder / image_path), plugin='tifffile')

        if args.channels == 1:
            # Mojca is "special"
            img = img.reshape((1,) + img.shape)

        crops_path = args.output_folder / image_path
        crops_path.parent.mkdir(parents=True, exist_ok=True)

        crops_map = np.memmap(
            crops_path,
            dtype=img.dtype,
            mode='w+',
            shape=(cell_data.shape[0], args.channels, args.crop_size, args.crop_size)
        )

        internal_cell_id = 0
        for locs_idx, row in cell_data.iterrows():
            field = 0
            if args.field == 'multi':
                field = row.field

            crops_map[internal_cell_id] = img[
                                          (field * 2):(field * 2) + 2,
                                          row.center_y - half_crop:row.center_y + half_crop,
                                          row.center_x - half_crop:row.center_x + half_crop]

            cell_id_map.append((locs_idx, internal_cell_id))
            internal_cell_id += 1

        del crops_map

    log.info("Attaching internal cell indexing column")
    df_map = p.DataFrame(cell_id_map, columns=['idx', 'internal_cell_id'], dtype=int)
    df_map = df_map.set_index('idx')
    locs = locs.join(df_map, how='inner')

    output_path = args.output_folder / pathlib.Path(args.coordinates.name).name
    log.info("Saving modified coordinates to %s", output_path)
    locs.to_csv(output_path, index=False)

    log.info("DONE")


if __name__ == '__main__':
    main()
