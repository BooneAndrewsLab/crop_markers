import csv
import hashlib
import os
import pathlib
from collections import defaultdict

import numpy as np
from skimage.io import imread


def filter_coordinates(im_shape, coordinates, size):
    h, w = im_shape

    filtered = []
    for x, y in coordinates:
        if y - size > 0 and y + size < h and x - size > 0 and x + size < w:
            filtered.append((x, y))

    return filtered


def crop_image(im, cropped_path, coordinates, crop_size=64):
    """ Crop list of cells from image, save it to disk and return the data.
    Remember to delete returned reference for proper garbage collection just in case.

    :param im: array representation of an image
    :param cropped_path: where to save the cropped data
    :param coordinates: list of (x,y) coordinates (centre points) to crop
    :param crop_size: width and lenght of each crop
    :type im: np.ndarray
    :type cropped_path: str
    :type coordinates: list[tuple[int,int]]
    :type crop_size: int
    :return: cropped cells of shape (num_cells, channels, radius*2, radius*2)
    :rtype: np.ndarray
    """
    radius = crop_size // 2
    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    fp = np.memmap(cropped_path, dtype=im.dtype, mode='w+', shape=(len(coordinates), channels, crop_size, crop_size))
    for idx, coord in enumerate(coordinates):
        x, y = coord

        # Handle also images with only one channel so that the cropped files are always 4D
        if channels == 1:
            fp[idx, 0, :, :] = im[y - radius:y + radius, x - radius:x + radius]
        else:
            fp[idx, :, :, :] = im[:, y - radius:y + radius, x - radius:x + radius]
    fp.flush()  # Write data to disk

    return fp


def get_image_measurements(im):
    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    if channels == 1:
        yield 0, np.amin(im), np.amax(im), np.mean(im), np.std(im), np.var(im)
    else:
        for idx, ch in enumerate(im):
            yield idx, np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch)


def parse_coordinates(args):
    image_coordinates = defaultdict(list)
    screen_name = None
    num_cells = 0

    with open(args.cell_coordinates, newline='') as cell_coordinates:
        for row in csv.reader(cell_coordinates):
            image = row[0]
            if not screen_name:  # First image, lets check a few things
                screen_name = image.split(os.path.sep)[0]

            if args.multi_field_images:
                field, cell_x, cell_y = map(int, row[1:])
            else:
                field = 0
                cell_x, cell_y = map(int, row[1:])

            image = os.path.abspath(os.path.join(args.root_folder, image))
            image_coordinates[image].append((field, cell_x, cell_y))
            num_cells += 1

    return image_coordinates, screen_name, num_cells


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root-folder", help="Set base folder of images; defaults to CWD", default=os.getcwd())
    parser.add_argument("-s", "--crop-size", help="Size of the cropped cell", default=64)
    parser.add_argument("-f", "--multi-field-images", action='store_true', help="Images contain multiple fields")

    parser.add_argument("cell_coordinates", help="File containing location (image, x, y) of cells")
    parser.add_argument("output_folder", help="Recreate input structure in this folder")

    parser.add_argument("images", nargs="+", help="List of input images")
    args = parser.parse_args()

    image_coordinates, screen_name, num_cells = parse_coordinates(args)

    # This should exist if there is at least one row in coordinates file
    if not screen_name:
        parser.error("Coordinates file is empty")

    print("Processing %d images from screen '%s'" % (len(args.images), screen_name))
    print("Read coordinates for %d cells in %d images" % (num_cells, len(image_coordinates)))

    # Generate unique hash for our image set
    image_hash = hashlib.md5()
    for image in sorted(args.images):
        image_hash.update(image.encode())
    image_hash = image_hash.hexdigest()

    # Path where we'll save image measurements (min, max, std, ...)
    img_meas_path = os.path.abspath(os.path.join(args.output_folder, '%s_image_measurements' % screen_name, image_hash))
    crop_meas_path = os.path.abspath(os.path.join(args.output_folder, '%s_crop_measurements' % screen_name, image_hash))

    os.makedirs(os.path.dirname(img_meas_path), exist_ok=True)
    os.makedirs(os.path.dirname(crop_meas_path), exist_ok=True)

    print("Image measurements are saved in hash '%s'" % img_meas_path)

    with open(img_meas_path, 'w', newline='') as img_meas_file, \
            open(crop_meas_path, 'w', newline='') as crop_meas_file:
        img_meas_writer = csv.writer(img_meas_file)
        crop_meas_writer = csv.writer(crop_meas_file)

        for rel_image_path in args.images:
            image_path = os.path.abspath(rel_image_path)  # normalize path, useful later
            cells = image_coordinates.get(image_path, [])
            cropped_path = pathlib.Path(os.path.abspath(image_path.replace(args.root_folder, args.output_folder)))
            cropped_path = cropped_path.with_suffix('.dat')

            os.makedirs(os.path.dirname(cropped_path), exist_ok=True)

            img = imread(image_path, plugin='tifffile')

            coords = filter_coordinates(img.shape[-2:], cells, args.crop_size // 2)

            print("Cropping %d (%d excluded) cells from %s to '%s'" % (
                len(coords), len(cells) - len(coords), image_path, cropped_path))

            if not coords:  # Empty image, crop would throw an exception
                continue

            cropped = crop_image(img, cropped_path, coords, crop_size=args.crop_size)
            cell_idx = 0
            for crop, crop_coordinates in zip(cropped, coords):
                row_common = (pathlib.Path(rel_image_path).with_suffix('.dat'), cell_idx) + crop_coordinates
                for values in get_image_measurements(crop):
                    # noinspection PyTypeChecker
                    crop_meas_writer.writerow(row_common + values)
                cell_idx += 1

            del cropped

            for values in get_image_measurements(img):
                # noinspection PyTypeChecker
                img_meas_writer.writerow((rel_image_path,) + values)


if __name__ == '__main__':
    main()
