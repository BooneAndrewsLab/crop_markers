import hashlib
import os
import pathlib
from collections import defaultdict

import numpy as np
from skimage.io import imread


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
        y, x = coord

        # Handle also images with only one channel so that the cropped files are always 4D
        if channels == 1:
            fp[idx, 0, :, :] = im[x - radius:x + radius, y - radius:y + radius]
        else:
            fp[idx, :, :, :] = im[:, x - radius:x + radius, y - radius:y + radius]
    fp.flush()  # Write data to disk

    return fp


def get_image_measurements(im):
    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    if channels == 1:
        yield 0, np.amin(im), np.amax(im), np.mean(im), np.std(im), np.var(im)
    else:
        for idx, ch in enumerate(im):
            yield idx, np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root-folder", help="Set base folder of images; defaults to CWD", default=os.getcwd())

    parser.add_argument("cell_coordinates", help="File containing location (image, x, y) of cells")
    parser.add_argument("output_folder", help="Recreate input structure in this folder")

    parser.add_argument("images", nargs="+", help="List of input images")
    args = parser.parse_args()

    screen_name = None

    # Read coordinates group them together per image
    image_coordinates = defaultdict(list)
    num_cells = 0
    with open(args.cell_coordinates) as cell_coordinates:
        for line in cell_coordinates:
            image, cell_x, cell_y = line.strip().split(',')
            if not screen_name:  # First image, lets check a few things
                screen_name = image.split(os.path.sep)[0]

            image = os.path.abspath(os.path.join(args.root_folder, image))
            image_coordinates[image].append((int(cell_x), int(cell_y)))
            num_cells += 1

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
    measurements_path = os.path.abspath(os.path.join(args.output_folder, '%s_measurements' % screen_name, image_hash))
    os.makedirs(os.path.dirname(measurements_path), exist_ok=True)

    print("Image measurements are saved in hash '%s'" % measurements_path)

    with open(measurements_path, 'w') as meas_file:
        for rel_image_path in args.images:
            image_path = os.path.abspath(rel_image_path)  # normalize path, useful later
            cells = image_coordinates.get(image_path, [])
            cropped_path = pathlib.Path(os.path.abspath(image_path.replace(args.root_folder, args.output_folder)))
            cropped_path = cropped_path.with_suffix('.dat')

            os.makedirs(os.path.dirname(cropped_path), exist_ok=True)

            print("Cropping %d cells from '%s' to '%s'" % (len(cells), image_path, cropped_path))

            img = imread(image_path, plugin='tifffile')
            cropped = crop_image(img, cropped_path, cells)
            del cropped

            for values in get_image_measurements(img):
                meas_file.write('%s,' % (rel_image_path, ))
                meas_file.write(','.join(map(str, values)) + '\n')


if __name__ == '__main__':
    main()
