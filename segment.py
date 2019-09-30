import csv
import hashlib
import os
import pathlib

import mahotas as mh
import numpy as np
import scipy.ndimage as nd
import segmentation
from scipy.ndimage import binary_erosion
from skimage.io import imread
from skimage.morphology import disk


def Watershed_MRF(Iin, I_MM):
    # ------------------------------------------------------------------------------------ #
    #                                                                                      #
    # This algorithm is implemented by Oren Kraus on July 2013                             #
    #                                                                                      #
    # ------------------------------------------------------------------------------------ #

    Fgm = (I_MM > 0)
    SdsLab = mh.label(I_MM == 2)[0]
    SdsSizes = mh.labeled.labeled_size(SdsLab)
    too_small_Sds = np.where(SdsSizes < 30)
    SdsLab = mh.labeled.remove_regions(SdsLab, too_small_Sds)
    Sds = SdsLab > 0

    se2 = nd.generate_binary_structure(2, 2).astype(np.int)
    dilatedNuc = nd.binary_dilation(Sds, se2)
    Fgm = (dilatedNuc.astype(np.int) + Fgm.astype(np.int)) > 0

    FgmLab = mh.label(Fgm)[0]
    FgmSizes = mh.labeled.labeled_size(FgmLab)
    too_small_Fgm = np.where(FgmSizes < 30)
    FgmLab = mh.labeled.remove_regions(FgmLab, too_small_Fgm)
    Fgm = FgmLab > 0

    se3 = nd.generate_binary_structure(2, 1).astype(np.int)
    Fgm = nd.binary_erosion(Fgm, structure=se3)

    Fgm_Lab, Fgm_num = nd.measurements.label(Fgm)

    Nuc_Loc_1d = np.where(np.ravel(Sds == 1))[0]
    for Lab in range(Fgm_num):
        Fgm_Loc_1d = np.where(np.ravel(Fgm_Lab == Lab))[0]
        if not bool((np.intersect1d(Fgm_Loc_1d, Nuc_Loc_1d)).any()):
            Fgm[Fgm_Lab == Lab] = 0

    Im_ad = (np.double(Iin) * 2 ** 16 / Iin.max()).round()
    Im_ad = nd.filters.gaussian_filter(Im_ad, .5, mode='constant')

    Im_ad_comp = np.ones(Im_ad.shape)
    Im_ad_comp = Im_ad_comp * Im_ad.max()
    Im_ad_comp = Im_ad_comp - Im_ad
    mask = ((Sds == 1).astype(np.int) + (Fgm == 0).astype(np.int))
    mask = nd.label(mask)[0]
    LabWater = mh.cwatershed(np.uint16(Im_ad_comp), mask)
    back_loc_1d = np.where(np.ravel(Fgm == 0))[0]
    for Lab in range(2, LabWater.max()):
        cell_Loc_1d = np.where(np.ravel(LabWater == Lab))[0]
        if bool((np.intersect1d(cell_Loc_1d, back_loc_1d)).any()):
            LabWater[LabWater == Lab] = 1

    return LabWater


def add_name_suffix(path, suffix):
    return path.with_name(path.stem + suffix).with_suffix(path.suffix)


def filter_coordinate(im_shape, x, y, size):
    h, w = im_shape
    return y - size > 0 and y + size < h and x - size > 0 and x + size < w


def filter_coordinates(im_shape, coordinates, size):
    """ Filter a list of coordinates, exclude items too close to border to avoid slicing exceptions

    :param im_shape: width and height of input image
    :param coordinates: list of x,y coordinates of single cells in this image
    :param size: crop size as "radius", half of one side
    :type im_shape: tuple[int, int]
    :type coordinates: list[tuple[int, int]]
    :type size: int
    :return: coordinates that are far enough from the border for cropping
    :rtype: list[tuple[int, int]]
    """
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
    :type coordinates: list[tuple[int, int]]
    :type crop_size: int
    :return: cropped cells of shape (num_cells, channels, radius*2, radius*2)
    :rtype: np.ndarray
    """
    radius = crop_size // 2
    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    # Fastest way to save arbitrarily shaped data
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
    """ Extract some basic pixel value stats

    :param im: current image
    :type im: np.ndarray
    :return: min, max, mean, std and variance for each channel
    :rtype: tuple[int, float, float, float, float, float]
    """
    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    if channels == 1:
        yield 0, np.amin(im), np.amax(im), np.mean(im), np.std(im), np.var(im)
    else:
        for idx, ch in enumerate(im):
            yield idx, np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch)


class Segmentation:
    def __init__(self, image_path, cropped_base):
        self.img = imread(str(image_path), plugin='tifffile')
        red = self.img[1]  # rfp

        print("Segmenting %s" % image_path)
        segmented, _ = segmentation.mixture_model(segmentation.blur_frame(red))

        print("Watersheding %s" % image_path)
        watershed = Watershed_MRF(red, segmented)
        labeled, c = mh.labeled.relabel(watershed)
        bboxes = mh.labeled.bbox(labeled)

        useful_cells = []  # are the ones that are far away from the border to crop

        for cell_num in range(1, c + 1):
            bbox = bboxes[cell_num]
            height = bbox[1] - bbox[0]
            width = bbox[3] - bbox[2]

            y = bbox[0] + (height // 2)
            x = bbox[1] + (width // 2)

            if filter_coordinate(red.shape, x, y, 32):
                useful_cells.append(cell_num)

        print(c, len(useful_cells))
        return

        crops_nomask = self.init_crop(cropped_base, '_nomask', c)
        crops_mask = self.init_crop(cropped_base, '_masked', c)
        crops_maskerode = self.init_crop(cropped_base, '_maskederode', c)

        for cell_num in range(1, c + 1):
            bbox = bboxes[cell_num]


            # masked = self.img.copy()
            # masked[:, labeled != cell_num] = 0
            #
            # masked2 = self.img.copy()
            # masked2[:, binary_erosion(labeled != cell_num, iterations=3, structure=disk(1))] = 0

            # gfp_nomask = make_crop(image, 1, bbox)
            # gfp_mask = make_crop(masked, 1, bbox)
            # gfp_maskblur = make_crop(masked2, 1, bbox)

        crops_nomask.flush()
        crops_mask.flush()
        crops_maskerode.flush()

    def init_crop(self, base, suffix, cells):
        return np.memmap(add_name_suffix(base, suffix), dtype=self.img.dtype, mode='w+', shape=(cells, 2, 64, 64))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--save-crop", help="Save cropped cells")
    parser.add_argument("-m", "--save-masked-crop", help="Save masked cropped cells, background is set to 0")
    parser.add_argument("-M", "--save-mask", help="Save labelled cells")
    parser.add_argument("-r", "--root-folder", help="Set base folder of images; defaults to CWD", default=os.getcwd())
    # parser.add_argument("-s", "--crop-size", help="Size of the cropped cell", default=64)
    # parser.add_argument("-f", "--multi-field-images", action='store_true', help="Images contain multiple fields")

    parser.add_argument("output_folder", help="Recreate input structure in this folder")
    parser.add_argument("images", nargs="+", help="List of input images")
    args = parser.parse_args()

    output_folder = pathlib.Path(args.output_folder).resolve()
    root_folder = pathlib.Path(args.root_folder).resolve()
    images = [pathlib.Path(i).resolve() for i in args.images]

    try:
        rel = images[0].relative_to(root_folder)
        screen_name = rel.parts[0]
    except ValueError:
        parser.error("Root folder does not appear in image paths")
        return  # Not needed, but it avoids warnings

    print("Processing %d images in screen %s" % (len(images), screen_name))

    # Generate unique hash for our image set
    image_hash = hashlib.md5()
    for image in sorted(args.images):
        image_hash.update(image.encode())
    image_hash = image_hash.hexdigest()

    print("Measurements are saved in hash %s" % image_hash)

    def mfile(suffix):
        """ Create and open a measurements file with given suffix

        :param suffix: folder suffix
        :type suffix: str
        :return: opened file for writing with csv
        """
        mf = output_folder / (screen_name + suffix) / image_hash
        mf.parent.mkdir(parents=True, exist_ok=True)
        return mf.open('w', newline='')

    with mfile('_image_measurements') as img_meas_file, mfile('_crop_measurements') as crop_meas_file:
        img_meas_writer = csv.writer(img_meas_file)
        crop_meas_writer = csv.writer(crop_meas_file)

        for image_path in args.images:
            image_path = pathlib.Path(image_path).resolve()

            cropped_base = output_folder / image_path.relative_to(args.root_folder).with_suffix('.dat')
            cropped_base.parent.mkdir(parents=True, exist_ok=True)

            print("Processing %s" % image_path)

            seg = Segmentation(image_path, cropped_base)

            # for field, cells in cells_in_fields.items():
            #     cropped_path = cropped_base
            #     field_path = image_path
            #
            #     # In multi field images the images are stacked as: f1-gfp,f1-rfp,f2-gfp,f2-rfp,...
            #     field_index = field * 2
            #
            #     if args.multi_field_images:  # Adjust crop name if multi-field
            #         cropped_path = cropped_path.with_name(
            #             cropped_path.name.replace('000.dat', '00%d.dat' % (field + 1)))
            #         field_path = field_path.with_name(
            #             field_path.name.replace('000.flex', '00%d.flex' % (field + 1)))
            #
            #     coords = filter_coordinates(img.shape[-2:], cells, args.crop_size // 2)
            #
            #     print("Cropping %d (%d excluded) cells from %s to '%s'" % (
            #         len(coords), len(cells) - len(coords), field_path, cropped_path))
            #
            #     if not coords:  # Empty image, crop would throw an exception
            #         continue
            #
            #     field_image = img[field_index:field_index + 2]
            #
            #     cropped = crop_image(field_image, cropped_path, coords, crop_size=args.crop_size)
            #
            #     cell_idx = 0
            #     for crop, crop_coordinates in zip(cropped, coords):
            #         row_common = (cropped_path.relative_to(output_folder), cell_idx) + crop_coordinates
            #         for values in get_image_measurements(crop):
            #             # noinspection PyTypeChecker
            #             crop_meas_writer.writerow(row_common + values)
            #         cell_idx += 1
            #
            #     del cropped
            #
            #     for values in get_image_measurements(field_image):
            #         # noinspection PyTypeChecker
            #         img_meas_writer.writerow((field_path.relative_to(args.root_folder),) + values)


if __name__ == '__main__':
    main()
