import csv
import hashlib
import os
import pathlib

import mahotas as mh
import numpy as np
import scipy.ndimage as nd
import segmentation
from scipy.ndimage import binary_erosion
from skimage.io import imread, imsave
from skimage.measure import regionprops
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


def filter_coordinate(im_shape, y, x, size):
    h, w = im_shape
    return y - size > 0 and y + size < h and x - size > 0 and x + size < w


def center(bbox):
    return bbox[2] + ((bbox[3] - bbox[2]) // 2), bbox[0] + ((bbox[1] - bbox[0]) // 2)


def get_image_measurements(im):
    """ Extract some basic pixel value stats

    :param im: current image
    :type im: np.ndarray
    :return: min, max, mean, std and variance for each channel
    :rtype: tuple[int, float, float, float, float, float]
    """
    for idx, ch in enumerate(im):
        yield idx, np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch)


class Segmentation:
    def __init__(self, image_path, cropped_base, meas_writer, output_folder):
        self.image_path = image_path
        self.img = imread(str(image_path), plugin='tifffile')
        red = self.img[1]  # rfp

        print("Segmenting %s" % image_path)
        self.segmented, _ = segmentation.mixture_model(segmentation.blur_frame(red))

        print("Watersheding %s" % image_path)
        self.watershed = Watershed_MRF(red, self.segmented) - 1
        self.filter_labels()

        red_props = regionprops(self.watershed, intensity_image=self.red)
        green_props = regionprops(self.watershed, intensity_image=self.green)

        imsave(add_name_suffix(cropped_base, '_labeled').with_suffix('.tiff'), self.watershed.astype(np.int16))

        c = len(red_props)

        if not c:
            print("Skipping empty image %s" % image_path)
            return

        crops_nomask = self.init_crop(cropped_base, '_nomask', c)
        crops_mask = self.init_crop(cropped_base, '_masked', c)
        crops_maskerode = self.init_crop(cropped_base, '_maskederode', c)

        idx = 0
        for pred, pgreen in zip(red_props, green_props):
            x, y = map(int, pred.centroid)

            mask = self.watershed != pred.label
            crops_nomask[idx, :, :, :] = self.make_crop(x, y)
            crops_mask[idx, :, :, :] = self.make_crop(x, y, mask)
            crops_maskerode[idx, :, :, :] = self.make_crop(x, y, binary_erosion(mask, iterations=3, structure=disk(1)))

            row_common = (cropped_base.relative_to(output_folder), idx, x, y)
            for prop, channel in zip((pgreen, pred), (0, 1)):
                ch = crops_nomask[idx, channel, :, :]
                meas_writer.writerow(
                    row_common + (channel,
                                  np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch),
                                  prop.area) + prop.centroid + prop.bbox + (
                        prop.bbox_area, prop.eccentricity, prop.extent, prop.major_axis_length, prop.max_intensity,
                        prop.mean_intensity, prop.min_intensity, prop.minor_axis_length, prop.perimeter, prop.solidity,
                        prop.area / prop.perimeter, prop.major_axis_length / prop.minor_axis_length
                    ))
                # 'path', 'cell_index', 'row', 'column', 'channel', 'crop_min', 'crop_max', 'crop_mean', 'crop_std', 'crop_var', 'area', 'centroid_row', 'centroid_column', 'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col', 'bbox_area', 'eccentricity', 'extent', 'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length', 'perimeter', 'solidity'
            idx += 1

        crops_nomask.flush()
        crops_mask.flush()
        crops_maskerode.flush()

        del crops_nomask
        del crops_mask
        del crops_maskerode

    @property
    def green(self):
        return self.img[0]

    @property
    def red(self):
        return self.img[1]

    def filter_labels(self):
        red_props = regionprops(self.watershed, intensity_image=self.red)

        removed = 0
        for prop in red_props:
            if not filter_coordinate(self.red.shape, *prop.centroid, 32):
                self.watershed[self.watershed == prop.label] = 0
                removed += 1
        print("Removed %d bad cells" % removed)

    def make_crop(self, y, x, mask=None, size=32):
        img = self.img
        if mask is not None:
            img = self.img.copy()
            img[:, mask] = 0

        return img[:, y - size:y + size, x - size:x + size]

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

            seg = Segmentation(image_path, cropped_base, crop_meas_writer, output_folder)

            for values in get_image_measurements(seg.img):
                # noinspection PyTypeChecker
                img_meas_writer.writerow((image_path.relative_to(args.root_folder),) + values)

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