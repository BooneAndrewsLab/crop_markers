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
from skimage.exposure import adjust_sigmoid, adjust_gamma, equalize_adapthist
from skimage.morphology import disk


# noinspection PyPep8Naming,PyUnresolvedReferences,PyArgumentList
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
    """
    :param path:
    :param suffix:
    :return: Ammended path
    :rtype: pathlib.Path
    """
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
    def __init__(self, image_path, cropped_base, meas_writer, output_folder, crop_sizes=(64,), ext_label=None,
                 save_measurements=True, no_crop=False, adjust_contrast=False, border_remove=False):
        self.image_path = image_path
        self.img = imread(str(image_path), plugin='tifffile')
        self.watershed = None
        red = self.img[1]  # rfp

        if adjust_contrast:
            print("Adjusting RFP contrast %s" % image_path)
            red_seg = adjust_sigmoid(red, cutoff=0.25)
        else:
            red_seg = red

        labeled_path = add_name_suffix(cropped_base, '_labeled').with_suffix('.tiff')

        if ext_label:
            ext_labeled_path = add_name_suffix(ext_label, '_labeled').with_suffix('.tiff')
            if ext_labeled_path.exists():
                print("Found existing segmented watershed %s" % ext_labeled_path)
                self.watershed = imread(str(ext_labeled_path), plugin='tifffile')

        if self.watershed is None and labeled_path.exists():  # try normal location
            print("Found existing segmented watershed %s" % labeled_path)
            try:
                self.watershed = imread(str(labeled_path), plugin='tifffile')
            except ValueError:
                # Bad tiff, previous run probably failed while saving it
                print("Ignoring bad watershed image")

        if self.watershed is None:
            print("Segmenting %s" % image_path)
            segmented, _ = segmentation.mixture_model(red_seg) # segmented, _ = segmentation.mixture_model(segmentation.blur_frame(red))

            print("Watersheding %s" % image_path)
            self.watershed = Watershed_MRF(red_seg, segmented) - 1

            if border_remove:
                print("Removing cells touching the border")
                self.watershed = mh.labeled.remove_bordering(self.watershed)

            imsave(str(labeled_path), self.watershed.astype(np.int16))

        watershed_copy = self.watershed.copy()

        if no_crop:
            return

        for crop_size in crop_sizes:
            self.half_crop_size = crop_size // 2
            self.crop_size = crop_size

            self.watershed = watershed_copy.copy()
            print('Processing %dx%d crops' % (crop_size, crop_size))
            self.filter_labels()  # Will fail badly if reducing crop size

            red_props = regionprops(self.watershed, intensity_image=self.red)
            green_props = regionprops(self.watershed, intensity_image=self.green)

            c = len(red_props)

            if not c:
                print("Skipping empty image %s" % image_path)
                continue

            crops_nomask = self.init_crop(cropped_base, '_crop%d_nomask' % crop_size, c)
            crops_mask = self.init_crop(cropped_base, '_crop%d_masked' % crop_size, c)
            crops_maskerode = self.init_crop(cropped_base, '_crop%d_maskederode' % crop_size, c)

            idx = 0
            for pred, pgreen in zip(red_props, green_props):
                x, y = map(int, pred.centroid)

                mask = self.watershed != pred.label
                crops_nomask[idx, :, :, :] = self.make_crop(x, y)
                crops_mask[idx, :, :, :] = self.make_crop(x, y, mask)
                crops_maskerode[idx, :, :, :] = self.make_crop(x, y,
                                                               binary_erosion(mask, iterations=3, structure=disk(1)))

                if save_measurements:
                    row_common = (cropped_base.relative_to(output_folder), idx, x, y, crop_size)
                    for prop, channel in zip((pgreen, pred), (0, 1)):
                        ch = crops_nomask[idx, :, :, channel]
                        meas_writer.writerow(
                            row_common + (channel,
                                          np.amin(ch), np.amax(ch), np.mean(ch), np.std(ch), np.var(ch),
                                          prop.area) + prop.centroid + prop.bbox + (
                                prop.bbox_area, prop.eccentricity, prop.extent, prop.major_axis_length,
                                prop.max_intensity,
                                prop.mean_intensity, prop.min_intensity, prop.minor_axis_length, prop.perimeter,
                                prop.solidity,
                                prop.area / prop.perimeter, prop.major_axis_length / prop.minor_axis_length
                            ))
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
            if not filter_coordinate(self.red.shape, *prop.centroid, self.half_crop_size):
                self.watershed[self.watershed == prop.label] = 0
                removed += 1
        print("Removed %d bad cells" % removed)

    def make_crop(self, y, x, mask=None):
        size = self.half_crop_size
        img = self.img
        if mask is not None:
            img = self.img.copy()
            img[:, mask] = 0

        crop = img[:, y - size:y + size, x - size:x + size]

        return np.moveaxis(crop, 0, -1)

    def init_crop(self, base, suffix, cells):
        return np.memmap(add_name_suffix(base, suffix), dtype=self.img.dtype, mode='w+',
                         shape=(cells, self.crop_size, self.crop_size, 2))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--save-crop", help="Save cropped cells")
    parser.add_argument("-m", "--save-masked-crop", help="Save masked cropped cells, background is set to 0")
    parser.add_argument("-M", "--save-mask", help="Save labelled cells")
    parser.add_argument("-r", "--root-folder", help="Set base folder of images; defaults to CWD", default=os.getcwd())
    parser.add_argument("-s", "--crop-sizes", help="Comma delimited sizes of the cropped cells", default='64')
    parser.add_argument("--no-crop", action='store_true', help="Don't crop")
    # parser.add_argument("-f", "--multi-field-images", action='store_true', help="Images contain multiple fields")
    parser.add_argument("-l", "--label-path", help="Use existing labeled images")
    parser.add_argument("-n", "--no-measurements", dest='measure', help="Don't save measurements", action='store_false')
    parser.add_argument("-a", "--adjust-contrast", action='store_true', help="Adjust image contrast of nuclear channel "
                                                                             "prior to segmentation by performing "
                                                                             "Sigmoid Correction")
    parser.add_argument("-b", "--border-remove", action='store_true', help="Filter out cells touching the border")
    parser.add_argument("output_folder", help="Recreate input structure in this folder")
    parser.add_argument("images", nargs="+", help="List of input images")
    args = parser.parse_args()

    crop_sizes = [int(s) for s in args.crop_sizes.split(',')]

    images = []
    for im in args.images:
        im = pathlib.Path(im).resolve()
        if im.is_dir():
            images.extend(im.rglob('*.flex'))
            images.extend(im.rglob('*.tiff'))
        else:
            images.append(im)

    print("Found %d images" % len(images))

    output_folder = pathlib.Path(args.output_folder).resolve()
    root_folder = pathlib.Path(args.root_folder).resolve()

    label_path = None
    if args.label_path:
        label_path = pathlib.Path(args.label_path).resolve()

    try:
        rel = images[0].relative_to(root_folder)
        screen_name = rel.parts[0]
    except ValueError:
        parser.error("Root folder does not appear in image paths")
        return  # Not needed, but it avoids warnings

    print("Processing %d images in screen %s" % (len(images), screen_name))

    # Generate unique hash for our image set
    image_hash = hashlib.md5()
    for image in sorted(images):
        image_hash.update(str(image).encode())
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

        for image_path in images:
            cropped_base = output_folder / image_path.relative_to(args.root_folder).with_suffix('.dat')
            cropped_base.parent.mkdir(parents=True, exist_ok=True)

            ext_labels = None
            if label_path:
                ext_labels = label_path / image_path.relative_to(args.root_folder).with_suffix('.dat')

            print("Processing %s" % image_path)

            seg = Segmentation(image_path, cropped_base, crop_meas_writer, output_folder, crop_sizes, ext_labels,
                               save_measurements=args.measure, no_crop=args.no_crop,
                               adjust_contrast=args.adjust_contrast, border_remove=args.border_remove)

            for values in get_image_measurements(seg.img):
                # noinspection PyTypeChecker
                img_meas_writer.writerow((image_path.relative_to(args.root_folder),) + values)


if __name__ == '__main__':
    main()
