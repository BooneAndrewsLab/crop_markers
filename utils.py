import logging
from pathlib import Path

from skimage.io import imread

log = logging.getLogger(__file__)


class Image:
    def __init__(self, path, channels=2, multifield=False):
        """

        Arguments:
            path(str): Path to image (flex or tiff)
            channels(int): Number of channels in this image
            multifield(bool): Is this image multi-field
        """
        if not isinstance(path, str):
            path = Path(path)

        log.info("Reading image %s", path)
        self.im = imread(str(path), plugin='tifffile')
        if len(self.im.shape) == 2:
            log.debug("Reshaping to 3 dimensions to have uniform input")
            self.im = self.im.reshape((1,) + self.im.shape)

        self.channels = channels
        self.multifield = multifield
        self.fields = self.im.shape[0] / channels

    def __iter__(self):
        pass
