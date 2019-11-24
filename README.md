## Pre-requisites
Dependencies are listed in requirements.txt and requirements.yml, use them like this:
```bash
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

## Scripts
### od_crop.py
Crop cells from microscopy images in various formats with provided coordinates. Coordinates are provided in a locations csv file.
Comma-delimited csv file is accepted, with following named columns required:
* path
* center_y
* center_x
* field (required when field "-f" is "multi")

Other columns are ignored and copied over to output file.
Output file has one extra column named "internal_cell_id" that is the index of a cell in the cropped file.
 
```bash
# optional: activate environment
# conda activate crop_markers
python od_crop.py -f multi /path/to/Rad52_locations.csv /path/to/screens/root/folder /path/to/output/folder
```

## Useful info
Cropped cells are saved using numpy's memmap. They can be read like this:
```python
import numpy as np

# Adjust these two parameters
channels = 2
crop_size = 64

# uint16 is the usual type for microscopy images, but it can be different for different hardware
crops = np.memmap('/path/to/crops/file.dat', dtype=np.uint16)
crops = crops.reshape((-1, channels, crop_size, crop_size))
for cell in crops:
    gfp = cell[0]  # cropped cell, first channel
    rfp = cell[1]  # cropped cell, second channel
    # Operate on crops ...
```
