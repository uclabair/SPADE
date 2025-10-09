from scipy.stats import rankdata
from skimage import measure, morphology
from skimage import color
import skimage.morphology as skmp
from skimage.transform import resize
import time
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy.stats import percentileofscore
import pyvips
import matplotlib.colors as mcolors
import tifffile
import scipy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def score2percentile(scores):
    return rankdata(scores, method='max') / len(scores)


def get_location_label(loc, mask):
    for (y, x) in [
        loc,
        (loc[0] - 1, loc[1]),
        (loc[0], loc[1] - 1),
        (loc[0] + 1, loc[1]),
        (loc[0], loc[1] + 1),
    ]:
        if mask[x, y]:
            return int(mask[x, y])
        
def get_locations(slide_name, slide_path, mask_path, locations):
    locations = locations[['x', 'y']].values
    slide = openslide.OpenSlide(slide_path)
    mask = np.array(Image.open(mask_path))
    thumbnail = slide.get_thumbnail(mask.shape[::-1])
    downsample_rate = slide.level_dimensions[0][0] / mask.shape[1]
    locations_downsample = (locations / downsample_rate).astype("int")
    loc_labels = np.array(
        [get_location_label(loc, mask) for loc in locations_downsample]
    )
    return slide_name, thumbnail, mask, locations_downsample, loc_labels

def visualize_location(dataset, location_dict, slide_path, slide_name, mask_path):
    slide_name, thumbnail, mask, locations_downsample, loc_labels = get_locations(
        slide_name, slide_path, mask_path, location_dict
    )
    plt.figure(figsize=(20, 10))
    plt.title(slide_name)
    plt.subplot(1, 2, 1)
    plt.imshow(thumbnail)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="jet", alpha = 0.1)
    plt.axis("off")
    plt.scatter(locations_downsample[:, 0], locations_downsample[:, 1], color="y", s=3)
    plt.show()
    plt.clf()
    plt.close()

def average_heatmap_block(x, y, step_size, heatmap, heatmap_count, prob):
    """
    Average overlapped region prediction between patches, in-place operation for input heatmap
    Args:
        x (int): x coordinate
        y (int): y coordinate
        step_size (int): step size of a single patch
        heatmap (np.array): heatmap of predicted tumor probability (float from 0 to 1)
        heatmap_count (np.array): count map of overlapped region numbers (int)
        prob (float): predicted tumor probability
    """
    heatmap_block = heatmap[y : y + step_size, x : x + step_size]
    heatmap_count_block = heatmap_count[y : y + step_size, x : x + step_size]
    heatmap_block = heatmap_block * heatmap_count_block
    heatmap_block += prob
    heatmap_count_block += 1
    heatmap_block = heatmap_block / heatmap_count_block
    heatmap[y : y + step_size, x : x + step_size] = heatmap_block
    heatmap_count[y : y + step_size, x : x + step_size] = heatmap_count_block
    return

def load_data(slide_path, tissue_mask):
    """
    load matched image and data for visualization
    Args:
        slide_path (str): file path of whole slide image
        tissue_mask (str): file path of tissue mask
        
    Returns:
        slide (openslide image): slide image
        mask (np.array): annotation mask image
        thumbnail (PIL.Image): slide thumbnail image
        downsample_rate (float): downsample rate from the highest slide resolution to thumbnail resolution
    """

    slide = openslide.OpenSlide(slide_path)
    mask = Image.open(tissue_mask)
    thumbnail = slide.get_thumbnail(mask.size)
    downsample_rate = slide.level_dimensions[0][0] / mask.size[0]
    return (slide, mask, thumbnail, downsample_rate)

def postprocessing(
    heatmap, element_size, tumor_thred, negative_thred, min_size_ratio=0.01, plot=False
):
    """
    postprocssing for generating binarize tumor/negative map
    Args:
        heatmap (np.array): tumor probability heatmap
        element_size (int): element size of morphology operation
        tumor_thred (float): threshold for selecting tumor regions
        negative_thred (float): threshold for selecting negative regions
        min_size_ratio (float, optional): min size ration for removing small objects. Defaults to 0.01.
        plot (bool, optional): if plot visualization. Defaults to False.

    Returns:
        heatmap with post preprocessing
    """
    element = morphology.disk(element_size)
    tumor_region = heatmap > tumor_thred
    tumor_region = smooth(tumor_region, element, min_size_ratio)
    negative_region = (heatmap < negative_thred) * (heatmap > 0)
    negative_region = smooth(negative_region, element, min_size_ratio)
    heatmap_post = (tumor_region * 1) + (negative_region * 2)
    # remove overlapped regions
    heatmap_post[heatmap_post == 3] = 0
    #heatmap_post = remove_adjacent_region(heatmap_post)
    if plot:
        plt.figure()
        plt.imshow(heatmap_post, cmap="Blues")
        plt.colorbar()
        plt.show()
    return heatmap_post

def tumor_heatmap(
    slide,
    mask,
    thumbnail,
    attentions,
    locations_downsample,
    tile_size=256,
    mag=20,
    downsample_rate=100,
    plot=False,
    percentile = False
):
    """
    get tumor probability heatmap

    Args:
        slide (openslide image): slide image
        thumbnail (PIL.Image): slide thumbnail image
        attentions (np.array): predicted tumor probability, aligned with locations
        locations_downsample (np.array): tile (x, y) locations at thumbnail resolution
        tile_size (int): size of a single tile. Defaults to 256.
        mag (int): slide magnification for tile extraction. Defaults to 20.
        downsample_rate (float): downsample rate from the highest slide resolution to thumbnail resolution
    """
    max_mag = int(slide.properties["openslide.objective-power"])
    tile_rescaled_rate = max_mag / mag
    # rescale tile from extracted shape to the shape at the highest resolution
    rescaled_tile_size = int(tile_size * tile_rescaled_rate)
    # tile size at the thumbnail resolution
    tile_size_downsample = int(rescaled_tile_size / downsample_rate) + 1
    heatmap = np.zeros_like(mask).astype("float")
    heatmap_count = np.zeros_like(mask).astype("int")
    if percentile == True:
        attentions = score2percentile(attentions)
    
    for idx in range(len(attentions)):
        x, y = locations_downsample[idx]
        prob = attentions[idx]
        average_heatmap_block(x, y, tile_size_downsample, heatmap, heatmap_count, prob)

    if plot:
        plt.figure(figsize=(20, 40))
        plt.subplot(1, 3, 1)
        plt.title("Slide Thumbnail")
        plt.imshow(thumbnail)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask)
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Tumor Probability Heatmap")
        plt.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        plt.axis("off")
        plt.show()
    return heatmap

def extract_40x_coords_xy(filename, tile_size = 224, downsample = 2):
    basename = os.path.basename(filename)
    name_part = os.path.splitext(basename)[0]
    col, row = map(int, name_part.split('_')[:2])
    
    x_20x = col * tile_size
    y_20x = row * tile_size
    
    x_40x = x_20x * downsample
    y_40x = y_20x * downsample
    
    return (x_20x, y_20x), (x_40x, y_40x)
