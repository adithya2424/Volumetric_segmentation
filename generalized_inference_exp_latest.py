import argparse
from data_io import DataFolder, ScanWrapper
from utils import get_logger
from paral import AbstractParallelRoutine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from utils import mkdir_p
import os
import torch
# from vis.pptx import save_pptx
import cv2 as cv
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
import nibabel as nib
# importing matplotlib package
import matplotlib.pyplot as plt
# mpl_toolkits
from mpl_toolkits import mplot3d
from surface import *
from main import load_config
from lungmask import mask
from skimage.transform import resize
from monai.transforms import (
    Compose,
    EnsureType,
    AsDiscrete,
    Spacing,
    Resize,
    Orientation,
    ScaleIntensityRange,
    InvertibleTransform,
    AddChannel
)
from models import unet256, unet64, unet64_binary
from postprocess import get_largest_cc, lungmask_filling

logger = get_logger('Clip with mask')

PIX_DIM = (1, 1, 1)
HU_WINDOW = (-1024, 600)


def _clip_image(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
    else:
        raise NotImplementedError

    return clip


def _clip_image_sitk(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 2
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 0
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset
    print("clip location is: ")
    print(clip_location, idx_dim)
    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[:, :, clip_location]
        clip = np.flip(np.flip(clip, 0), 1)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.flip(clip, 0)
    elif clip_plane == 'axial':
        clip = image_data[clip_location, :, :]
        clip = np.flip(clip, 0)
    else:
        raise NotImplementedError

    return clip


def _clip_image_RAS(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[-clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    else:
        raise NotImplementedError

    return clip


def multiple_clip_overlay_with_mask(in_nii, in_mask, out_png, clip_plane='axial', img_vrange=(-1000, 600), dim_x=4,
                                    dim_y=4):
    '''Creates overlay from input nifti file paths'''
    num_clip = dim_x * dim_y
    print(f'reading {in_nii}')
    print(f'reading {in_mask}')
    in_img = ScanWrapper(in_nii).get_data()
    in_mask_img = ScanWrapper(in_mask).get_data()

    pixdim = ScanWrapper(in_nii).get_header()['pixdim'][1:4]
    dim_physical = np.multiply(np.array(in_img.shape), pixdim).astype(int)

    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image(in_mask_img, clip_plane, num_clip, idx_clip)

        clip_in_img = cv.resize(clip_in_img, (dim_physical[0], dim_physical[1]), interpolation=cv.INTER_CUBIC)
        clip_mask_img = cv.resize(clip_mask_img, (dim_physical[0], dim_physical[1]),
                                  interpolation=cv.INTER_NEAREST)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask_img), np.max(in_mask_img)),
                                              dim_x=4,
                                              dim_y=4)


def multiple_clip_overlay_with_mask_from_npy(in_img, in_mask, out_png, clip_plane='axial', img_vrange=(-1000, 600),
                                             dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_RAS(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image_RAS(in_mask, clip_plane, num_clip, idx_clip)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask), np.max(in_mask)),
                                              dim_x=4,
                                              dim_y=4)


def multiple_clip_overlay_with_mask_from_np_sitk(in_img, in_mask, out_png, clip_plane='axial', img_vrange=(-1000, 600),
                                                 dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_sitk(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image_sitk(in_mask, clip_plane, num_clip, idx_clip)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask), np.max(in_mask)),
                                              dim_x=4,
                                              dim_y=4)


def multiple_clip_overlay_from_np_sitk(in_img, out_png, clip_plane='axial', img_vrange=(-1024, 600), dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_sitk(in_img, clip_plane, num_clip, idx_clip)
        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_in_img_list.append(clip_in_img)

    multiple_clip_overlay_from_list(clip_in_img_list,
                                    out_png,
                                    img_vrange=img_vrange,
                                    dim_x=4,
                                    dim_y=4)


def multiple_clip_overlay_from_list(clip_in_img_list, out_png, img_vrange=(-1024, 600), dim_x=4, dim_y=4):
    '''Creates x by y clip of images from input list of clipped npy images. No mask overlay'''
    in_img_row_list = []
    for idx_row in range(dim_y):
        in_img_block_list = []
        for idx_column in range(dim_x):
            in_img_block_list.append(clip_in_img_list[idx_column + dim_x * idx_row])
        in_img_row = np.concatenate(in_img_block_list, axis=1)
        in_img_row_list.append(in_img_row)

    in_img_plot = np.concatenate(in_img_row_list, axis=0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        in_img_plot,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=img_vrange[0], vmax=img_vrange[1]),
        alpha=0.8)

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def multiple_clip_overlay_with_mask_from_list(clip_in_img_list, clip_mask_img_list, out_png, img_vrange=(-1000, 600),
                                              mask_vrange=(0, 5), dim_x=4, dim_y=4):
    '''Creates overlay from input list of clipped npy images'''
    in_img_row_list = []
    mask_img_row_list = []
    for idx_row in range(dim_y):
        in_img_block_list = []
        mask_img_block_list = []
        for idx_column in range(dim_x):
            in_img_block_list.append(clip_in_img_list[idx_column + dim_x * idx_row])
            mask_img_block_list.append(clip_mask_img_list[idx_column + dim_x * idx_row])
        in_img_row = np.concatenate(in_img_block_list, axis=1)
        mask_img_row = np.concatenate(mask_img_block_list, axis=1)
        in_img_row_list.append(in_img_row)
        mask_img_row_list.append(mask_img_row)

    in_img_plot = np.concatenate(in_img_row_list, axis=0)
    mask_img_plot = np.concatenate(mask_img_row_list, axis=0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        in_img_plot,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=img_vrange[0], vmax=img_vrange[1]),
        alpha=0.8)
    ax.imshow(
        mask_img_plot,
        interpolation='none',
        cmap='jet',
        norm=colors.Normalize(vmin=mask_vrange[0], vmax=mask_vrange[1]),
        alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def clip_overlay_with_mask(in_nii, in_mask, out_png):
    # Only do the clip on axial plane.
    print(f'reading {in_nii}')
    print(f'reading {in_mask}')
    in_img = ScanWrapper(in_nii).get_data()
    in_mask_img = ScanWrapper(in_mask).get_data()
    clip_in_img = in_img[:, :, int(in_img.shape[2] / 2.0)]
    clip_in_img = np.rot90(clip_in_img)
    clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)

    clip_mask_img = in_mask_img[:, :, int(in_img.shape[2] / 2.0)]
    clip_mask_img = np.rot90(clip_mask_img)
    clip_mask_img = np.concatenate(
        [np.zeros((in_img.shape[0], in_img.shape[1]), dtype=int),
         clip_mask_img], axis=1
    )
    clip_mask_img = clip_mask_img.astype(float)

    clip_mask_img[clip_mask_img == 0] = np.nan

    vmin_img = -1200
    vmax_img = 600

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        clip_in_img,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=vmin_img, vmax=vmax_img),
        alpha=0.8)
    ax.imshow(
        clip_mask_img,
        interpolation='none',
        cmap='jet',
        norm=colors.Normalize(vmin=np.min(in_mask_img), vmax=np.max(in_mask_img)),
        alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


class ParalPPVMask(AbstractParallelRoutine):
    def __init__(self,
                 in_folder_obj,
                 out_mask_folder_obj,
                 out_clip_png_folder_obj,
                 num_process):
        super().__init__(in_folder_obj, num_process)
        self.out_mask_folder_obj = out_mask_folder_obj
        self.out_clip_png_folder_obj = out_clip_png_folder_obj

    def _run_single_scan(self, idx):
        # try:
        in_nii = self._in_data_folder.get_file_path(idx)
        out_mask_nii = self.out_mask_folder_obj.get_file_path(idx)
        out_png = self.out_clip_png_folder_obj.get_file_path(idx)

        if not os.path.exists(out_png):
            multiple_clip_overlay_with_mask(in_nii, out_mask_nii, out_png)
        else:
            logger.info(f'{out_png} already exists.')

        return out_png
        # except:
        #     print(f'Something wrong with {self._in_data_folder.get_file_path}')
        #     return []


def padding(dim, patch):
    i_dim = 0
    while i_dim * patch < dim:
        i_dim += 1
    return i_dim


def main():
    # the method used here is in the first step we slice the lung into two halves
    # we do not know the exact location of the lung, so we just slice it in the middle
    # this can be error-prone and can lead to errors in the lung segmentation
    input_image = np.load("dataout/00000006time20140122.npy")
    test_transforms = Compose([Orientation(axcodes="SPL")])
    input_data_spl = test_transforms(input_image)
    input_data_spl = input_data_spl.numpy()
    input_data_spl_trinary = mask.apply(input_data_spl[0])
    total_lung = input_data_spl[0]
    # Create a new numpy array with the same shape as the trinary mask, initialized with zeros
    left_lung = np.zeros_like(input_data_spl_trinary)
    # Set values equal to 1 in the result array
    left_lung[input_data_spl_trinary == 1] = 1
    # now we pad any of the dimension to a multiple of 96
    right_lung = np.zeros_like(input_data_spl_trinary)
    right_lung[input_data_spl_trinary == 2] = 1
    l_main, w_main, h_main = total_lung.shape
    left_lung_indices = np.nonzero(left_lung)
    min_x1, max_x1 = np.min(left_lung_indices[0]), np.max(left_lung_indices[0])
    min_y1, max_y1 = np.min(left_lung_indices[1]), np.max(left_lung_indices[1])
    min_z1, max_z1 = np.min(left_lung_indices[2]), np.max(left_lung_indices[2])
    right_lung_indices = np.nonzero(right_lung)
    min_x2, max_x2 = np.min(right_lung_indices[0]), np.max(right_lung_indices[0])
    min_y2, max_y2 = np.min(right_lung_indices[1]), np.max(right_lung_indices[1])
    min_z2, max_z2 = np.min(right_lung_indices[2]), np.max(right_lung_indices[2])
    final_left_lung = total_lung[min_x1:max_x1, min_y1:max_y1, 0:max_z1]
    final_right_lung = total_lung[min_x2:max_x2, min_y2:max_y2, max_z1:]
    final_test = total_lung[min_x1:max_x2, min_y1:max_y2, min_z2:max_z2]
    # process left lung padding
    x_pad = padding(max_x1 - min_x1, 96)
    diff_x = x_pad * 96 - (max_x1 - min_x1)
    diff_x = x_pad * 96 + min_x1
    diff_main_x = l_main - diff_x

    y_pad = padding(max_y1 - min_y1, 96)
    diff_y = y_pad * 96 - (max_y1 - min_y1)
    diff_y = y_pad * 96 + min_y1
    diff_main_y = w_main - diff_y

    case_l = 0

    if diff_main_x < 0 and diff_main_y < 0:
        case_l = 1
        left_lung_pad = np.zeros((abs(diff_main_x), abs(diff_main_y), final_left_lung.shape[2]))
        left_lung_buffer = total_lung[min_x1:, min_y1:, 0:max_z1]
        left_lung = np.concatenate((left_lung_buffer, left_lung_pad), axis=(0, 1))

    elif diff_main_x < 0 < diff_main_y:
        case_l = 2
        left_lung_pad = np.zeros((abs(diff_main_x), y_pad * 96, final_left_lung.shape[2]))
        left_lung_buffer = total_lung[min_x1:, min_y1:diff_y, 0:max_z1]
        left_lung = np.concatenate((left_lung_buffer, left_lung_pad), axis=0)

    elif diff_main_x > 0 > diff_main_y:
        case_l = 3
        left_lung_pad = np.zeros((x_pad * 96, abs(diff_main_y), final_left_lung.shape[2]))
        left_lung_buffer = total_lung[min_x1:diff_x, min_y1:, 0:max_z1]
        left_lung = np.concatenate((left_lung_buffer, left_lung_pad), axis=1)

    elif diff_main_x > 0 and diff_main_y > 0:
        case_l = 4
        left_lung_pad = np.zeros((diff_main_x, diff_main_y, final_left_lung.shape[2]))
        left_lung = total_lung[min_x1:diff_x, min_y1:diff_y, 0:max_z1]
        # left_lung = np.concatenate((left_lung_buffer, left_lung_pad), axis=(0, 1))

    # steps to pad left side of left lung
    # first calculate padding factor
    z_pad = padding(left_lung.shape[2], 96)
    # calculate the difference between the padded dimension and the actual dimension
    diff_z = z_pad * 96 - left_lung.shape[2]
    left_pad = np.zeros((left_lung.shape[0], left_lung.shape[1], diff_z))
    # obtain padded left lung across z dimension
    left_lung = np.concatenate((left_pad, left_lung), axis=2)
    multiple_clip_overlay_with_mask_from_npy(left_lung, np.zeros_like(left_lung), "leftpad1.png",
                                             clip_plane="coronal", img_vrange=(-1000, 0))

    # process right lung padding
    x_pad_2 = padding(max_x2 - min_x2, 96)
    diff_x_2 = x_pad_2 * 96 - (max_x2 - min_x2)
    diff_x_2 = x_pad_2 * 96 + min_x2
    diff_main_x_2 = l_main - diff_x_2

    y_pad_2 = padding(max_y2 - min_y2, 96)
    diff_y_2 = y_pad_2 * 96 - (max_y2 - min_y2)
    diff_y_2 = y_pad_2 * 96 + min_y2
    diff_main_y_2 = w_main - diff_y_2

    case_r = 0

    if diff_main_x_2 < 0 and diff_main_y_2 < 0:
        case_r = 1
        right_lung_pad = np.zeros((abs(diff_main_x_2), abs(diff_main_y_2), final_right_lung.shape[2]))
        right_lung_buffer = total_lung[min_x2:, min_y2:, max_z1:]
        right_lung = np.concatenate((right_lung_buffer, right_lung_pad), axis=(0, 1))

    elif diff_main_x_2 < 0 < diff_main_y_2:
        case_r = 2
        right_lung_pad = np.zeros((abs(diff_main_x_2), y_pad_2 * 96, final_right_lung.shape[2]))
        right_lung_buffer = total_lung[min_x2:, min_y2:diff_y_2, max_z1:]
        right_lung = np.concatenate((right_lung_buffer, right_lung_pad), axis=0)

    elif diff_main_x_2 > 0 > diff_main_y_2:
        case_r = 3
        right_lung_pad = np.zeros((x_pad_2 * 96, abs(diff_main_y_2), final_right_lung.shape[2]))
        right_lung_buffer = total_lung[min_x2:diff_x_2, min_y2:, max_z1:]
        right_lung = np.concatenate((right_lung_buffer, right_lung_pad), axis=1)

    elif diff_main_x_2 > 0 and diff_main_y_2 > 0:
        case_r = 4
        right_lung_pad = np.zeros((diff_main_x_2, diff_main_y_2, final_right_lung.shape[2]))
        right_lung = total_lung[min_x2:diff_x_2, min_y2:diff_y_2, max_z1:]
        # right_lung = np.concatenate((right_lung_buffer, right_lung_pad), axis=(0, 1))

    # steps to pad right side of right lung
    # first calculate padding factor
    z_pad_2 = padding(h_main - max_z1, 96)
    diff_z_2 = z_pad_2 * 96 - (h_main - max_z1)
    right_pad = np.zeros((right_lung.shape[0], right_lung.shape[1], diff_z_2))
    right_lung = np.concatenate((right_lung, right_pad), axis=2)

    multiple_clip_overlay_with_mask_from_npy(right_lung, np.zeros_like(right_lung), "rightpad1.png",
                                             clip_plane="coronal", img_vrange=(-1000, 0))

    # convert final lung one to ras
    final_lung_one_ras = np.transpose(left_lung)
    final_lung_one_ras = np.flip(final_lung_one_ras, axis=0)
    final_lung_one_ras = np.flip(final_lung_one_ras, axis=1)

    # convert final lung two to ras
    final_lung_two_ras = np.transpose(right_lung)
    final_lung_two_ras = np.flip(final_lung_two_ras, axis=0)
    final_lung_two_ras = np.flip(final_lung_two_ras, axis=1)

    # initialize model environment
    # Setup
    CONFIG_DIR = "configs"
    config_id = "config_infer_TS"
    config = load_config(f"{config_id}.yaml", CONFIG_DIR)

    data_dir = config["data_dir"]
    model_dir = os.path.join(config["model_dir"])
    model_path = config["pretrained"]
    device = torch.device(config["device"])
    # load model
    model = unet256(6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_transforms = Compose([
        ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                            clip=True),
    ])

    normalised_lung_data1 = test_transforms(final_lung_one_ras)
    normalised_lung_data2 = test_transforms(final_lung_two_ras)

    final_labels = np.zeros_like(normalised_lung_data1)
    # inference on final_lung_one_pad
    x, y, z = np.shape(final_lung_one_ras)
    num_cubes_x = x // 96
    num_cubes_y = y // 96
    num_cubes_z = z // 96

    with torch.no_grad():
        for i in range(num_cubes_x):
            for j in range(num_cubes_y):
                for k in range(num_cubes_z):
                    cube = normalised_lung_data1[i * 96:(i + 1) * 96, j * 96:(j + 1) * 96, k * 96:(k + 1) * 96]
                    cube = cube.reshape(1, 1, 96, 96, 96)
                    input_tensor = cube.cuda()
                    output = model(input_tensor)
                    post_pred_transforms = Compose([
                        EnsureType(),
                        AsDiscrete(argmax=True),
                    ])
                    output = post_pred_transforms(output[0])
                    output = output[0].detach().cpu().numpy()
                    # do inference on cube
                    final_labels[i * 96:(i + 1) * 96, j * 96:(j + 1) * 96, k * 96:(k + 1) * 96] = output

    multiple_clip_overlay_with_mask_from_npy(final_lung_one_ras, final_labels, "cube1labeltest_c_1.png",
                                             clip_plane="coronal", img_vrange=(-1000, 0))
    multiple_clip_overlay_with_mask_from_npy(final_lung_one_ras, final_labels, "cube1labeltest_a_1.png",
                                             clip_plane="axial", img_vrange=(-1000, 0))
    multiple_clip_overlay_with_mask_from_npy(final_lung_one_ras, final_labels, "cube1labeltest_s_1.png",
                                             clip_plane="sagittal", img_vrange=(-1000, 0))

    # Convert final lung two back to SPL
    final_labels_spl_1 = np.flip(final_labels, axis=1)
    final_labels_spl_1 = np.flip(final_labels_spl_1, axis=0)
    final_labels_spl_1 = np.transpose(final_labels_spl_1)

    final_labels = np.zeros_like(normalised_lung_data2)
    # inference on final_lung_one_pad
    x, y, z = np.shape(final_lung_two_ras)
    num_cubes_x = x // 96
    num_cubes_y = y // 96
    num_cubes_z = z // 96

    with torch.no_grad():
        for i in range(num_cubes_x):
            for j in range(num_cubes_y):
                for k in range(num_cubes_z):
                    cube = normalised_lung_data2[i * 96:(i + 1) * 96, j * 96:(j + 1) * 96, k * 96:(k + 1) * 96]
                    cube = cube.reshape(1, 1, 96, 96, 96)
                    input_tensor = cube.cuda()
                    output = model(input_tensor)
                    post_pred_transforms = Compose([
                        EnsureType(),
                        AsDiscrete(argmax=True),
                    ])
                    output = post_pred_transforms(output[0])
                    output = output[0].detach().cpu().numpy()
                    # do inference on cube
                    final_labels[i * 96:(i + 1) * 96, j * 96:(j + 1) * 96, k * 96:(k + 1) * 96] = output

    multiple_clip_overlay_with_mask_from_npy(final_lung_two_ras, final_labels, "cube2labeltest_c_2.png",
                                             clip_plane="coronal", img_vrange=(-1000, 0))
    multiple_clip_overlay_with_mask_from_npy(final_lung_two_ras, final_labels, "cube2labeltest_a_2.png",
                                             clip_plane="axial", img_vrange=(-1000, 0))
    multiple_clip_overlay_with_mask_from_npy(final_lung_two_ras, final_labels, "cube2labeltest_s_2.png",
                                             clip_plane="sagittal", img_vrange=(-1000, 0))

    # Convert final lung two back to SPL
    final_labels_spl_2 = np.flip(final_labels, axis=1)
    final_labels_spl_2 = np.flip(final_labels_spl_2, axis=0)
    final_labels_spl_2 = np.transpose(final_labels_spl_2)

    total_labels = np.zeros_like(total_lung)

    total_left = final_labels_spl_1[0:max_x1, 0:max_y1, 0:max_z1]

    multiple_clip_overlay_with_mask_from_npy(total_lung, total_left, "leftTlung.png",
                                              clip_plane="coronal", img_vrange=(-1000, 0))




    #
    # print("checkpoint")
    #
    # # multiple_clip_overlay_with_mask_from_npy(input_data_spl[0], left_lung, "leftlung.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # # multiple_clip_overlay_with_mask_from_npy(input_data_spl[0], right_lung, "rightlung.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # #
    # multiple_clip_overlay_with_mask_from_npy(final_left_lung, np.zeros_like(final_left_lung), "leftlung1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(left_lung, np.zeros_like(left_lung), "leftpad1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    #
    # multiple_clip_overlay_with_mask_from_npy(final_right_lung, np.zeros_like(final_right_lung), "rightlung1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    #
    # multiple_clip_overlay_with_mask_from_npy(right_lung, np.zeros_like(right_lung), "rightpad1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # # multiple_clip_overlay_with_mask_from_npy(final_test, np.zeros_like(final_test), "finaltest.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    #
    # # multiple_clip_overlay_with_mask_from_npy(final_lung_one_pad, np.zeros_like(final_lung_one_pad), "leftlung2.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # # multiple_clip_overlay_with_mask_from_npy(final_lung_two_pad, np.zeros_like(final_lung_two_pad), "rightlung2.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))


if __name__ == "__main__":
    main()
