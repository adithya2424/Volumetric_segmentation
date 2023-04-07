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
    # below is the image with transforms applied
    input_image = np.load("dataout/00000004time20140113.npy")
    input_image_data = input_image[0]
    trinary = mask.apply(input_image_data)
    binary_mask = np.where(trinary == 0, 0, 1)
    l, w, h = input_image_data.shape
    multiplier = padding(l, 96)
    input_image_data = np.pad(input_image_data, ((0, multiplier * 96 - l), (0, 0), (0, 0)))
    sub_cube1 = input_image_data[:96, :, :]
    sub_cube2 = input_image_data[96:, :, :]

    trinary_mask1 = mask.apply(sub_cube1)
    binary_mask1 = np.where(trinary_mask1 == 0, 0, 1)
    lung_indices = np.nonzero(binary_mask1)
    min_x1, max_x1 = np.min(lung_indices[0]), np.max(lung_indices[0])
    min_y1, max_y1 = np.min(lung_indices[1]), np.max(lung_indices[1])
    min_z1, max_z1 = np.min(lung_indices[2]), np.max(lung_indices[2])
    lung_data_1 = sub_cube1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]
    binary_mask1_1 = binary_mask1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]

    l, w, h = lung_data_1.shape
    m1, m2, m3 = sub_cube1.shape
    m2_mult = padding(m2, 96)
    m3_mult = padding(m3, 96)
    sub_cube_1_main = np.pad(sub_cube1, ((0, 0), (0, m2_mult * 96 - m2), (0, m3_mult * 96 - m3)))
    patch1 = sub_cube_1_main[:, min_y1: 96 + min_y1, min_z1: 96+min_z1]
    # w_mult = padding(w, 96)
    # h_mult = padding(h, 96)
    # padded_array = np.zeros((96, w_mult * 96, h_mult * 96))
    # padded_array[:l, :w, :h] = lung_data_1
    # padded_array[l:, w:, h:] = sub_cube1

    # 192, 176, 176
    # 96, 176, 176
    # 19:95, 40:126, 17:150

    # # perform inference on lung data 1
    # # size for each inference patch is 96, 96, 96
    # # extract all patches from lung data 1
    # patch_cube1 = lung_data_1[0:96, 0:96, 0:96]
    # patch_cube2 = lung_data_1[0:96, 0:96, 96:192]

    test_transforms = Compose([
        ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                            clip=True),
        Orientation(axcodes="RAS"),
    ])

    normalised_lung_data1 = test_transforms(patch1)
    input_tensor1 = normalised_lung_data1.reshape(1, 1, 96, 96, 96)

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

    with torch.no_grad():
        input_tensor1 = input_tensor1.cuda()
        output1 = model(input_tensor1)
        post_pred_transforms = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
            Orientation(axcodes="RAS"),
        ])
        output1 = post_pred_transforms(output1[0])
        output1 = output1[0].detach().cpu().numpy()
        dummy_overlay = np.zeros_like(output1)
        multiple_clip_overlay_with_mask_from_npy(dummy_overlay, output1, "patch1test.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))

    # multiple_clip_overlay_with_mask_from_npy(input_image_data, binary_mask, "testbasecase1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(input_image_data, trinary, "testbasecase2.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(sub_cube1, trinary_mask1, "testbasecase3.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(sub_cube2, trinary_mask2, "testbasecase4.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(lung_data_1, binary_mask1_1, "testbasecase5.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(lung_data_2, binary_mask1_2, "testbasecase6.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))




if __name__ == "__main__":
    main()
