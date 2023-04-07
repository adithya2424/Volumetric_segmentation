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
    # below is the image with transforms applied
    input_image = np.load("dataout/00000004time20140113.npy")
    input_image_data = input_image[0]
    test_transforms = Compose([
        Orientation(axcodes="SPL"),
    ])
    input_data = test_transforms(input_image)
    input_data = input_data[0].numpy()
    trinary = mask.apply(input_data)
    binary_mask = np.where(trinary == 0, 0, 1)
    l, w, h = input_data.shape
    multiplier = padding(l, 96)
    input_data_2 = np.pad(input_data, ((0, multiplier * 96 - l), (0, 0), (0, 0)))
    # in general try to divide by 2 for this case since it was 192, we simply indexed!
    sub_cube1 = input_data_2[:96, :, :]
    sub_cube2 = input_data_2[96:, :, :]
    l1, w1, h1 = sub_cube1.shape
    mult_w = padding(w1, 96)
    mult_h = padding(h1, 96)
    sub_cube1 = np.pad(sub_cube1, ((0, 0), (0, mult_w * 96 - w1), (0, mult_h * 96 - h1)))

    l2, w2, h2 = sub_cube2.shape
    mult_w = padding(w2, 96)
    mult_h = padding(h2, 96)
    sub_cube2 = np.pad(sub_cube2, ((0, 0), (0, mult_w * 96 - w2), (0, mult_h * 96 - h2)))

    trinary_mask1 = mask.apply(sub_cube1)
    trinary_mask2 = mask.apply(sub_cube2)

    binary_mask1 = np.where(trinary_mask1 == 0, 0, 1)
    binary_mask2 = np.where(trinary_mask2 == 0, 0, 1)

    lung_indices = np.nonzero(binary_mask1)
    min_x1, max_x1 = np.min(lung_indices[0]), np.max(lung_indices[0])
    min_y1, max_y1 = np.min(lung_indices[1]), np.max(lung_indices[1])
    min_z1, max_z1 = np.min(lung_indices[2]), np.max(lung_indices[2])
    localised_cube1 = sub_cube1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]
    binary_mask1 = binary_mask1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]
    # convert SPL localised cube to RAS
    localised_ras_1 = np.transpose(localised_cube1)
    localised_ras_1 = np.flip(localised_ras_1, axis=0)
    localised_ras_1 = np.flip(localised_ras_1, axis=1)

    # convert SPL binary mask to RAS
    binary_mask1 = np.transpose(binary_mask1)
    binary_mask1 = np.flip(binary_mask1, axis=0)
    binary_mask1 = np.flip(binary_mask1, axis=1)

    multiple_clip_overlay_with_mask_from_npy(localised_ras_1, binary_mask1, "testlocal1.png",
                                             clip_plane="coronal", img_vrange=(-1000, 0))

    lung_indices = np.nonzero(binary_mask2)
    min_x2, max_x2 = np.min(lung_indices[0]), np.max(lung_indices[0])
    min_y2, max_y2 = np.min(lung_indices[1]), np.max(lung_indices[1])
    min_z2, max_z2 = np.min(lung_indices[2]), np.max(lung_indices[2])

    localised_cube2 = sub_cube2[min_x2:max_x2, min_y2:max_y2, min_z2:max_z2]
    l1, w1, h1 = localised_cube1.shape

    multi_l1 = padding(l1, 96)
    multi_w1 = padding(w1, 96)
    multi_h1 = padding(h1, 96)
    localised_patch1 = sub_cube1[19: 115, 40: 136, 17:192]
    localised_patch1 = np.pad(localised_patch1, ((0, 19), (0, 0), (0, 17)))

    # change localised_patch1 from spl to ras

    localised_ras = np.transpose(localised_patch1)
    localised_ras = np.flip(localised_ras, axis=0)
    localised_ras = np.flip(localised_ras, axis=1)

    patch1 = localised_ras[0:96, 0:96, 0:96]
    patch2 = localised_ras[96:192, 0:96, 0:96]

    l2, w2, h2 = localised_cube2.shape

    multi_l2 = padding(l2, 96)
    multi_w2 = padding(w2, 96)
    multi_h2 = padding(h2, 96)
    localised_patch2 = sub_cube2[0:96, 41:137, 28:192]
    localised_patch2 = np.pad(localised_patch2, ((0, 0), (0, 0), (0, 28)))
    localised_ras2 = np.transpose(localised_patch2)
    localised_ras2 = np.flip(localised_ras2, axis=0)
    localised_ras2 = np.flip(localised_ras2, axis=1)

    patch3 = localised_ras2[0:96, 0:96, 0:96]
    patch4 = localised_ras2[96:192, 0:96, 0:96]

    test_transforms = Compose([
        ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                            clip=True),
    ])

    normalised_lung_data1 = test_transforms(patch1)
    normalised_lung_data2 = test_transforms(patch2)
    normalised_lung_data3 = test_transforms(patch3)
    normalised_lung_data4 = test_transforms(patch4)
    input_tensor1 = normalised_lung_data1.reshape(1, 1, 96, 96, 96)
    input_tensor2 = normalised_lung_data2.reshape(1, 1, 96, 96, 96)
    input_tensor3 = normalised_lung_data3.reshape(1, 1, 96, 96, 96)
    input_tensor4 = normalised_lung_data4.reshape(1, 1, 96, 96, 96)
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
        input_tensor2 = input_tensor2.cuda()
        input_tensor3 = input_tensor3.cuda()
        input_tensor4 = input_tensor4.cuda()
        output1 = model(input_tensor1)
        output2 = model(input_tensor2)
        output3 = model(input_tensor3)
        output4 = model(input_tensor4)
        post_pred_transforms = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
        ])
        output1 = post_pred_transforms(output1[0])
        output2 = post_pred_transforms(output2[0])
        output3 = post_pred_transforms(output3[0])
        output4 = post_pred_transforms(output4[0])
        output1 = output1[0].detach().cpu().numpy()
        output2 = output2[0].detach().cpu().numpy()
        output3 = output3[0].detach().cpu().numpy()
        output4 = output4[0].detach().cpu().numpy()
        multiple_clip_overlay_with_mask_from_npy(patch1, output1, "patch1test.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))
        multiple_clip_overlay_with_mask_from_npy(patch2, output2, "patch2test.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))
        multiple_clip_overlay_with_mask_from_npy(patch3, output3, "patch3test.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))
        multiple_clip_overlay_with_mask_from_npy(patch4, output4, "patch4test.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))


    # localised_patch = np.pad(localised_cube1, ((0, multi_l1 * 96 - l1), (0, multi_w1 * 96 - w1), (0, multi_h1 * 96 - h1)))
    # localised_patch = sub_cube2[min_x2:max_x2, min_y2:max_y2, min_z2:max_z2]
    # binary_mask2 = binary_mask2[min_x2:max_x2, min_y2:max_y2, min_z2:max_z2]
    # # convert SPL localised cube to RAS
    # localised_ras_2 = np.transpose(localised_cube2)
    # localised_ras_2 = np.flip(localised_ras_2, axis=0)
    # localised_ras_2 = np.flip(localised_ras_2, axis=1)
    #
    # # convert SPL binary mask to RAS
    # binary_mask2 = np.transpose(binary_mask2)
    # binary_mask2 = np.flip(binary_mask2, axis=0)
    # binary_mask2 = np.flip(binary_mask2, axis=1)
    #
    # multiple_clip_overlay_with_mask_from_npy(localised_ras_2, binary_mask2, "testlocal2.png",
    #                                             clip_plane="coronal", img_vrange=(-1000, 0))
    #
    # multiple_clip_overlay_with_mask_from_npy(input_data_2, np.zeros_like(input_data_2), "testlocal_main.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    #
    # # processing for localised ras 1
    # l, w, h = localised_ras_1.shape
    # mult_h = padding(h, 96)
    # mult_l = padding(l, 96)
    # mult_w = padding(w, 96)
    # localised_ras_1_pad = np.pad(localised_ras_1, ((0, mult_l * 96 - l), (0, mult_w * 96 - w), (0, mult_h * 96 - h)))
    # # the data to pad obtained from sub cube 1 ras
    # # convert sub cube 1 to RAS
    # sub_cube1_ras = np.transpose(sub_cube1)
    # sub_cube1_ras = np.flip(sub_cube1_ras, axis=0)
    # sub_cube1_ras = np.flip(sub_cube1_ras, axis=1)
    # # np.zeros((mult_w * 96 - l), (mult_h * 96 - w), (0, 0))
    # multiple_clip_overlay_with_mask_from_npy(localised_ras_1_pad, np.zeros_like(localised_ras_1_pad), "testlocal_main_localised1.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    #
    #
    #
    #
    #
    # print("hello")




    #
    # input_data_ras = np.transpose(input_data_2)
    # input_data_ras = np.flip(input_data_ras, axis=0)
    # input_data_ras = np.flip(input_data_ras, axis=1)
    #
    # # steps to convert from SPL TO RAS
    # # test_patch = np.transpose(input_data)
    # # test_patch = np.flip(test_patch, axis=0)
    # # test_patch = np.flip(test_patch, axis=1)
    # l, w, h = sub_cube1.shape
    # multiplierw = padding(w, 96)
    # multiplierh = padding(h, 96)
    # # pad wih mode as constant
    # sub_cube_main = np.pad(sub_cube1, ((0, 0), (0, multiplierw * 96 - w), (0, multiplierh * 96 - h)), mode='constant')
    # array = np.transpose(sub_cube_main)
    # array = np.flip(array, axis=0)
    # array = np.flip(array, axis=1)
    # sub_cube_main = array
    #
    # patch1 = sub_cube_main[0:96, 0:96, 0:96]
    # # array = np.transpose(patch1)
    # # array = np.flip(array, axis=0)
    # # array = np.flip(array, axis=1)
    # # patch1 = array
    #
    # patch2 = sub_cube_main[0:96, 96:192, 0:96]
    # # array = np.transpose(patch2)
    # # array = np.flip(array, axis=0)
    # # array = np.flip(array, axis=1)
    # # patch2 = array
    #
    # patch3 = sub_cube_main[96:192, 96:192, 0:96]
    # # array = np.transpose(patch3)
    # # array = np.flip(array, axis=0)
    # # array = np.flip(array, axis=1)
    # # patch3 = array
    #
    # patch4 = sub_cube_main[96:192, 0:96, 0:96]
    # # array = np.transpose(patch4)
    # # array = np.flip(array, axis=0)
    # # array = np.flip(array, axis=1)
    # # patch4 = array
    # test_transforms = Compose([
    #     ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
    #                         clip=True),
    # ])
    # normalised_lung_data1 = test_transforms(patch1)
    # input_tensor1 = normalised_lung_data1.reshape(1, 1, 96, 96, 96)
    #
    # normalised_lung_data2 = test_transforms(patch2)
    # input_tensor2 = normalised_lung_data2.reshape(1, 1, 96, 96, 96)
    #
    # normalised_lung_data3 = test_transforms(patch3)
    # input_tensor3 = normalised_lung_data3.reshape(1, 1, 96, 96, 96)
    #
    # normalised_lung_data4 = test_transforms(patch4)
    # input_tensor4 = normalised_lung_data4.reshape(1, 1, 96, 96, 96)
    # #
    # # # initialize model environment
    # # # Setup
    # CONFIG_DIR = "configs"
    # config_id = "config_infer_TS"
    # config = load_config(f"{config_id}.yaml", CONFIG_DIR)
    #
    # data_dir = config["data_dir"]
    # model_dir = os.path.join(config["model_dir"])
    # model_path = config["pretrained"]
    # device = torch.device(config["device"])
    # # load model
    # model = unet256(6).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    #
    # with torch.no_grad():
    #     input_tensor1 = input_tensor1.cuda()
    #     input_tensor2 = input_tensor2.cuda()
    #     input_tensor3 = input_tensor3.cuda()
    #     input_tensor4 = input_tensor4.cuda()
    #     output1 = model(input_tensor1)
    #     output2 = model(input_tensor2)
    #     output3 = model(input_tensor3)
    #     output4 = model(input_tensor4)
    #     post_pred_transforms = Compose([
    #         EnsureType(),
    #         AsDiscrete(argmax=True),
    #     ])
    #     output1 = post_pred_transforms(output1[0])
    #     output2 = post_pred_transforms(output2[0])
    #     output3 = post_pred_transforms(output3[0])
    #     output4 = post_pred_transforms(output4[0])
    #     output1 = output1[0].detach().cpu().numpy()
    #     output2 = output2[0].detach().cpu().numpy()
    #     output3 = output3[0].detach().cpu().numpy()
    #     output4 = output4[0].detach().cpu().numpy()
    #
    #     final = np.zeros_like(sub_cube_main)
    #     final[0:96, 0:96, 0:96] = output1
    #     final[0:96, 96:192, 0:96] = output2
    #     final[96:192, 96:192, 0:96] = output3
    #     final[96:192, 0:96, 0:96] = output4
    #     multiple_clip_overlay_with_mask_from_npy(patch1, output1, "patch1test_old.png",
    #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    #     multiple_clip_overlay_with_mask_from_npy(patch2, output2, "patch2test_old.png",
    #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    #     multiple_clip_overlay_with_mask_from_npy(patch3, output3, "patch3test_old.png",
    #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    #     multiple_clip_overlay_with_mask_from_npy(patch4, output4, "patch4test_old.png",
    #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    #
    #     multiple_clip_overlay_with_mask_from_npy(sub_cube_main, final, "patch5test_old.png",
    #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    #
    #
    #
    #     # sub cube 2 processing
    #
    #     l, w, h = sub_cube2.shape
    #     multiplierw = padding(w, 96)
    #     multiplierh = padding(h, 96)
    #     # pad wih mode as constant
    #     sub_cube_main2 = np.pad(sub_cube2, ((0, 0), (0, multiplierw * 96 - w), (0, multiplierh * 96 - h)),
    #                            mode='constant')
    #     array = np.transpose(sub_cube_main2)
    #     array = np.flip(array, axis=0)
    #     array = np.flip(array, axis=1)
    #     sub_cube_main2 = array
    #
    #     patch1 = sub_cube_main2[0:96, 0:96, 0:96]
    #     # array = np.transpose(patch1)
    #     # array = np.flip(array, axis=0)
    #     # array = np.flip(array, axis=1)
    #     # patch1 = array
    #
    #     patch2 = sub_cube_main2[0:96, 96:192, 0:96]
    #     # array = np.transpose(patch2)
    #     # array = np.flip(array, axis=0)
    #     # array = np.flip(array, axis=1)
    #     # patch2 = array
    #
    #     patch3 = sub_cube_main2[96:192, 96:192, 0:96]
    #     # array = np.transpose(patch3)
    #     # array = np.flip(array, axis=0)
    #     # array = np.flip(array, axis=1)
    #     # patch3 = array
    #
    #     patch4 = sub_cube_main2[96:192, 0:96, 0:96]
    #     # array = np.transpose(patch4)
    #     # array = np.flip(array, axis=0)
    #     # array = np.flip(array, axis=1)
    #     # patch4 = array
    #     test_transforms = Compose([
    #         ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
    #                             clip=True),
    #     ])
    #
    #     normalised_lung_data1 = test_transforms(patch1)
    #     input_tensor1 = normalised_lung_data1.reshape(1, 1, 96, 96, 96)
    #
    #     normalised_lung_data2 = test_transforms(patch2)
    #     input_tensor2 = normalised_lung_data2.reshape(1, 1, 96, 96, 96)
    #
    #     normalised_lung_data3 = test_transforms(patch3)
    #     input_tensor3 = normalised_lung_data3.reshape(1, 1, 96, 96, 96)
    #
    #     normalised_lung_data4 = test_transforms(patch4)
    #     input_tensor4 = normalised_lung_data4.reshape(1, 1, 96, 96, 96)
    #     #
    #     # # initialize model environment
    #     # # Setup
    #     CONFIG_DIR = "configs"
    #     config_id = "config_infer_TS"
    #     config = load_config(f"{config_id}.yaml", CONFIG_DIR)
    #
    #     data_dir = config["data_dir"]
    #     model_dir = os.path.join(config["model_dir"])
    #     model_path = config["pretrained"]
    #     device = torch.device(config["device"])
    #     # load model
    #     model = unet256(6).to(device)
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     model.eval()
    #     with torch.no_grad():
    #         input_tensor1 = input_tensor1.cuda()
    #         input_tensor2 = input_tensor2.cuda()
    #         input_tensor3 = input_tensor3.cuda()
    #         input_tensor4 = input_tensor4.cuda()
    #         output1 = model(input_tensor1)
    #         output2 = model(input_tensor2)
    #         output3 = model(input_tensor3)
    #         output4 = model(input_tensor4)
    #         post_pred_transforms = Compose([
    #             EnsureType(),
    #             AsDiscrete(argmax=True),
    #         ])
    #         output1 = post_pred_transforms(output1[0])
    #         output2 = post_pred_transforms(output2[0])
    #         output3 = post_pred_transforms(output3[0])
    #         output4 = post_pred_transforms(output4[0])
    #         output1 = output1[0].detach().cpu().numpy()
    #         output2 = output2[0].detach().cpu().numpy()
    #         output3 = output3[0].detach().cpu().numpy()
    #         output4 = output4[0].detach().cpu().numpy()
    #
    #         final2 = np.zeros_like(sub_cube_main2)
    #         final2[0:96, 0:96, 0:96] = output1
    #         final2[0:96, 96:192, 0:96] = output2
    #         final2[96:192, 96:192, 0:96] = output3
    #         final2[96:192, 0:96, 0:96] = output4
    #         multiple_clip_overlay_with_mask_from_npy(patch1, output1, "2patch1test_old.png",
    #                                                  clip_plane="coronal", img_vrange=(-1000, 0))
    #         multiple_clip_overlay_with_mask_from_npy(patch2, output2, "2patch2test_old.png",
    #                                                  clip_plane="coronal", img_vrange=(-1000, 0))
    #         multiple_clip_overlay_with_mask_from_npy(patch3, output3, "2patch3test_old.png",
    #                                                  clip_plane="coronal", img_vrange=(-1000, 0))
    #         multiple_clip_overlay_with_mask_from_npy(patch4, output4, "2patch4test_old.png",
    #                                                  clip_plane="coronal", img_vrange=(-1000, 0))
    #
    #         multiple_clip_overlay_with_mask_from_npy(sub_cube_main2, final2, "2patch5test_old.png",
    #                                                  clip_plane="coronal", img_vrange=(-1000, 0))
    #
    #
    # final_big_cube = np.zeros((192, 192, 192))
    # final_big_cube[:, :, :96] = final
    # final_big_cube[:, :, 96:] = final2
    #
    # dummy_overlay = np.zeros_like(final_big_cube)
    # dummy_overlay[:, :, :96] = sub_cube_main
    # dummy_overlay[:, :, 96:] = sub_cube_main2
    #
    # multiple_clip_overlay_with_mask_from_npy(dummy_overlay, final_big_cube, "patchallfinal_coronal.png",
    #                                             clip_plane="coronal", img_vrange=(-1000, 0))
    # multiple_clip_overlay_with_mask_from_npy(dummy_overlay, final_big_cube, "patchallfinal_axial.png",
    #                                          clip_plane="axial", img_vrange=(-1000, 0))
    #
    # multiple_clip_overlay_with_mask_from_npy(dummy_overlay, final_big_cube, "patchallfinal_sagittal.png",
    #                                          clip_plane="sagittal", img_vrange=(-1000, 0))
    #
    # #     array = np.transpose(patch4)
    # #
    # #     mainf = sub_cube_main
    # #     array = np.transpose(mainf)
    # #     array = np.flip(array, axis=0)
    # #     array = np.flip(array, axis=1)
    # #     mainf = array
    # #     multiple_clip_overlay_with_mask_from_npy(mainf, final, "patch5test_old.png",
    # #                                              clip_plane="coronal", img_vrange=(-1000, 0))
    # #
    # # multiple_clip_overlay_with_mask_from_npy(input_image_data, binary_mask, "testbasecase1.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # #
    # # multiple_clip_overlay_with_mask_from_npy(sub_cube1, trinary_mask1, "testbasecase2.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # #
    # # multiple_clip_overlay_with_mask_from_npy(sub_cube2, trinary_mask2, "testbasecase3.png",
    # #                                          clip_plane="coronal", img_vrange=(-1000, 0))
    # # l, w, h = input_image_data.shape
    # # multiplier = padding(l, 96)
    # # input_image_data = np.pad(input_image_data, ((0, multiplier * 96 - l), (0, 0), (0, 0)))
    # sub_cube1 = input_image_data[:96, :, :]
    # sub_cube2 = input_image_data[96:, :, :]
    #
    # trinary_mask1 = mask.apply(sub_cube1)
    # binary_mask1 = np.where(trinary_mask1 == 0, 0, 1)
    # lung_indices = np.nonzero(binary_mask1)
    # min_x1, max_x1 = np.min(lung_indices[0]), np.max(lung_indices[0])
    # min_y1, max_y1 = np.min(lung_indices[1]), np.max(lung_indices[1])
    # min_z1, max_z1 = np.min(lung_indices[2]), np.max(lung_indices[2])
    # lung_data_1 = sub_cube1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]
    # binary_mask1_1 = binary_mask1[min_x1:max_x1, min_y1:max_y1, min_z1:max_z1]
    #
    #

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
