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
from models import unet256
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


def make_cube_isotropic(cube):
    # determine maximum dimension
    max_dim = max(cube.shape)

    # create new array of zeros with max_dim size for all dimensions
    padded_cube = np.zeros((max_dim, max_dim, max_dim))

    # calculate amount of padding for each dimension
    pad_x = (max_dim - cube.shape[0]) // 2
    pad_y = (max_dim - cube.shape[1]) // 2
    pad_z = (max_dim - cube.shape[2]) // 2

    # pad original array with zeros on all sides
    padded_cube[pad_x:pad_x + cube.shape[0], pad_y:pad_y + cube.shape[1], pad_z:pad_z + cube.shape[2]] = cube

    # add extra row, column or slice to each dimension if needed
    if max_dim % 2 == 1:
        padded_cube = np.concatenate((padded_cube, np.zeros((max_dim, max_dim, 1))), axis=2)
    if max_dim % 2 == 1:
        padded_cube = np.concatenate((padded_cube, np.zeros((max_dim, 1, padded_cube.shape[2]))), axis=1)
    if padded_cube.shape[0] % 2 == 1:
        padded_cube = np.concatenate((padded_cube, np.zeros((1, padded_cube.shape[1], padded_cube.shape[2]))), axis=0)

    return padded_cube

def main():
    # below is the image with transforms applied
    input_image = np.load("/home/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/test_data_out_dir/00000020time20140322.npy")
    input_image_data = input_image[0]
    #
    # # below is the label data
    # label_data = np.load("label_out_dir/00000004time20140113.npy")
    # label_data = label_data[0]

    # below is the transformed binary mask of the input image
    mask_image = np.load("/home/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/test_mask_out_dir/00000020time20140322.npy")
    mask_image_data = mask_image[0]

    # convert mask to binary mask
    binary_mask = np.where(mask_image_data == 0, 0, 1)

    # obtain the localised lung data

    # step 1: obtain the indices where the lung is present
    lung_indices = np.nonzero(binary_mask)

    # step 2: Find the minimum and maximum indices along each axis
    min_x, max_x = np.min(lung_indices[0]), np.max(lung_indices[0])
    min_y, max_y = np.min(lung_indices[1]), np.max(lung_indices[1])
    min_z, max_z = np.min(lung_indices[2]), np.max(lung_indices[2])

    # step 3: use min_x, min_y, min_z to obtain the bounding box of the lung
    lung_data = input_image_data[min_x:max_x, min_y:max_y, min_z:max_z]

    # step 4: create a dummy overlay
    dummy_overlay = np.zeros(lung_data.shape)


    # step 5: visualize the lung_data variable
    # multiple_clip_overlay_with_mask_from_npy(lung_data, dummy_overlay, "coronal_latest_with_bounding_box.png",
    #                                          clip_plane="coronal", img_vrange=(-1000, 0))

    # step 6: normalize the lung data
    test_transforms = Compose([
        ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                            clip=True),
    ])
    normalised_lung_data = test_transforms(lung_data)
    normalised_lung_data = normalised_lung_data.detach().cpu().numpy()
    print("hello")

    # step 9: numpy logical comparison with boxes and normalised_lung_data variable
    big_lung_cube = normalised_lung_data
    l, w, h = big_lung_cube.shape
    l, w = l // 2, w // 2
    # Slice out four sub-cubes of size (5, 5, 5) from the big cube
    sub_cube1 = big_lung_cube[:l, :w, :h]  # top left corner
    sub_cube2 = big_lung_cube[:l, w:, :h]  # top right corner
    sub_cube3 = big_lung_cube[l:, :w, :h]  # bottom left corner
    sub_cube4 = big_lung_cube[l:, w:, :h]  # bottom right corner

    # resize each sub cube to a shape of (96, 96, 96) using resize trilinear monai transform
    raw_transforms = resize(sub_cube1, (96, 96, 96), order=1,  mode='edge', anti_aliasing=False)
    # change shape from 96, 96, 96 to 1, 96, 96, 96
    resize_transform = raw_transforms.reshape((1, 1, 96, 96, 96))
    input_tensor1 = torch.from_numpy(resize_transform)

    raw_transforms = resize(sub_cube2, (96, 96, 96), order=1,  mode='edge', anti_aliasing=True)
    # change shape from 96, 96, 96 to 1, 96, 96, 96
    resize_transform = raw_transforms.reshape((1, 1, 96, 96, 96))
    input_tensor2 = torch.from_numpy(resize_transform)

    raw_transforms = resize(sub_cube3, (96, 96, 96), order=1,  mode='edge', anti_aliasing=True)
    # change shape from 96, 96, 96 to 1, 96, 96, 96
    resize_transform = raw_transforms.reshape((1, 1, 96, 96, 96))
    input_tensor3 = torch.from_numpy(resize_transform)

    raw_transforms = resize(sub_cube4, (96, 96, 96), order=1,  mode='edge', anti_aliasing=True)
    # change shape from 96, 96, 96 to 1, 96, 96, 96
    resize_transform = raw_transforms.reshape((1, 1, 96, 96, 96))
    input_tensor4 = torch.from_numpy(resize_transform)

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
        raw_nii = nib.load("/home/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/test_data_dir/00000020time20140322.nii.gz")
        axcodes = nib.orientations.aff2axcodes(raw_nii.affine)
        axcodes = ''.join(axcodes)
        pixdim = raw_nii.header.get_zooms()
        spatial_size = raw_nii.shape

        post_pred_transforms1 = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=pixdim, mode="nearest"),
            Resize(spatial_size=sub_cube1.shape, mode="nearest"),
        ])
        post_pred_transforms2 = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=pixdim, mode="nearest"),
            Resize(spatial_size=sub_cube2.shape, mode="nearest"),
        ])
        post_pred_transforms3 = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=pixdim, mode="nearest"),
            Resize(spatial_size=sub_cube3.shape, mode="nearest"),
        ])
        post_pred_transforms4 = Compose([
            EnsureType(),
            AsDiscrete(argmax=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=pixdim, mode="nearest"),
            Resize(spatial_size=sub_cube4.shape, mode="nearest"),
        ])

        pred1 = post_pred_transforms1(output1[0])
        label_map1 = pred1[0].detach().cpu().numpy()
        pred2 = post_pred_transforms2(output2[0])
        label_map2 = pred2[0].detach().cpu().numpy()
        pred3 = post_pred_transforms3(output3[0])
        label_map3 = pred3[0].detach().cpu().numpy()
        pred4 = post_pred_transforms4(output4[0])
        label_map4 = pred4[0].detach().cpu().numpy()

        print("stage 1")
        final_stitched_cube = np.zeros_like(big_lung_cube)
        # obtain the final cube
        final_stitched_cube[:l, :w, :h] = label_map1
        final_stitched_cube[:l, w:, :h] = label_map2
        final_stitched_cube[l:, :w, :h] = label_map3
        final_stitched_cube[l:, w:, :h] = label_map4
        image_path = "/home/adi/scripts/pycharm_project_lobeseg/lobe_seg_downsampled/test_data_dir/00000020time20140322.nii.gz"

        final_label_data = np.zeros_like(input_image_data)
        # Assign the processed lung data back into the original input image data
        final_label_data[min_x:max_x, min_y:max_y, min_z:max_z] = final_stitched_cube

        label_map_final = resize(final_label_data, spatial_size, order=1,  mode='edge', anti_aliasing=True)

        label_map_filled = lungmask_filling(get_largest_cc(label_map_final), image_path)

        # print(np.amax(label_map_filled))
        # # step 10: visualize the final_stitched_cube variable
        multiple_clip_overlay_with_mask_from_npy(raw_nii.get_fdata(), label_map_filled, "coronal_latest_with_bounding_box_stitched_predicted_test_2.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))

        multiple_clip_overlay_with_mask_from_npy(input_image_data, final_label_data,
                                                 "coronal_latest_with_bounding_box_stitched_predicted_test_transformed_2.png",
                                                 clip_plane="coronal", img_vrange=(-1000, 0))

    # perform inference on the boxes
    # step 1: load the model
    # step 8: boxes
    print("boxes ready")

if __name__ == "__main__":
    main()
