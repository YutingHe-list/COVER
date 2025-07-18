import os
from time import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import comb
from torch import nn
import torch.nn.functional as nnf
import SimpleITK as sitk

class CropTransform(object):
    def __init__(self, crop_shape):
        super(CropTransform, self).__init__()
        self.crop_shape = crop_shape

    def rand_code(self, img_shape):
        code = []
        for i in range(3):
            code.append(np.random.randint(img_shape[i] - self.crop_shape[i]))
        return code

    def augment_crop(self, image, code):
        image = image[
                :,
                :,
                code[0]:code[0]+self.crop_shape[0],
                code[1]:code[1]+self.crop_shape[1],
                code[2]:code[2]+self.crop_shape[2]
                ]
        return image

class AffineTransformer(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        norm = torch.tensor([[1, 1, 1, src.shape[2]], [1, 1, 1, src.shape[3]], [1, 1, 1, src.shape[4]]], dtype=torch.float).cuda()
        norm = norm[np.newaxis, :, :]
        mat_new = mat/norm
        grid = nnf.affine_grid(mat_new, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]])
        return nnf.grid_sample(src, grid, mode=mode)

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear', padding_mode='zeros'):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        if torch.cuda.is_available():
            grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)


class SpatialTransform3D(object):
    def __init__(self, do_rotation=True, angle_x=(-np.pi / 12, np.pi / 12), angle_y=(-np.pi / 12, np.pi / 12),
                 angle_z=(-np.pi / 12, np.pi / 12), do_scale=True, scale_x=(0.75, 1.25), scale_y=(0.75, 1.25),
                 scale_z=(0.75, 1.25), do_translate=True, trans_x=(-0.1, 0.1), trans_y=(-0.1, 0.1), trans_z=(-0.1, 0.1),
                 do_shear=True, shear_xy=(-np.pi / 18, np.pi / 18), shear_xz=(-np.pi / 18, np.pi / 18),
                 shear_yx=(-np.pi / 18, np.pi / 18), shear_yz=(-np.pi / 18, np.pi / 18),
                 shear_zx=(-np.pi / 18, np.pi / 18), shear_zy=(-np.pi / 18, np.pi / 18),
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.)):
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_translate = do_translate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.trans_z = trans_z
        self.do_shear = do_shear
        self.shear_xy = shear_xy
        self.shear_xz = shear_xz
        self.shear_yx = shear_yx
        self.shear_yz = shear_yz
        self.shear_zx = shear_zx
        self.shear_zy = shear_zy

        self.stn = SpatialTransformer()
        self.atn = AffineTransformer()

    def augment_spatial(self, data, code, mode='bilinear'):
        data = self.stn(data, code, mode=mode, padding_mode='zeros')
        return data

    def rand_coords(self, patch_size):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
        mat = np.identity(len(coords))
        if self.do_rotation:
            a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
            mat = self.rotate_mat(mat, a_x, a_y, a_z)

        if self.do_scale:
            sc_x = np.random.uniform(self.scale_x[0], self.scale_x[1])
            sc_y = np.random.uniform(self.scale_y[0], self.scale_y[1])
            sc_z = np.random.uniform(self.scale_z[0], self.scale_z[1])
            mat = self.scale_mat(mat, sc_x, sc_y, sc_z)

        if self.do_shear:
            s_xy = np.random.uniform(self.shear_xy[0], self.shear_xy[1])
            s_xz = np.random.uniform(self.shear_xz[0], self.shear_xz[1])
            s_yx = np.random.uniform(self.shear_yx[0], self.shear_yx[1])
            s_yz = np.random.uniform(self.shear_yz[0], self.shear_yz[1])
            s_zx = np.random.uniform(self.shear_zx[0], self.shear_zx[1])
            s_zy = np.random.uniform(self.shear_zy[0], self.shear_zy[1])
            mat = self.shear_mat(mat, s_xy, s_xz, s_yx, s_yz, s_zx, s_zy)

        if self.do_translate:
            t_x = np.random.uniform(self.trans_x[0] * patch_size[0], self.trans_x[1] * patch_size[0])
            t_y = np.random.uniform(self.trans_y[0] * patch_size[1], self.trans_y[1] * patch_size[1])
            t_z = np.random.uniform(self.trans_z[0] * patch_size[2], self.trans_z[1] * patch_size[2])
            mat = self.translate_mat(mat, t_x, t_y, t_z)
        else:
            mat = self.translate_mat(mat, 0, 0, 0)

        affine_coords = np.dot(coords.reshape(len(coords), -1).transpose(), mat[:, :-1]).transpose().reshape(
            coords.shape) + mat[:, -1, np.newaxis, np.newaxis, np.newaxis]
        if self.do_elastic_deform:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = self.deform_coords(affine_coords, a, s)
        else:
            coords = affine_coords

        ctr = np.asarray([patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2])
        grid = np.where(np.ones(patch_size) == 1)
        grid = np.concatenate([grid[0].reshape((1,) + patch_size), grid[1].reshape((1,) + patch_size),
                               grid[2].reshape((1,) + patch_size)], axis=0)
        grid = grid.astype(np.float32)

        coords += ctr[:, np.newaxis, np.newaxis, np.newaxis] - grid
        coords = coords.astype(np.float32)
        coords = torch.from_numpy(coords[np.newaxis, :, :, :, :]).cuda()
        mat = torch.from_numpy(mat[np.newaxis].astype(np.float32)).cuda()
        return coords, mat

    def create_zero_centered_coordinate_mesh(self, shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    def rotate_mat(self, mat, angle_x, angle_y, angle_z):
        rot_mat_x = np.stack(
            [np.stack([1, 0, 0], axis=0),
             np.stack([0, np.cos(angle_x), -np.sin(angle_x)], axis=0),
             np.stack([0, np.sin(angle_x), np.cos(angle_x)], axis=0)], axis=1)
        rot_mat_y = np.stack(
            [np.stack([np.cos(angle_y), 0, np.sin(angle_y)], axis=0),
             np.stack([0, 1, 0], axis=0),
             np.stack([-np.sin(angle_y), 0, np.cos(angle_y)], axis=0)], axis=1)
        rot_mat_z = np.stack(
            [np.stack([np.cos(angle_z), -np.sin(angle_z), 0], axis=0),
             np.stack([np.sin(angle_z), np.cos(angle_z), 0], axis=0),
             np.stack([0, 0, 1], axis=0)], axis=1)
        mat = np.matmul(rot_mat_z, np.matmul(rot_mat_y, np.matmul(rot_mat_x, mat)))
        return mat

    def deform_coords(self, coords, alpha, sigma):
        n_dim = len(coords)
        offsets = []
        for _ in range(n_dim):
            offsets.append(
                gaussian_filter((np.random.random(coords.shape[1:]) * 2 - 1), sigma, mode="constant",
                                cval=0) * alpha)
        offsets = np.array(offsets)
        indices = offsets + coords
        return indices

    def scale_mat(self, mat, scale_x, scale_y, scale_z):
        scale_mat = np.stack(
            [np.stack([scale_x, 0, 0], axis=0),
             np.stack([0, scale_y, 0], axis=0),
             np.stack([0, 0, scale_z], axis=0)], axis=1)
        mat = np.matmul(scale_mat, mat)
        return mat

    def shear_mat(self, mat, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy):
        shear_mat = np.stack(
            [np.stack([1, np.tan(shear_xy), np.tan(shear_xz)], axis=0),
             np.stack([np.tan(shear_yx), 1, np.tan(shear_yz)], axis=0),
             np.stack([np.tan(shear_zx), np.tan(shear_zy), 1], axis=0)], axis=1)
        mat = np.matmul(shear_mat, mat)
        return mat

    def translate_mat(self, mat, trans_x, trans_y, trans_z):
        trans = np.stack([trans_x, trans_y, trans_z], axis=0)
        trans = trans[:, np.newaxis]
        mat = np.concatenate([mat, trans], axis=-1)
        return mat


class AppearanceTransform(object):
    def __init__(self, do_noise=True, noise_variance=(0, 0.02), do_blur=True, sigma_range=(0, 0.05), do_contrast=True,
                 contrast_range=(0.5, 1.5), do_brightness=True, mu=0., sigma=0.1, do_inpaint=True):

        self.do_noise = do_noise
        self.noise_variance = noise_variance

        self.do_blur = do_blur
        self.sigma_range = sigma_range
        self.do_inpaint = do_inpaint

        self.do_contrast = do_contrast
        self.contrast_range = contrast_range

        self.do_brightness = do_brightness
        self.mu = mu
        self.sigma = sigma

    def rand_aug(self, data):
        if self.do_noise:
            variance = np.random.uniform(self.noise_variance[0], self.noise_variance[1])
            data = self.augment_gaussian_noise(data, variance)

        if self.do_blur:
            blur_sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            data = self.augment_gaussian_blur(data, blur_sigma)

        if self.do_contrast:
            factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
            data = self.augment_contrast(data, factor)

        if self.do_brightness:
            rnd_nb = np.random.normal(self.mu, self.sigma)
            data = self.augment_brightness_additive(data, rnd_nb)

        if self.do_inpaint:
            data = self.image_in_painting(data)

        return data

    def image_in_painting(self, x):
        B, _, img_H, img_W, img_D = x.shape
        for i in range(B):
            cnt = 10
            while cnt > 0 and np.random.random() < 0.95:
                block_noise_size_x = np.random.randint(3*img_H // 30, 4*img_H // 30)
                block_noise_size_y = np.random.randint(3*img_W // 30, 4*img_W // 30)
                block_noise_size_z = np.random.randint(3*img_D // 30, 4*img_D // 30)
                noise_x = np.random.randint(3, img_H - block_noise_size_x - 3)
                noise_y = np.random.randint(3, img_W - block_noise_size_y - 3)
                noise_z = np.random.randint(3, img_D - block_noise_size_z - 3)
                x[i, :,
                noise_x:noise_x + block_noise_size_x,
                noise_y:noise_y + block_noise_size_y,
                noise_z:noise_z + block_noise_size_z] = torch.rand(block_noise_size_x, block_noise_size_y, block_noise_size_z,).cuda() * 1.0
                cnt -= 1
        return x

    def augment_gaussian_noise(self, data, variance=0.05):
        data = data + torch.from_numpy(np.random.normal(0.0, variance, size=data.shape).astype(np.float32)).cuda()
        return data

    def augment_gaussian_blur(self, data, sigma):
        data = data.data.cpu().numpy()
        data = gaussian_filter(data, sigma, order=0)
        data = torch.from_numpy(data).cuda()
        return data

    def augment_contrast(self, data, factor):
        mn = data.mean()
        data = (data - mn) * factor + mn
        return data

    def augment_brightness_additive(self, data, rnd_nb):

        data += rnd_nb

        return data


class AppearanceTransform_Genesis(object):
    def __init__(self, local_rate=0.8, nonlinear_rate=0.9, paint_rate=0.9, inpaint_rate=0.2, is_local=True, is_nonlinear=True, is_in_painting=True):
        self.is_local = is_local
        self.is_nonlinear = is_nonlinear
        self.is_in_painting = is_in_painting
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate

        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate


    def rand_aug(self, data):
        # a = time()
        if self.is_local:
            data = self.local_pixel_shuffling(data, prob=self.local_rate)
        # b = time()
        if self.is_nonlinear:
            data = self.nonlinear_transformation(data, self.nonlinear_rate)
        # c = time()
        if self.is_in_painting:
            data = self.image_in_painting(data)
        # d = time()
        # print(d-a)
        data = data.astype(np.float32)
        return data


    def bernstein_poly(self, i, n, t):

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals


    def nonlinear_transformation(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        points = [[0, 0], [np.random.random(), np.random.random()], [np.random.random(), np.random.random()], [1, 1]]

        xvals, yvals = self.bezier_curve(points, nTimes=100000)

        xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x


    def local_pixel_shuffling(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        image_temp = x.copy()
        orig_image = x.copy()
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 5000
        # for _ in range(num_block):
        block_noise_size_x = int(img_rows // 20)
        block_noise_size_y = int(img_cols // 20)
        block_noise_size_z = int(img_deps // 20)
        noise_x = np.random.randint(low=img_rows - block_noise_size_x, size=num_block)
        noise_y = np.random.randint(low=img_cols - block_noise_size_y, size=num_block)
        noise_z = np.random.randint(low=img_deps - block_noise_size_z, size=num_block)
        window=[orig_image[:, noise_x[i]:noise_x[i] + block_noise_size_x, noise_y[i]:noise_y[i] + block_noise_size_y,
                     noise_z[i]:noise_z[i] + block_noise_size_z,] for i in range(num_block)]
        window = np.concatenate(window, axis=0)
        window = window.reshape(num_block, -1)
        # window = window.T
        np.random.shuffle(window.T)
        # window = window.T
        window = window.reshape((num_block, block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        for i in range(num_block):
            image_temp[0, noise_x[i]:noise_x[i] + block_noise_size_x,
            noise_y[i]:noise_y[i] + block_noise_size_y,
            noise_z[i]:noise_z[i] + block_noise_size_z] = window[i]
        local_shuffling_x = image_temp

        return local_shuffling_x


    def image_in_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 30
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = np.random.randint(img_rows // 10, img_rows // 5)
            block_noise_size_y = np.random.randint(img_cols // 10, img_cols // 5)
            block_noise_size_z = np.random.randint(img_deps // 10, img_deps // 5)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x


    def image_out_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = x.copy()
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
        return x



# a = SpatialTransform()
# coord = a.create_zero_centered_coordinate_mesh((4,4,4))
# k = len(coord)
# p = a.rand_coords((4, 4, 4))
# print(coord)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# #
# mirror_aug = MirrorTransform()
# spatial_aug = AppearanceTransform()
# app_aug = AppearanceTransform_Genesis()
#
# image = sitk.ReadImage('/media/E/yt/NabN/heat_junzong_import/image/1000126740.nii')
# image = sitk.GetArrayFromImage(image)

# image = np.where(image < 0., 0., image)
# image = np.where(image > 2048., 2048., image)
# image = image / 2048.
# image = image[np.newaxis, np.newaxis, :, :, :].astype(np.float32)
# image = torch.from_numpy(image)
# image = image.cuda()
# image_aug = spatial_aug.rand_aug(image)
#
# image = sitk.GetImageFromArray(image[0, 0].data.cpu().numpy())
# sitk.WriteImage(image, 'aaa.nii')
# image_aug = sitk.GetImageFromArray(image_aug[0, 0].data.cpu().numpy())
# sitk.WriteImage(image_aug, 'bbb.nii')



# # # # imageo = sitk.GetImageFromArray(image)
# # # # sitk.WriteImage(imageo, 'ooo.nii')
# image = image[np.newaxis, :, :, :].astype(np.float32)
# # image = torch.from_numpy(image)
# # # # image = image.cuda()
# # k = spatial_aug._gaussian_kernel3d(5)
# # mat = torch.from_numpy(np.identity(3, dtype=np.float32))
# # trans = torch.FloatTensor([5, 5, 5])
# # trans = trans[:, np.newaxis]
# # mat = torch.cat([mat, trans], dim=-1)
# # mat = mat[np.newaxis, :, :]
# image = spatial_aug.augment_spatial(image, mat)
# image = app_aug.rand_aug(image)
# new_image = sitk.GetImageFromArray(image[0])
# sitk.WriteImage(new_image, 'bbb.nii')
# # print(1)
# #
# # def _gaussian_kernel1d(self, sigma):
# #     sd = float(sigma)
# #     radius = int(4 * sd + 0.5)
# #     sigma2 = sigma * sigma
# #     x = np.arange(-radius, radius + 1)
# #     phi_x = np.exp(-0.5 / sigma2 * x ** 2)
# #     phi_x = phi_x / phi_x.sum()
# #
# #     return phi_x
# #
# #
# # def _gaussian_kernel3d(self, sigma):
# #     kernel1d = self._gaussian_kernel1d(sigma)[np.newaxis, :]
# #     kernel3d = np.matmul(np.matmul(kernel1d, kernel1d.T), kernel1d)
# #     return kernel3d