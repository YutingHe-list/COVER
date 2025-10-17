import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
import torch.nn.functional as nnf


class CropTransform(object):
    def __init__(self, crop_shape):
        super(CropTransform, self).__init__()
        self.crop_shape = crop_shape

    def rand_code(self, img_shape):
        code = []
        for i in range(2):
            code.append(np.random.randint(img_shape[i] - self.crop_shape[i]))
        return code

    def augment_crop(self, image, code):
        image = image[:, :, code[0]:code[0]+self.crop_shape[0], code[1]:code[1]+self.crop_shape[1]]
        return image

class AffineTransformer2D(nn.Module):
    """
    2-D Affine Transformer
    """

    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        norm = torch.tensor([[1, 1, src.shape[2]], [1, 1, src.shape[3]]], dtype=torch.float).cuda()
        norm = norm[np.newaxis, :, :]
        mat_new = mat/norm
        grid = nnf.affine_grid(mat_new, [src.shape[0], 2, src.shape[2], src.shape[3]])
        return nnf.grid_sample(src, grid, mode=mode)

class SpatialTransformer2D(nn.Module):
    def __init__(self):
        super(SpatialTransformer2D, self).__init__()

    def forward(self, src, flow, mode='bilinear', padding_mode='zeros'):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        if torch.cuda.is_available():
            grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)


class SpatialTransform2D(object):
    def __init__(self, do_rotation=True,
                 angle_x=(-np.pi / 12, np.pi / 12),
                 angle_y=(-np.pi / 12, np.pi / 12),
                 do_scale=True,
                 scale_x=(0.75, 1.25),
                 scale_y=(0.75, 1.25),
                 do_translate=True,
                 trans_x=(-0.1, 0.1),
                 trans_y=(-0.1, 0.1),
                 do_shear=True,
                 shear_xy=(-np.pi / 18, np.pi / 18),
                 shear_yx=(-np.pi / 18, np.pi / 18),
                 do_elastic_deform=True,
                 alpha=(0., 1000.),
                 sigma=(10., 13.),
                 device='cpu'):
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.do_scale = do_scale
        self.scale_x = scale_x
        self.scale_y = scale_y

        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_translate = do_translate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.do_shear = do_shear
        self.shear_xy = shear_xy
        self.shear_yx = shear_yx

        self.stn = SpatialTransformer2D()
        self.atn = AffineTransformer2D()
        self.device = device

    def augment_spatial(self, data, code, mode='bilinear'):
        data = self.stn(data, code, mode=mode, padding_mode='zeros')
        return data

    def rand_coords(self, patch_size):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
        mat = np.identity(len(coords))
        if self.do_rotation:
            a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            mat = self.rotate_mat(mat, a_x, a_y)

        if self.do_scale:
            sc_x = np.random.uniform(self.scale_x[0], self.scale_x[1])
            sc_y = np.random.uniform(self.scale_y[0], self.scale_y[1])
            mat = self.scale_mat(mat, sc_x, sc_y)

        if self.do_shear:
            s_xy = np.random.uniform(self.shear_xy[0], self.shear_xy[1])
            s_yx = np.random.uniform(self.shear_yx[0], self.shear_yx[1])
            mat = self.shear_mat(mat, s_xy, s_yx)

        if self.do_translate:
            t_x = np.random.uniform(self.trans_x[0] * patch_size[0], self.trans_x[1] * patch_size[0])
            t_y = np.random.uniform(self.trans_y[0] * patch_size[1], self.trans_y[1] * patch_size[1])
            mat = self.translate_mat(mat, t_x, t_y)
        else:
            mat = self.translate_mat(mat, 0, 0)

        affine_coords = np.dot(coords.reshape(len(coords), -1).transpose(), mat[:, :-1]).transpose().reshape(
            coords.shape) + mat[:, -1, np.newaxis, np.newaxis]
        if self.do_elastic_deform:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = self.deform_coords(affine_coords, a, s)
        else:
            coords = affine_coords

        ctr = np.asarray([patch_size[0] // 2, patch_size[1] // 2])
        grid = np.where(np.ones(patch_size) == 1)
        grid = np.concatenate([grid[0].reshape((1,) + patch_size), grid[1].reshape((1,) + patch_size)], axis=0)
        grid = grid.astype(np.float32)

        coords += ctr[:, np.newaxis, np.newaxis] - grid
        coords = coords.astype(np.float32)
        coords = torch.from_numpy(coords[np.newaxis, :, :, :]).to(device=self.device)
        mat = torch.from_numpy(mat[np.newaxis].astype(np.float32)).to(device=self.device)
        return coords, mat

    def create_zero_centered_coordinate_mesh(self, shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    def rotate_mat(self, mat, angle_x, angle_y):
        rot_mat_x = np.stack(
            [np.stack([np.cos(angle_x), -np.sin(angle_x)], axis=0),
             np.stack([np.sin(angle_x), np.cos(angle_x)], axis=0)], axis=1)
        rot_mat_y = np.stack(
            [np.stack([np.cos(angle_y), np.sin(angle_y)], axis=0),
             np.stack([-np.sin(angle_y), np.cos(angle_y)], axis=0)], axis=1)
        mat = np.matmul(rot_mat_y, np.matmul(rot_mat_x, mat))
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

    def scale_mat(self, mat, scale_x, scale_y):
        scale_mat = np.stack(
            [np.stack([scale_x, 0], axis=0),
             np.stack([0, scale_y], axis=0)], axis=1)
        mat = np.matmul(scale_mat, mat)
        return mat

    def shear_mat(self, mat, shear_xy, shear_yx):
        shear_mat = np.stack(
            [np.stack([1, np.tan(shear_xy)], axis=0),
             np.stack([np.tan(shear_yx), 1], axis=0)], axis=1)
        mat = np.matmul(shear_mat, mat)
        return mat

    def translate_mat(self, mat, trans_x, trans_y):
        trans = np.stack([trans_x, trans_y], axis=0)
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
        B, _, img_rows, img_cols = x.shape
        for i in range(B):
            cnt = 10
            while cnt > 0 and np.random.random() < 0.95:
                block_noise_size_x = np.random.randint(3*img_rows // 30, 4*img_rows // 30)
                block_noise_size_y = np.random.randint(3*img_cols // 30, 4*img_cols // 30)
                noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
                noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
                x[i, :,
                noise_x:noise_x + block_noise_size_x,
                noise_y:noise_y + block_noise_size_y] = torch.rand(block_noise_size_x, block_noise_size_y,).to(x.device) * 1.0
                cnt -= 1
        return x

    def augment_gaussian_noise(self, data, variance=0.05):
        data = data + torch.from_numpy(np.random.normal(0.0, variance, size=data.shape).astype(np.float32)).to(data.device)
        return data

    def augment_gaussian_blur(self, data, sigma):
        data = data.data.cpu().numpy()
        data = gaussian_filter(data, sigma, order=0)
        data = torch.from_numpy(data).to(data.device)
        return data

    def augment_contrast(self, data, factor):
        mn = data.mean()
        data = (data - mn) * factor + mn
        return data

    def augment_brightness_additive(self, data, rnd_nb):

        data += rnd_nb

        return data