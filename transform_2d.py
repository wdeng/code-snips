"""
	Transforms to be applied to numpy arrays with format of (H W C)
"""
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage import exposure
import numpy as np
import torch

# https://github.com/juliandewit/kaggle_ndsb2/blob/42216c1709366e0d18597cab334ffd4efadb455d/step1_train_segmenter.py
# ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.

# def elastic_transform(image, alpha, sigma, random_state=None):
#     """Elastic deformation of images as described in [Simard2003]_.
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#        Convolutional Neural Networks applied to Visual Document Analysis", in
#        Proc. of the International Conference on Document Analysis and
#        Recognition, 2003.
#     """
#     global ELASTIC_INDICES
#     shape = image.shape

#     if ELASTIC_INDICES == None:
#         if random_state is None:
#             random_state = np.random.RandomState(1301)

#         dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#         dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#         x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
#         ELASTIC_INDICES = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
#     return map_coordinates(image, ELASTIC_INDICES, order=1, mode='reflect').reshape(shape)

# https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
# https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(image.shape)

class RandomElastic(object):
    """
    The alpha could be either tuple or int, indicating the level of random shift.
    The sigma is the smoothness of the gaussian filters, need to test before use.

    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    def __init__(self, alpha=12, sigma=16, order=1, edge_mode='constant', random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.normalize = Normalize((-1.0, 1.0))

        self.order = order
        self.mode = edge_mode

    def __call__(self, sample):
        image = sample.astype('float32')
        alpha = self.alpha
        if isinstance(alpha, (tuple, list)):
            alpha = np.random.uniform(alpha[0], alpha[1])

        shape = image.shape
        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0)
        dx = self.normalize(dx) * alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0)
        dy = self.normalize(dy) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distorted_image = map_coordinates(image, indices, order=self.order, mode=self.mode)
        return distorted_image.reshape(image.shape)

class RandomMask(object):
    def __init__(self, sigma, random_state=None, dtype='float32'):
        if random_state is None:
            random_state = np.random.RandomState(None)
        self.random_state = random_state
        self.sigma = sigma
        self.dtype = dtype

    def __call__(self, shape):
        s = shape[:2]
        if isinstance(self.sigma, (tuple, list)):
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma
        
        mask = gaussian_filter(self.random_state.rand(*s), sigma, mode='reflect')
        mask = (mask - mask.min())/(mask.max()-mask.min())
        mask = np.expand_dims(mask, axis=2)

        if len(shape)==3 and shape[2] > 1:
            mask = np.repeat(mask, shape[2], axis=2)
        if np.random.rand() > 0.5:
            mask = mask > 0.5
        else:
            mask = mask < 0.5

        return mask.astype(self.dtype)

class RandomLocalContrast(object):
    def __init__(self, sigma, constrast_range=(0.5, 1), random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        self.random_state = random_state
        self.sigma = sigma
        self.mask_scale = RandomContrast(constrast_range)

    def __call__(self, sample):
        sample = sample.astype('float32')
        shape = sample.shape[:2]

        if isinstance(self.sigma, (tuple, list)):
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma
        
        mask = gaussian_filter(self.random_state.rand(*shape), sigma, mode='reflect')
        mask = self.mask_scale((mask - mask.min())/(mask.max()-mask.min()))
        mask = np.expand_dims(mask, axis=2)

        if len(sample.shape)==3 and sample.shape[2] > 1:
            mask = np.repeat(mask, sample.shape[2], axis=2)

        return mask*sample


class PilToNumpy(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, sample):
        return np.array(sample, self.dtype)


class ToTensor(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, np_channel_first=False):
        self.ch_first = np_channel_first

    def __call__(self, sample):
        if (not self.ch_first) and sample.ndim == 3:
            sample = sample.transpose(2, 0, 1)
        sample = torch.from_numpy(sample)
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        height, width = sample.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        sample = sample[top: top + new_height, left: left + new_width, ...]

        return sample

class RandomGamma(object):
    """
    Performs Gamma Correction on the input image. 
    O = I**gamma after scaling each pixel to the range 0 to 1
    should be applied before normalization
    """
    def __init__(self, sigmas=(0.5, 1.5)):
        assert isinstance(sigmas, tuple)
        self.sigmas = sigmas
        self.norm = Normalize()

    def __call__(self, sample):
        sigma = np.random.uniform(self.sigmas[0], self.sigmas[1])
        sample = exposure.adjust_gamma(self.norm(sample), sigma)

        return sample

class RandomBlur(object):
    def __init__(self, sigmas=(0.0, 5.0)):
        assert isinstance(sigmas, tuple)
        self.sigmas = sigmas

    def __call__(self, sample):
        sigma = np.random.uniform(self.sigmas[0], self.sigmas[1])
        extra_dim = sample.ndim - 2
        if extra_dim > 0:
            sigma = [sigma, sigma] + [0.0] * extra_dim
        sample = gaussian_filter(sample, sigma)

        return sample

class RandomRotation(object):
    """Rotate randomly the image.

    Args:
        angle range, axes.
    """

    def __init__(self, angle_range=(-180, 180), reshape=False, axes=(0, 1)):
        assert isinstance(angle_range, tuple)
        assert isinstance(axes, tuple)
        self.angle_range = angle_range
        self.axes = axes
        self.reshape=reshape

    def __call__(self, sample):
        rand = np.random.uniform(self.angle_range[0], self.angle_range[1])
        sample = rotate(sample,
                        rand,
                        axes=self.axes,
                        reshape=self.reshape,
                        order=1)

        return sample


class RandomHorizontalFlip(object):
    """Flip left to right randomly the image in a sample.
    """

    def __call__(self, sample):
        if np.random.randint(0, 2):
            sample = np.fliplr(sample)

        return sample


class RandomVerticalFlip(object):
    """Rotate randomly the image in a sample.
    """

    def __call__(self, sample):
        if np.random.randint(0, 2):
            sample = np.flipud(sample)

        return sample


class Normalize(object):
    """Rotate randomly the image in a sample.
    """

    def __init__(self, intensity_range=(0, 1), ftype=np.float32):
        assert isinstance(intensity_range, tuple)
        self.range = intensity_range
        self.ftype = ftype

    def __call__(self, sample):
        sample = sample.astype(self.ftype)
        im_min, im_max = np.min(sample), np.max(sample)
        sample = ((sample - im_min) *
                  (self.range[1] - self.range[0]) / (im_max - im_min)) + self.range[0]

        return sample.astype(self.ftype)


class RandomContrast(object):
    """
        Change intensity randomly in the image
    """

    def __init__(self, contrast_range=(0.5, 1)):
        assert isinstance(contrast_range, tuple)
        self.range = contrast_range

    def __call__(self, sample):
        sample = sample.astype('float32')
        im_min, im_max = np.min(sample), np.max(sample)

        rand1 = np.random.uniform(self.range[0], self.range[1])
        rand2 = np.random.uniform(0, (im_max - im_min) * (rand1 - self.range[0]))

        sample = (sample - im_min) * rand1 + (rand2 + im_min)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        image = transform.resize(image, (new_height, new_width), mode='reflect')

        # height and width are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_width / width, new_height / height]

        return image

class RandomRescaleUpDown(object):
    """
        Randomly rescale the height of image
    """

    def __init__(self, height_range):
        assert isinstance(height_range, tuple)
        self.height_range = height_range

    def __call__(self, image):
        _, width = image.shape[:2]

        new_height = np.random.randint(self.height_range[0], self.height_range[1])
        image = transform.resize(image, (new_height, width), mode='reflect')

        return image

class RandomRescale(object):
    """
        Randomly rescale image
    """

    def __init__(self, ratio_range):
        assert isinstance(ratio_range, tuple)
        self.range = ratio_range

    def __call__(self, image):
        r = np.random.uniform(*self.range)

        height, width = int(image.shape[0]*r), int(image.shape[1]*r)
        image = transform.resize(image, (height, width), mode='reflect').astype('float32')
        return image

class CropCenter(object):
    """
        Crop the center of the ndimage
    """

    def __init__(self, new_size):
        if isinstance(new_size, int):
            new_size = (new_size, new_size)
        self.size = new_size

    def __call__(self, image):
        pad_size = image.shape[:2]
        y_size, x_size = self.size
        start_y = (pad_size[0] - y_size + 1) // 2
        start_x = (pad_size[1] - x_size + 1) // 2
        ims_new = image[start_y: start_y+y_size, start_x: start_x+x_size, ...]
        return ims_new

class PadToSize(object):
    """
        Add padding to the ndimage
    """

    def __init__(self, target_size, padding_intensity=0):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.size = target_size
        self.intensity = padding_intensity

    def __call__(self, image):
        dim_y, dim_x = image.shape[0], image.shape[1]
        assert self.size[0] >= dim_y and self.size[1] >= dim_x

        pad_y, pad_x = int((self.size[0] - dim_y)/2), int((self.size[1] - dim_x)/2)
        pad_size = [(pad_y, pad_y), (pad_x, pad_x)]

        if (self.size[0] - dim_y) % 2:
            pad_size[0] = (pad_y, pad_y + 1)
        if (self.size[1] - dim_x) % 2:
            pad_size[1] = (pad_x, pad_x + 1)

        if image.ndim == 3:
            pad_size.append((0, 0))

        return np.pad(image, pad_size,
                      'constant', constant_values=self.intensity)


class ThresholdToBinary(object):
    """
        Add padding to the ndimage
    """

    def __init__(self, percentile=0.5):
        self.percentile = percentile

    def __call__(self, image):
        im_min, im_max = image.min(), image.max()

        threshold = (im_max - im_min) * self.percentile + im_min

        return image > threshold
