import torch
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class SobelTransform:
    """
    Applies two Sobel filters on a 2d tensor with 1 color channel.
    Gives an approximation of intensity gradient, more suitable
    for CV applications.
    Filter along x is: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    Filter along y us: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] 
    """

    def __init__(self):
        self.filter_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]
        ).float()
        self.filter_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]
        ).float()

    def __call__(self, x):
        """
        :returns:
            2d tensor, each pixel of which contains l2 norm of the
            'gradient' of image in current pount
        """
        grad_x = F.conv2d(x, self.filter_x, padding=1)
        grad_y = F.conv2d(x, self.filter_y, padding=1)     
        res = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return res


class DifferentialTransform:
    """
    For each pixel computes l2 norm of numerical gradient 
    of intensity of a 2d tensor with 1 color channel, 
    which is equivalent to application of two filters
    Filter along x is: [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
    Filter along y us: [[0, 0, 0], [-1, 0, 1], [0, 0, 0]] 
    """

    def __init__(self):
        self.filter_x = torch.tensor(
            [[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]]
        ).float()
        self.filter_y = torch.tensor(
            [[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]]
        ).float() 

    def __call__(self, x):
        """
        :returns:
            2d tensor, each pixel of which contains l2 norm of the
            'gradient' of image in current pount
        """
        grad_x = F.conv2d(x, self.filter_x, padding=1)
        grad_y = F.conv2d(x, self.filter_y, padding=1)     
        res = torch.sqrt(grad_x ** 2 + grad_y ** 2) > 0
        return res        


class RandomVerticalFlip():
    """
    A class to perform randomvertical flips both on image and it's mask 
    """
    def __init__(self, p=0.5):
        """
        :params:
            p - probabily of the flip, default=0.5
        """
        self.p = [p, 1. - p]

    def transform(self, img, msk):
        """ 
        :returns:
            (img, msk) - a tuple of flipped image and mask
        """
        if np.random.choice([1, 0], p=self.p):
            img = TF.vflip(img)
            msk = TF.vflip(msk)

        return img, msk


class RandomHorizontalFlip():
    """
    A class to perform random horizontal flips on both image and it's mask 
    """
    def __init__(self, p=0.5):
        """
        :params:
            p - probabily of the flip, default=0.5
        """
        self.p = [p, 1. - p]

    def transform(self, img, msk):
        """ 
        :returns:
            (img, msk) - a tuple of flipped image and mask
        """
        if np.random.choice([1, 0], p=self.p):
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        
        return img, msk


class RandomRotation():
    """
    A class to perform random rotation transform 
    on both image and it's mask 
    """
    def __init__(self, max_angle, p=0.5):
        """
        :params:
           max angle - maximum rotation angle
        """
        self.max_angle = max_angle
        self.p = [p, 1. - p]

    def transform(self, img, msk):
        """ 
        :returns:
            (img, msk) - a tuple of rotated image and mask
        """
        if np.random.choice([1, 0], p=self.p):
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            img = TF.rotate(img, angle)
            msk = msk.rotate(angle)
        
        return img, msk


class RandomCrop():
    """
    A class to perform random crop on both image and it's mask 
    """
    def __init__(self, p=0.5):
        """
        :params:
           p - probability of the crop
        """
        self.p = [p, 1. - p]

    def transform(self, img, msk):
        """ 
        :returns:
            (img, msk) - a tuple of cropped image and mask
        """

        if np.random.choice([1, 0], p=self.p):

            w, h   = img.size

            # we want our crops to be informative, so it is essential
            # to give a symmetric prior over it's sizes

            parabolic_distr_x = lambda x: ((w - 50 - x) * (x - 50)) ** 3
            parabolic_distr_y = lambda x: ((h - 50 - x) * (x - 50)) ** 3
            parabolic_distr_w = lambda x: ((w - 100 - x) * (x - 100)) ** 3
            parabolic_distr_h = lambda x: ((h - 100 - x) * (x - 100)) ** 3
            p_x = np.array(
                [parabolic_distr_x(k) for k in np.arange(50, w - 50)])
            p_x = p_x / np.sum(p_x)
            p_y = np.array(
                [parabolic_distr_y(k) for k in np.arange(50, w - 50)])
            p_y = p_y / np.sum(p_y)

            p_w = np.array(
                [parabolic_distr_w(k) for k in np.arange(100, w - 100)])
            p_w = p_w / np.sum(p_w)
            p_h = np.array(
                [parabolic_distr_h(k) for k in np.arange(100, w - 100)])
            p_h = p_h / np.sum(p_h)


            crop_x = np.random.choice(np.arange(50, w - 50), p=p_x)
            crop_y = np.random.choice(np.arange(50, h - 50), p=p_y)
            crop_w = np.random.choice(np.arange(100, w - 100), p=p_w)
            crop_h = np.random.choice(np.arange(100, h - 100), p=p_h)
            img = TF.resized_crop(img, crop_x, crop_y, crop_w, crop_h, img.size)
            msk = TF.resized_crop(msk, crop_x, crop_y, crop_w, crop_h, img.size)
        
        return img, msk
      

