import json
import torchvision
import numpy as np
import os
import torch 


from torch.utils.data import Dataset
from PIL import Image
from augmentations import SobelTransform
from functools import reduce


def compute_weight_map(mask: torch.tensor):
    """
    Computes a weight map for a given mask, to balance
    amount of pixels of each kind on a mask
    Weights are computed with formula
    'w(x) = 1 + scale * (mask(x) + 2 * is_border_pixel(x))', 
    where scale is ratio multiplier
    :params:
        mask - numpy array of shape (W, H)
    :returns:
        weight_map - torch.tensor of shape (1, W, H)
    """

    scale_mult = 1
    if torch.sum(mask):
        scale_mult = reduce(lambda x, y: x * y, mask.shape) / torch.sum(mask)

    weight_map = mask.view(1, 1, mask.shape[0], mask.shape[1]).float()
    weight_map = 1 + scale_mult * weight_map 

    return weight_map.view(1, mask.shape[0], mask.shape[1])


class CigButtDataset(Dataset):
    """
    An artificial dataset for cigarette butts
    (see https://www.immersivelimit.com/datasets/cigarette-butts)
    Implement torch.utils.data.Dataset interface
    Names of the files must contain numerals, due to the indexing issues 
    """

    def __init__(
        self, root_dir: str, 
        transforms=None
    ):
        """
        :params:
            root_dir - Directory, which should contain both images and masks 
              in folders 'images' and 'masks' file respectively.
            transforms - Optional list transforms to be applied
              on image and mask 
        """
        self.root_dir     = os.path.join(os.getcwd(), root_dir)
        self.image_dir    = os.path.join(self.root_dir, 'images')
        self.mask_dir     = os.path.join(self.root_dir, 'masks')
        self.transforms   = transforms
        self.dir_content = [file for file in os.listdir(self.image_dir)]

        # required to use torchvision.transforms methods for PIL images
        self.tensor_to_image = torchvision.transforms.ToPILImage()
        self.image_to_tensor = torchvision.transforms.ToTensor()
        
    def __len__(self):
        return len(self.dir_content)  

    def __getitem__(self, idx):
        """
        Method allows to do image loading from disk
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.dir_content[idx])
        img      = Image.open(img_path).convert('RGB')
        msk_path = os.path.join(self.mask_dir, self.dir_content[idx])
        msk      = Image.open(msk_path).convert('L')

        if self.transforms:
            for tf in self.transforms:
              img, msk = tf.transform(img, msk)

        weights  = self.image_to_tensor(msk)
        weights  = compute_weight_map(weights.view(weights.shape[1], weights.shape[2]))

        return {
            'image':   self.image_to_tensor(img).float(), 
            'mask' :   self.image_to_tensor(msk).float(),
            'weights': weights.float()
        }



class SeparableCigButtDataset(CigButtDataset):
    """
     An artificial dataset for cigarette butts
    (see https://www.immersivelimit.com/datasets/cigarette-butts)
    Implement torch.utils.data.Dataset interface
    Names of the files must contain numerals, due to the indexing issues.

    Contains two subsets, separated by 'complexity' of the image.
    Complexity is estimated with the total weight of the boder pixels, 
    extracted from converted to grayscale images with Sobel's transform.
    """

    def __init__(
        self, root_dir: str, 
        transforms=None
    ):
        """
        :params:
            root_dir - Directory, which should contain both images and masks 
              in folders 'images' and 'masks' file respectively.
            transforms - Optional list transforms to be applied
              on image and mask 
        """
        super(SeparableCigButtDataset, self).__init__(root_dir, transforms)

        self.SIMPLE  = 0
        self.COMPLEX = 1

        sobel = SobelTransform()
        img_to_weight = {}

        for image in self.dir_content:

            img_path = os.path.join(self.image_dir, image)
            img      = torch.unsqueeze(self.image_to_tensor(
                    Image.open(img_path).convert('L').resize((64, 64))),0)
            
            img = img / torch.max(img)
            total_weight = torch.sum(sobel(img)  > 1)
            img_to_weight[image] = total_weight.data.numpy()

        weights = np.array([a for a in img_to_weight.values()])
        q25 = np.quantile(weights, 0.15)
        q75 = np.quantile(weights, 0.85)

        self.dir_content_simple, self.dir_content_complex = [], []
        for a in img_to_weight:
           if (img_to_weight[a] > q25) and (img_to_weight[a] < q75):
              self.dir_content_complex.append(a)
           else: self.dir_content_simple.append(a)
                   
        self.complexity  = self.SIMPLE # an indicator of current complexity
        self.dir_content = self.dir_content_simple
        
    def switch_type(self, complexity: int):
        """
        A method to switch current complexity of dataset
        """

        if complexity != self.SIMPLE and complexity != self.COMPLEX:
            raise ValueError("Invalid type argument")

        self.complexity  = complexity
        self.dir_content = self.dir_content_complex if complexity else self.dir_content_simple