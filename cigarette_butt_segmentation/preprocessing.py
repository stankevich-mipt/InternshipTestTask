import json
import os
import augmentations
import torch
import numpy as np

from functools import reduce
from PIL import Image
from lib.utils import get_mask
from torchvision.transforms import ToPILImage, ToTensor


def add_cigg_butt(
	img, mask, img_dir, mask_dir,  
	img_dir_content, dataset_size, 
	annotations, p=0.1
	):
	"""
	A helper function to create additional cigarette butt on an image
	:params:
		img      - tensor, corresponding to base image
		mask     - base image mask
		img_dir  - directory with dataset images
		mask_dir - directoy with dataset masks
		img_dir_content - list of all files in image directory
		dataset_size - initial dataset size
		annotations - .json COCO annotation file
		p - probability of activation, default=0.1
		
	"""

	if np.random.choice([1, 0], p=[1. - p, p]): return

	tensor_to_image = ToPILImage()
	image_to_tensor = ToTensor()

	start_img_num = np.random.randint(0, dataset_size - 200)
	mask_size     = np.sum(mask)

	for i in range(start_img_num, dataset_size): 

		idx_str_add  = img_dir_content[i].lstrip("0").split(".")[0]
		add_img_idx  = 0 if idx_str_add == '' else int(idx_str_add)
		if add_img_idx > dataset_size: return

		add_img_path = os.path.join(img_dir, img_dir_content[i])
		add_img      = image_to_tensor(Image.open(add_img_path).convert("RGB"))
		add_img_mask = get_mask(add_img_idx, annotations)
		add_mask_size = np.sum(add_img_mask)

		# check for intersection and suitable size
		
		if not (add_img_mask & mask).sum(): 
			if  0.7 * mask_size <= add_mask_size <= 1.2 * mask_size: 
				
				new_mask = add_img_mask[..., np.newaxis] + mask

				tmp = torch.from_numpy(
					add_img_mask[np.newaxis, ...]
				).expand(3, -1, -1) == 0
				tmp = tmp.bool()

				img = image_to_tensor(img) 
				img = torch.where(tmp, img, add_img)
				img = tensor_to_image(img)
				img.save(
					os.path.join(img_dir, str(
						add_img_idx + len(img_dir_content)) + '.jpg'))

				new_mask = tensor_to_image(new_mask)
				new_mask.save(
					os.path.join(mask_dir, str(
						add_img_idx + len(img_dir_content)) + '.jpg'))
				
				return


def extend_dataset(root_dir: str, transforms=None, add_new_cig_butts=False):
	"""
	Extends base dataset with transformed images and masks.
	Mask are saved in .jpg format in the 'masks' directory
	Weighted masks are saved in .jpg format in the 'weighted_masks' directory
	At the same time computes channel-wise mean values for all images in 
	dataset.
	:params:
            root_dir - Directory, which should contain both images and 
              annotations in folder 'images' and 'coco_annotations.json' file 
              respectively.
            transforms - Optional transforms to be applied
              on image and mask  
            add_new_butts - if True, randomly creates an additional image with
            two cigarette butts, sampled from different images
    :returns:
    	(IMG_MEAN_R, IMG_MEAN_G, IMG_MEAN_B) - tuple of floats
	"""

	# For renaming, I will multiply current index of image 
        # by len(transforms) + 1

	scale = (len(transforms) + 1) if transforms else 1
	root_dir    = os.path.join(os.getcwd(), root_dir)
	image_dir   = os.path.join(root_dir, 'images')

	inital_size  = len(os.listdir(image_dir))
	annotations = json.load(
    	open(os.path.join(root_dir, 'coco_annotations.json'), 'r'))
	tensor_to_image = ToPILImage()
	image_to_tensor = ToTensor()

	# directory for masks
	mask_dir = os.path.join(root_dir, 'masks')
	try:
		os.mkdir(mask_dir)
	except Exception as err:
		pass

	IMG_MEAN_R, IMG_MEAN_G, IMG_MEAN_B = 0.0, 0.0, 0.0


	for image in os.listdir(image_dir):
		
		img_path = os.path.join(image_dir, image)
		img      = Image.open(img_path).convert("RGB")
		try:
			os.remove(img_path)
		except OSError as err:
			raise err # just rethrow

		idx_str  = image.lstrip("0").split(".")[0]
		img_idx  = 0 if idx_str == '' else int(idx_str)
		img.save(os.path.join(image_dir, str(scale * img_idx) + '.jpg'))
		
		# compute channel-wise mean values for images on the run
		img_t = image_to_tensor(img)
		IMG_MEAN_R += torch.mean(img_t[0])
		IMG_MEAN_G += torch.mean(img_t[1])
		IMG_MEAN_B += torch.mean(img_t[2])

		mask     = get_mask(img_idx, annotations)[..., np.newaxis ]
		imask    = tensor_to_image(mask)
		imask.save(os.path.join(mask_dir, str(scale * img_idx) + '.jpg'))

		if add_new_cig_butts: 
			add_cigg_butt(
				img, mask, image_dir,
				mask_dir, os.listdir(image_dir), inital_size, 
				annotations, p=0.1)	

		if transforms:
			for i in range(len(transforms)):

				img_t, imsk_t  = transforms[i].transform(img, imask)
				img_t_name   = os.path.join(
					image_dir, str(scale * img_idx + i + 1) + '.jpg')
				img_t.save(img_t_name)
				imsk_t_name  = os.path.join(
					mask_dir, str(scale * img_idx + i + 1) + '.jpg')
				imsk_t.save(imsk_t_name)
	

	foo = len(os.listdir(image_dir))
	print(IMG_MEAN_R / foo)
	print(IMG_MEAN_G / foo)
	print(IMG_MEAN_B / foo)

	return IMG_MEAN_R / foo, IMG_MEAN_G / foo, IMG_MEAN_B / foo


def main():

	extend_dataset('cig_butts/train',     add_new_cig_butts=True)
	extend_dataset('cig_butts/val',       None)


if __name__ == "__main__":
	main()











