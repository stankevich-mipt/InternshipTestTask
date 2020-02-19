import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision

from utils import evaluate_dice
from PIL import Image


def get_optimal_dice_threshold(model, val_batch_gen):
	"""
	Performs a small grid search over the threshold value
	for maximizing Dice metrics on validation dataset
	:params:
		model - a model to be evaluated. Should implement
			'forward' method
		val_batch_gen - torch.Dataloader for validation dataset

	"""

	thresholds = np.linspace(0.05, 0.95, 11)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	optimal_treshold = 1
	MAX_DICE         = 0.0

	for t in thresholds:

		DICE = []

		for batch in val_batch_gen:
			batch_img = batch['image'  ].to(device)
			batch_msk = batch['mask'   ].data.numpy()
                	
			batch_preds = model.forward(batch_img).cpu().data.numpy()
			metrics = lambda x, y: evaluate_dice(x, y, tolerance=t)
			DICE.append(metrics(batch_preds, batch_msk))

		DICE = np.mean(np.array(DICE))
		if DICE > MAX_DICE:
			MAX_DICE = DICE
			optimal_treshold = t

	return optimal_treshold, MAX_DICE


def get_samples_with_low_dice_metrics(
	model, val_batch_gen, save_folder, threshold=0.5
	):
	"""
	For given model, saves samples from validational dataset
	which give Dice coefficient lower than the threshold to the 
	'save_folder' directory
	:params:
		val_batch_gen - torch.Dataloader for validation dataset

		save_folder   - full directory path. Directory with that name
			must exist, or the function will return an exception

		threshold     - a border value to consider samples 'difficult'
			to NN. Default=0.5
	"""

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")	
	# basic font precet for images
	font = {
	    'family': 'serif',
	    'weight': 'normal',
	    'size'  : 16,
	    'y'     : 1.03  
	}

	for batch in val_batch_gen:
		batch_img = batch['image'  ].to(device)
		batch_msk = batch['mask'   ].data.numpy()

		counter = 1

		batch_preds = model.forward(batch_img).cpu().data.numpy()
		for i in range(batch_msk.shape[0]):
			metrics = evaluate_dice(
				batch_preds[i][np.newaxis, ...], batch_msk[i][np.newaxis,   ...])
			if metrics < threshold: 
				f, axarr = plt.subplots(1, 3, figsize=(12, 12))
				axarr[0].imshow(
	              np.transpose(batch_img[i].cpu().data.numpy(), [1, 2, 0]))
				axarr[1].imshow(batch_msk[i].reshape([512, 512]))
				axarr[2].imshow(batch_preds[i].reshape([512, 512]))
				axarr[0].set_title("Original image", fontdict=font)
				axarr[0].axis(False)
				axarr[1].set_title("Ground truth", fontdict=font)
				axarr[1].axis(False)
				axarr[2].set_title("Net prediction (Dice coeff - {:.2})".format(metrics),
					                 fontdict=font)
				axarr[2].axis(False)
				plt.savefig(os.path.join(save_folder, f'batch_{counter}_image_{i}.jpg'))

		counter += 1


def evaluate_model_on_test(model, test_dataset):
	"""
	Plots all the images, located in 'test_dataset' folder,
	with their segmentation, predicted by a model 
	:params:
		model        - a model to evaluate
		test_dataset - full path to test dataset  
	"""

	# basic font precet for images
	font = {
	    'family': 'serif',
	    'weight': 'normal',
	    'size'  : 16,
	    'y'     : 1.03  
	}

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	image_to_tensor = torchvision.transforms.ToTensor()
	dir_content = [file for file in os.listdir(test_dataset)]
	for image in dir_content:
	    img_path = os.path.join(test_dataset, image)
	    img      = image_to_tensor(Image.open(img_path).convert('RGB'))
	    img      = torch.unsqueeze(img, 0).to(device)
	    preds    = model.forward(img).cpu()
	    f, axarr = plt.subplots(1, 2, figsize=(12, 6))
	    axarr[0].imshow(np.transpose(img.cpu().squeeze(0).data.numpy(), [1, 2, 0]))
	    axarr[0].axis(False)
	    axarr[0].set_title("Original image", fontdict=font)
	    axarr[1].imshow(preds.view(512, 512).data.numpy())
	    axarr[1].axis(False)
	    axarr[1].set_title("Net prediction", fontdict=font)
