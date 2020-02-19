import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

from lib.metrics import dice


def plot_loss_history(loss_history: dict, axes, title=None):
	"""
	:params:
		loss_history - dict, contating pairs 
		{label: str, history: numpy 1d array}. For each
		label the length of array should be the same
		title - string title for graphics
		axes  - axes to plot graphics
	"""

	foo = len(list(loss_history.values())[0]) 
	x   = np.array([a for a in range(1, foo + 1)])
	for label in loss_history:
		line,  = axes.plot(x, loss_history[label])
		line.set_label(label)

	if title: axes.set_title(title)
	axes.set_xlabel('epoch')
	axes.grid(False)
	axes.legend()


def evaluate_dice(
	preds: np.ndarray,
	labels: np.ndarray,
	tolerance = 0.95
	):
	"""
	A function to evaluate Dice coefficient over the batch
	of images
	:params:
		preds - batch of model predictions of pixel classes
		for images. Shape: (N_batch, 1, W_image, H_image)
		Labels - batch of correct pixel clasees for images 
		Shape: (N_batch, 1, W_image, H_image)
		tolerance - threshold value to consider a pixel
		to be in the second class. Default=0.9
	:returns: 
		DICE - average Dice metrics over the batch
	"""

	AVG_DICE = 0.0
	empty    = 0.0
	for true, pred in zip(labels, preds):
		if not np.sum(true):
			empty += 1.
		AVG_DICE += dice(true[0], pred[0] > tolerance)

	return AVG_DICE / (preds.shape[0] - empty) if empty != preds.shape[0] else 0.0

