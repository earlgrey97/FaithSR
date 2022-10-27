import torch
from torch import nn
import torch.nn.functional as F


class EncoderLoss(nn.Module):

	def __init__(self):
		super(EncoderLoss, self).__init__()
		self.cos_sim = torch.nn.CosineSimilarity()

	def forward(self, latent, latent_hr):
		return F.mse_loss(latent, latent_hr)
