"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders
from models.bayesian_encoders import baye_psp_encoders ##
from models.stylegan2.model import Generator
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()
		
	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BayesianGradualStyleEncoder':
			encoder = baye_psp_encoders.BayesianGradualStyleEncoder(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	##
	def get_code(self, x, mc_samples=1) :
		codes = self.encoder(x)
		for _ in range(mc_samples - 1) :
			codes += self.encoder(x)
		codes = codes / mc_samples
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		return codes

	def stack_latents(self, j, latents):
		#ex) i = 3
		if j == 0:
			temp = [latents[1]]
		return new_latents

	def calc_best_ws(self, x, mc_samples=5): # x = inputs # return [1,18,512] tensor
		ws = [self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x)]
		# sigmas의 각 리스트의 길이는 mc_samples
		best_idx = [] # length 18
		# for _ in range(mc_samples):
		# 	ws.append(self.encoder(x))
		#ws = [self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x)] 
		#print(ws[0].shape) # [1,18,512]
		for j in range(18):
			sigmas = [] # len : 5
			for i in range(mc_samples): 
				temp = []
				if i == 0:
					temp.extend(ws[i+1:])
				elif i == mc_samples-1:
					temp.extend(ws[:i])
				else:
					temp.extend(ws[:i])
					temp.extend(ws[i+1:])
				#print("i, temp: ",i,len(temp))
				curr = torch.stack(temp, dim=0) # curr latents to compute [4,1,18,512]
				#print("curr", curr.shape)
				sigma, curr = torch.std_mean(curr, dim=0)
				sigma = sigma.sum(dim=-1) # [1, 18] i=0이면 2,3,4,5 sigma. 거기서 j번째를 가져와야함.
				sigma = sigma.squeeze()
				#print("sigma j",float(sigma[j]))
				sigmas.append(float(sigma[j]))
			#print("sigmas",sigmas)
			best_idx.append(sigmas.index(max(sigmas)))
	
		best_ws = [] # should get 18 512x1 vectors
		for i, idx in enumerate(best_idx): #i는 0-17
			best_w = ws[idx][:,i,:]
			best_ws.append(best_w)
		#print("best_idx", best_idx)
		
		# curr = latents[1:,:,:,:]
		# sigma, codes = torch.std_mean(codes, dim=0)
		best_ws = torch.stack(best_ws, dim=1)

		# codes = torch.stack(ws, dim=0)
		# _, best_ws = torch.std_mean(codes, dim=0)
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				best_ws = best_ws + self.latent_avg.repeat(best_ws.shape[0], 1)
			else:
				best_ws = best_ws + self.latent_avg.repeat(best_ws.shape[0], 1, 1)
		return best_ws

	def get_code_w_sigma(self, x, mc_samples=1) :
		ws = [self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x), self.encoder(x)]
		codes = torch.stack(ws, dim=0)
		#print("code shape: ", codes.shape) # [5,1,18,512]
		sigma, codes = torch.std_mean(codes, dim=0)
		#print("code shape: ", codes.shape) # [1,18,512]
		#print("sigma shape: ", sigma.shape) # [1,18,512]
		sigma = sigma.sum(dim=-1)
		#print("sigma shape: ", sigma.shape) # [1,18]
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		return codes, sigma	
	##

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None, mc_samples = 1, mid_latent = False):
		#print("-----psp.py forward function-----")
		if input_code:
			codes = x
		else:
			codes = self.get_code(x, mc_samples)
			# normalize with respect to the center of an average face
			# if self.opts.start_from_latent_avg:
			# 	if self.opts.learn_in_w:
			# 		codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			# 	else:
			# 		codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = True
		images, result_latent = self.decoder([codes],
												input_is_latent=input_is_latent,
												randomize_noise=randomize_noise,
												return_latents=return_latents) 
		if mid_latent:
			return images, codes

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent

		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
