import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.PWCNet.PWCNet import pwc_dc_net
#from models.MegaDepth.MegaDepth_model import HourGlass
#from models.ContextNet import contextnet


import copy
import math
import pdb







##### For 3 frame input
class FullNet3(nn.Module):
	def __init__(self):
		super(FullNet3, self).__init__()
		################# Load flow network #################
		self.pwcnet = pwc_dc_net('./models/PWCNet/pwc_net.pth.tar')

		################# Create Non-linear Network #################
		class NonLinNet(nn.Module):
			def __init__(self):
				super(NonLinNet, self).__init__()

				class Encoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Encoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						return s

				class Decoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Decoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

					def forward(self, x, x_c):
						x = F.interpolate(x, scale_factor=2, mode='bilinear')
						x = torch.cat([x, x_c],1).cuda()
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						return s

				self.enc1 = Encoder(16, 32, stride=2, k_size=7, pad=3)
				self.enc2 = Encoder(32, 64, stride=2, k_size=5, pad=2)
				self.enc3 = Encoder(64, 64, stride=2, k_size=5, pad=2)
				self.enc4 = Encoder(64, 64, stride=2, k_size=3, pad=1)

				self.dec1 = Decoder(64+64, 64, stride=1, k_size=3, pad=1)
				self.dec2 = Decoder(64+64, 64, stride=1, k_size=3, pad=1)
				self.dec3 = Decoder(64+32, 32, stride=1, k_size=3, pad=1)
				self.dec4 = Decoder(32+16, 8, stride=1, k_size=3, pad=1)

			def forward(self, flo_01, flo_10, flo_12, flo_21, flo_02_1, flo_20_1, flo_02_2, flo_20_2):
				x = torch.cat([flo_01, flo_10, flo_12, flo_21, flo_02_1, flo_20_1, flo_02_2, flo_20_2],1).cuda()
				s1 = self.enc1(x)
				s2 = self.enc2(s1)
				s3 = self.enc3(s2)
				s4 = self.enc4(s3)

				d1 = self.dec1(s4, s3)
				d2 = self.dec2(d1, s2)
				d3 = self.dec3(d2, s1)
				flo_out = self.dec4(d3, x)
				return flo_out

		self.NonLinNet = NonLinNet()

		################# Create Frame Generation Network #################
		class FrGenNet(nn.Module):
			def __init__(self):
				super(FrGenNet, self).__init__()

				class Encoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Encoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

						self.GateConv = nn.Sequential(
							nn.ReplicationPad2d(pad),
							nn.Conv2d(out_nc, out_nc, kernel_size=k_size, stride=1, padding=0),
							nn.Sigmoid()
						)

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						g = s * self.GateConv(s)
						return g

				class ResGateBlock(nn.Module):
					def __init__(self, in_nc, stride, k_size, pad):
						super(ResGateBlock, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

						self.GateConv = nn.Sequential(
							nn.ReplicationPad2d(pad),
							nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=1, padding=0),
							nn.Sigmoid()
						)

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						g = s * self.GateConv(s)
						return g + x

				self.seq = nn.Sequential(
					Encoder(6, 64, stride=1, k_size=5, pad=2),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					Encoder(64, 3, stride=1, k_size=5, pad=2)
				)


			def forward(self, w1, w2):
				x = torch.cat([w1, w2],1).cuda()
				fr_hat = self.seq(x)

				return fr_hat

		self.FrGenNet = FrGenNet()

	def warp(self, x, flo):
		"""
		warp an image/tensor (im2) back to im1, according to the optical flow

		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow

		"""
		B, C, H, W = x.size()
		# mesh grid 
		xx = torch.arange(0, W).view(1,-1).cuda().repeat(H,1)
		yy = torch.arange(0, H).view(-1,1).cuda().repeat(1,W)
		xx = xx.view(1,1,H,W).repeat(B,1,1,1)
		yy = yy.view(1,1,H,W).repeat(B,1,1,1)
		grid = torch.cat((xx,yy),1).float()

		# # if x.is_cuda:
		# #     grid = grid.cuda()
		# vgrid = Variable(grid) + flo
		#assert(B <= self.B_MAX and H <= self.H_MAX and W <= self.W_MAX)
		vgrid = grid[:B,:,:H,:W] +flo

		# scale grid to [-1,1] 
		vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
		vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0


		vgrid = vgrid.permute(0,2,3,1)        
		output = nn.functional.grid_sample(x, vgrid)
		# mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
		mask = torch.autograd.Variable(torch.cuda.FloatTensor().resize_(x.size()).zero_() + 1, requires_grad = False)
		mask = nn.functional.grid_sample(mask, vgrid)

		# if W==128:
			# np.save('mask.npy', mask.cpu().data.numpy())
			# np.save('warp.npy', output.cpu().data.numpy())

		mask[mask<0.9999] = 0
		mask[mask>0] = 1

		return output*mask

	def getFlow(self, i0, i1):
		temp_w = int(math.floor(math.ceil(i0.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
		temp_h = int(math.floor(math.ceil(i0.size(2) / 64.0) * 64.0))

		temp_i0 = torch.nn.functional.interpolate(input=i0, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		temp_i1 = torch.nn.functional.interpolate(input=i1, size=(temp_h, temp_w), mode='bilinear', align_corners=False)

		flo = 20.0 * torch.nn.functional.interpolate(input=self.pwcnet(temp_i0, temp_i1), size=(i0.size(2), i0.size(3)), mode='bilinear', align_corners=False)
		return flo

	def getNL(self, flo_01, flo_10, flo_12, flo_21, flo_02_1, flo_20_1, flo_02_2, flo_20_2):
		temp_w = int(math.floor(math.ceil(flo_01.size(3) / 16.0) * 16.0)) # Due to Pyramid method?
		temp_h = int(math.floor(math.ceil(flo_01.size(2) / 16.0) * 16.0))

		t_flo_01 = torch.nn.functional.interpolate(input=flo_01, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_10 = torch.nn.functional.interpolate(input=flo_10, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_12 = torch.nn.functional.interpolate(input=flo_12, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_21 = torch.nn.functional.interpolate(input=flo_21, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_02_1 = torch.nn.functional.interpolate(input=flo_02_1, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_20_1 = torch.nn.functional.interpolate(input=flo_20_1, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_02_2 = torch.nn.functional.interpolate(input=flo_02_2, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_20_2 = torch.nn.functional.interpolate(input=flo_20_2, size=(temp_h, temp_w), mode='bilinear', align_corners=False)

		flo_out = torch.nn.functional.interpolate(input=self.NonLinNet(t_flo_01, t_flo_10, t_flo_12, t_flo_21, t_flo_02_1, t_flo_20_1, t_flo_02_2, t_flo_20_2), size=(flo_01.size(2), flo_01.size(3)), mode='bilinear', align_corners=False)
		return flo_out

	def switch_to_finetune(self):
		self.train()
		self.pwcnet.train()

	def switch_to_pretrain(self):
		self.train()
		self.pwcnet.eval()

	def forward(self, i0, i1, i2, time_step=0.5):
		##### Get flows #####
		flo_01 = self.getFlow(i0, i1) * time_step
		flo_10 = self.getFlow(i1, i0) * (1.0-time_step)
		flo_12 = self.getFlow(i1, i2) * time_step
		flo_21 = self.getFlow(i2, i1) * (1.0-time_step)

		flo_02_1 = self.getFlow(i0, i2) * 0.5*time_step # 0.25
		flo_20_1 = self.getFlow(i2, i0) * (1.0-(0.5*time_step)) # 0.75
		flo_02_2 = self.getFlow(i0, i2) * (0.5 + 0.5*time_step) # 0.75
		flo_20_2 = self.getFlow(i2, i0) * (1.0-(0.5 + 0.5*time_step)) # 0.25

		##### Non-linear module #####
		flo_out = self.getNL(flo_01, flo_10, flo_12, flo_21, flo_02_1, flo_20_1, flo_02_2, flo_20_2)

		##### Frame generation module #####
		w01 = self.warp(i1, flo_out[:,:2,:,:])
		w10 = self.warp(i0, flo_out[:,2:4,:,:])
		w12 = self.warp(i2, flo_out[:,4:6,:,:])
		w21 = self.warp(i1, flo_out[:,6:,:,:])

		fr_hat1 = self.FrGenNet(w01, w10)
		fr_hat2 = self.FrGenNet(w12, w21)

		##### Clamp output #####
		fr_hat1 = torch.clamp(fr_hat1, 0, 1.0)
		fr_hat2 = torch.clamp(fr_hat2, 0, 1.0)

		return fr_hat1, fr_hat2













##### For 2 frame input
class FullNet2(nn.Module):
	def __init__(self):
		super(FullNet2, self).__init__()
		################# Load flow network #################
		self.pwcnet = pwc_dc_net('./models/PWCNet/pwc_net.pth.tar')

		################# Create Non-linear Network #################
		class NonLinNet(nn.Module):
			def __init__(self):
				super(NonLinNet, self).__init__()

				class Encoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Encoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						return s

				class Decoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Decoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

					def forward(self, x, x_c):
						x = F.interpolate(x, scale_factor=2, mode='bilinear')
						x = torch.cat([x, x_c],1).cuda()
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						return s

				self.enc1 = Encoder(4, 32, stride=2, k_size=7, pad=3)
				self.enc2 = Encoder(32, 64, stride=2, k_size=5, pad=2)
				self.enc3 = Encoder(64, 64, stride=2, k_size=5, pad=2)
				self.enc4 = Encoder(64, 64, stride=2, k_size=3, pad=1)

				self.dec1 = Decoder(64+64, 64, stride=1, k_size=3, pad=1)
				self.dec2 = Decoder(64+64, 64, stride=1, k_size=3, pad=1)
				self.dec3 = Decoder(64+32, 32, stride=1, k_size=3, pad=1)
				self.dec4 = Decoder(32+4, 4, stride=1, k_size=3, pad=1)

			def forward(self, flo_01, flo_10):
				x = torch.cat([flo_01, flo_10],1).cuda()
				s1 = self.enc1(x)
				s2 = self.enc2(s1)
				s3 = self.enc3(s2)
				s4 = self.enc4(s3)

				d1 = self.dec1(s4, s3)
				d2 = self.dec2(d1, s2)
				d3 = self.dec3(d2, s1)
				flo_out = self.dec4(d3, x)
				return flo_out

		self.NonLinNet = NonLinNet()

		################# Create Frame Generation Network #################
		class FrGenNet(nn.Module):
			def __init__(self):
				super(FrGenNet, self).__init__()

				class Encoder(nn.Module):
					def __init__(self, in_nc, out_nc, stride, k_size, pad):
						super(Encoder, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

						self.GateConv = nn.Sequential(
							nn.ReplicationPad2d(pad),
							nn.Conv2d(out_nc, out_nc, kernel_size=k_size, stride=1, padding=0),
							nn.Sigmoid()
						)

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						g = s * self.GateConv(s)
						return g

				class ResGateBlock(nn.Module):
					def __init__(self, in_nc, stride, k_size, pad):
						super(ResGateBlock, self).__init__()

						self.padd = nn.ReplicationPad2d(pad)
						self.Conv = nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0)
						self.relu = nn.ReLU()

						self.GateConv = nn.Sequential(
							nn.ReplicationPad2d(pad),
							nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=1, padding=0),
							nn.Sigmoid()
						)

					def forward(self, x):
						s = self.padd(x)
						s = self.Conv(s)
						s = self.relu(s)
						g = s * self.GateConv(s)
						return g + x

				self.seq = nn.Sequential(
					Encoder(6, 64, stride=1, k_size=5, pad=2),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					ResGateBlock(64, stride=1, k_size=3, pad=1),
					Encoder(64, 3, stride=1, k_size=5, pad=2)
				)


			def forward(self, w1, w2):
				x = torch.cat([w1, w2],1).cuda()
				fr_hat = self.seq(x)

				return fr_hat

		self.FrGenNet = FrGenNet()

	def warp(self, x, flo):
		"""
		warp an image/tensor (im2) back to im1, according to the optical flow

		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow

		"""
		B, C, H, W = x.size()
		# mesh grid 
		xx = torch.arange(0, W).view(1,-1).cuda().repeat(H,1)
		yy = torch.arange(0, H).view(-1,1).cuda().repeat(1,W)
		xx = xx.view(1,1,H,W).repeat(B,1,1,1)
		yy = yy.view(1,1,H,W).repeat(B,1,1,1)
		grid = torch.cat((xx,yy),1).float()

		# # if x.is_cuda:
		# #     grid = grid.cuda()
		# vgrid = Variable(grid) + flo
		#assert(B <= self.B_MAX and H <= self.H_MAX and W <= self.W_MAX)
		vgrid = grid[:B,:,:H,:W] +flo

		# scale grid to [-1,1] 
		vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
		vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0


		vgrid = vgrid.permute(0,2,3,1)        
		output = nn.functional.grid_sample(x, vgrid)
		# mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
		mask = torch.autograd.Variable(torch.cuda.FloatTensor().resize_(x.size()).zero_() + 1, requires_grad = False)
		mask = nn.functional.grid_sample(mask, vgrid)

		# if W==128:
			# np.save('mask.npy', mask.cpu().data.numpy())
			# np.save('warp.npy', output.cpu().data.numpy())

		mask[mask<0.9999] = 0
		mask[mask>0] = 1

		return output*mask

	def getFlow(self, i0, i1):
		temp_w = int(math.floor(math.ceil(i0.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
		temp_h = int(math.floor(math.ceil(i0.size(2) / 64.0) * 64.0))

		temp_i0 = torch.nn.functional.interpolate(input=i0, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		temp_i1 = torch.nn.functional.interpolate(input=i1, size=(temp_h, temp_w), mode='bilinear', align_corners=False)

		flo = 20.0 * torch.nn.functional.interpolate(input=self.pwcnet(temp_i0, temp_i1), size=(i0.size(2), i0.size(3)), mode='bilinear', align_corners=False)
		return flo

	def getNL(self, flo_01, flo_10):
		temp_w = int(math.floor(math.ceil(flo_01.size(3) / 16.0) * 16.0)) # Due to Pyramid method?
		temp_h = int(math.floor(math.ceil(flo_01.size(2) / 16.0) * 16.0))

		t_flo_01 = torch.nn.functional.interpolate(input=flo_01, size=(temp_h, temp_w), mode='bilinear', align_corners=False)
		t_flo_10 = torch.nn.functional.interpolate(input=flo_10, size=(temp_h, temp_w), mode='bilinear', align_corners=False)

		flo_out = torch.nn.functional.interpolate(input=self.NonLinNet(t_flo_01, t_flo_10), size=(flo_01.size(2), flo_01.size(3)), mode='bilinear', align_corners=False)
		return flo_out

	def switch_to_finetune(self):
		self.train()
		self.pwcnet.train()

	def switch_to_pretrain(self):
		self.train()
		self.pwcnet.eval()

	def forward(self, i0, i1, time_step=0.5):
		##### Get flows #####
		flo_01 = self.getFlow(i0, i1) * time_step
		flo_10 = self.getFlow(i1, i0) * (1.0-time_step)

		##### Non-linear module #####
		flo_out = self.getNL(flo_01, flo_10)

		##### Frame generation module #####
		w01 = self.warp(i1, flo_out[:,:2,:,:])
		w10 = self.warp(i0, flo_out[:,2:,:,:])

		fr_hat = self.FrGenNet(w01, w10)

		##### Clamp output #####
		fr_hat = torch.clamp(fr_hat, 0, 1.0)

		return fr_hat