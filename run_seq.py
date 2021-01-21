import argparse
import os
import sys
from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.models import FullNet3

from frame2vid import frame2vid
from frame2vid_original import frame2vid_original

from PIL import Image
import numpy as np
from scipy import signal
import copy
import math
import pdb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./trained_models/FullNet_v5.pth') ####2
parser.add_argument('--in_file', default='/media/ssd3/js2/FrameInt/data/samsung/00/')
parser.add_argument('--out_file', default='/media/ssd3/js2/FrameInt/output/samsung/00/')
parser.add_argument('--n_iter', type=int, default=1, help='number of middle frames')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################
### Networks
FullNet = FullNet3()

# Place Network in cuda memory
if opt.cuda:
    FullNet.cuda()

### DataParallel
FullNet = nn.DataParallel(FullNet)
FullNet.eval()

FullNet.load_state_dict(torch.load(opt.model))

def gen_FullNet(f1, f2, f3):
    with torch.no_grad():
        orig_w = f1.size(3)
        orig_h = f1.size(2)
        temp_w = int(math.floor(math.ceil(f1.size(3) / 8.0) * 8.0)) # Due to stacked strided Conv layers
        temp_h = int(math.floor(math.ceil(f1.size(2) / 8.0) * 8.0))

        f1 = torch.nn.functional.interpolate(input=f1, size=(temp_h, temp_w), mode='bilinear')
        f2 = torch.nn.functional.interpolate(input=f2, size=(temp_h, temp_w), mode='bilinear')
        f3 = torch.nn.functional.interpolate(input=f3, size=(temp_h, temp_w), mode='bilinear')

        f1_5, f2_5 = FullNet(f1, f2, f3)

        f1_5 = torch.nn.functional.interpolate(input=f1_5, size=(orig_h, orig_w), mode='bilinear')
        f2_5 = torch.nn.functional.interpolate(input=f2_5, size=(orig_h, orig_w), mode='bilinear')

        return f1_5, f2_5

##########################################################

frameList = os.listdir(opt.in_file)
frameList.sort()

if os.path.exists(opt.out_file):
	pass
else:
	os.makedirs(opt.out_file)

param_list = torch.cuda.FloatTensor(len(frameList)-2, 1, 3).fill_(0) ########### change upon model

print('--n_avg must be an odd number!!!')

### Generate output sequence
for num_iter in range(opt.n_iter):
	print('\nIter: ' + str(num_iter+1))

	if num_iter == 0:
		src = opt.in_file
	else:
		src = opt.out_file
	#end

	frameList = os.listdir(src)
	frameList.sort()
	for f in range(len(frameList[:-2])): # frame range

		f1 = torch.cuda.FloatTensor(np.array(Image.open(src + frameList[f])).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
		f2 = torch.cuda.FloatTensor(np.array(Image.open(src + frameList[f+1])).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
		f3 = torch.cuda.FloatTensor(np.array(Image.open(src + frameList[f+2])).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
	
		with torch.no_grad():
			f1_5, _ = gen_FullNet(f1, f2, f3) # fr_2_5

		# Save image
		if num_iter == 0:
			img1 = Image.fromarray(np.uint8(f1.cpu().squeeze().permute(1,2,0)*255))
			img1.save(opt.out_file + frameList[f][:-4] + '_0.png')

			img1_5 = Image.fromarray(np.uint8(f1_5.cpu().squeeze().permute(1,2,0)*255))
			img1_5.save(opt.out_file + frameList[f][:-4] + '_' + str(2**(opt.n_iter-1)) + '.png') # assign mid number
		else:
			img1_5 = Image.fromarray(np.uint8(f1_5.cpu().squeeze().permute(1,2,0)*255))
			img1_5.save(opt.out_file + frameList[f][:-6] + '_' + str(int(frameList[f][14]) + 2**(opt.n_iter-1-num_iter)) + '.png')

		sys.stdout.write('\rFrame: ' + str(f+1) + '/' + str(len(frameList)-2))
		sys.stdout.flush()
	#end
#end

# ### Assess with metrics
# print('Computing metrics...')
# metrics(in_src=opt.in_file, out_src=opt.out_file[:-1] + '_f/')
