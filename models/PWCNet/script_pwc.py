import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import PWCNet
from flowlib import flow_to_image
import pdb
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())

def warp(x, flo):
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


im1_fn = 'data/frame_0010.png';
im2_fn = 'data/frame_0011.png';
flow_fn = './tmp/frame_0010.flo';

if len(sys.argv) > 1:
    im1_fn = sys.argv[1]
if len(sys.argv) > 2:
    im2_fn = sys.argv[2]
if len(sys.argv) > 3:
    flow_fn = sys.argv[3]

pwc_model_fn = './pwc_net.pth.tar';

im1 = imread(im1_fn)
im2 = imread(im2_fn)

# rescale the image size to be multiples of 64
divisor = 64.
H = im1.shape[0]
W = im1.shape[1]

H_ = int(ceil(H/divisor) * divisor)
W_ = int(ceil(W/divisor) * divisor)

im1 = cv2.resize(im1, (W_, H_))
im2 = cv2.resize(im2, (W_, H_))

im1 = im1[:, :, ::-1]
im1 = 1.0 * im1/255.0
im2 = im2[:, :, ::-1]
im2 = 1.0 * im2/255.0
	
im1 = np.transpose(im1, (2, 0, 1))
im1 = torch.from_numpy(im1)
im1 = im1.expand(1, im1.size()[0], im1.size()[1], im1.size()[2])	
im1 = im1.float().cuda()
im2 = np.transpose(im2, (2, 0, 1))
im2 = torch.from_numpy(im2)
im2 = im2.expand(1, im2.size()[0], im2.size()[1], im2.size()[2])	
im2 = im2.float().cuda()

##### Network #####
net = PWCNet.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

flo = net(im1, im2)


##### Display flow #####
flo_im = flo[0] * 20.0
flo_im = flo_im.cpu().data.numpy()

# scale the flow back to the input size 
flo_im = np.swapaxes(np.swapaxes(flo_im, 0, 1), 1, 2) # 
u_ = cv2.resize(flo_im[:,:,0],(W,H))
v_ = cv2.resize(flo_im[:,:,1],(W,H))
u_ *= W/ float(W_)
v_ *= H/ float(H_)
flo_im = np.dstack((u_,v_))

writeFlowFile(flow_fn, flo_im)

flo_img = flow_to_image(flo_im)
plt.imshow(flo_img)
plt.show()

##### Display warped image #####
flo_warp = 20.0 * torch.nn.functional.interpolate(flo, scale_factor=4, mode='bilinear', align_corners=False)

#pdb.set_trace()

img_warped = warp(im2, flo_warp)
img_warped = img_warped.squeeze()
img_warped = img_warped.cpu().data.numpy()
img_warped = np.transpose(img_warped, (1,2,0))
plt.imshow(img_warped)
plt.show()