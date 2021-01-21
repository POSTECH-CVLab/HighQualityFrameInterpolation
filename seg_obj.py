import os
import sys
import copy

import torch
from PIL import Image
import numpy as np

import pdb

imgPath = 'data/VOCdevkit/VOC2012/JPEGImages/'
segPath = 'data/VOCdevkit/VOC2012/SegmentationObject/'

outPath = 'data/VOCdevkit/VOC2012/ObjSegments/'

segList = os.listdir(segPath)
segList.sort()


def bbox2(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	ymin, ymax = np.where(rows)[0][[0, -1]]
	xmin, xmax = np.where(cols)[0][[0, -1]]
	return img[ymin:ymax+1, xmin:xmax+1]


for i,s in enumerate(segList):
	if not (segPath + s).endswith('.png'):
		continue
	seg = np.array(Image.open(segPath + s))
	img = np.array(Image.open(imgPath + s[:-4] + '.jpg'))

	seg_f = seg.flatten()
	seg_f[seg_f==255] = 0
	ent = set(seg_f.tolist())

	for idx in ent:
		if idx != 0:
			seg_i = copy.deepcopy(seg)
			seg_i[seg_i!=idx] = 0
			seg_i[seg_i==idx] = 1

			# Segmented object
			objSeg = bbox2(img * np.repeat(seg_i[:,:,np.newaxis], 3, axis=2))
			objSeg = Image.fromarray(objSeg).resize((64,64), Image.BILINEAR)

			# Save image
			objSeg.save(outPath + s[:-4] + '_' + str(idx) + '.png')

	sys.stdout.write('\rFrame: ' + str(i+1) + '/' + str(len(segList)))
	sys.stdout.flush()









	