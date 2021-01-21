import os
import cv2
import pdb

readDir = './data/custom/'
writeDir = './data/custom/'

vidList = []
for f in os.listdir(readDir):
	if f.endswith('.mp4'):
		vidList.append(f)
vidList.sort()
print('Number of videos: %d' % len(vidList))


for vid in vidList:
	os.makedirs(writeDir + vid[:-4] +'/')
	vidcap = cv2.VideoCapture(readDir+vid)
	success, image = vidcap.read()
	count = 0

	while success:
		cv2.imwrite(writeDir + vid[:-4] +'/' + '00_frame%05d.png' % (count), image)    
		success, image = vidcap.read()
		count += 1

	print('Done: '+vid)
