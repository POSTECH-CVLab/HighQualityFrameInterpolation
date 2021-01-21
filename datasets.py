import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import numbers
try:
	import accimage
except ImportError:
	accimage = None

import pdb



class Vimeo90K_full(Dataset):
    def __init__(self, path='data/vimeo_septuplet/', transform=None):
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        for sequence in dir_seq:
            seps = os.listdir(self.path + sequence+'/')
            seps.sort()

            for sep in seps:
                self.sep_list.append(self.path + sequence+'/' + sep+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Random reverse
        if random.random() < 0.5:
            frames.reverse()

        # Randome start for septuplet
        rand = random.random()
        if rand < 0.33:
            start_idx = 1
        elif rand < 0.66:
            start_idx = 2
        else:
            start_idx = 3

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+1) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+2) + frames[0][3:])
        fr_4 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+3) + frames[0][3:])
        fr_5 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+4) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)


class Vimeo90K_train(Dataset):
    def __init__(self, path='data/vimeo_septuplet/', transform=None):
        self.txt_path = path + 'sep_trainlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Random reverse
        if random.random() < 0.5:
            frames.reverse()

        # Randome start for septuplet
        rand = random.random()
        if rand < 0.33:
            start_idx = 1
        elif rand < 0.66:
            start_idx = 2
        else:
            start_idx = 3

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+1) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+2) + frames[0][3:])
        fr_4 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+3) + frames[0][3:])
        fr_5 = Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+4) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)


class Vimeo90K_test(Dataset):
    def __init__(self, path='data/vimeo_septuplet/', transform=None):
        self.txt_path = path + 'sep_testlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        testlist = [line for line in f.readlines()]
        f.close()

        for line in testlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (1) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (2) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (3) + frames[0][3:])
        fr_4 = Image.open(septuplet + frames[0][0:2] + '%d' % (4) + frames[0][3:])
        fr_5 = Image.open(septuplet + frames[0][0:2] + '%d' % (5) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)





###### Augmented Flying Objects Vimeo90K ######

class FO_Vimeo90K_train(Dataset):
    def __init__(self, path='data/vimeo_septuplet/', transform=None):
        self.txt_path = path + 'sep_trainlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Random reverse
        if random.random() < 0.5:
            frames.reverse()

        # Randome start for septuplet
        rand = random.random()
        if rand < 0.33:
            start_idx = 1
        elif rand < 0.66:
            start_idx = 2
        else:
            start_idx = 3

        # Read image
        fr_1 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx) + frames[0][3:]))
        fr_2 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+1) + frames[0][3:]))
        fr_3 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+2) + frames[0][3:]))
        fr_4 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+3) + frames[0][3:]))
        fr_5 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+4) + frames[0][3:]))



        ###### Add flying object ######

#         if random.random() < 0.75:
# #             objPath = 'data/VOCdevkit/VOC2012/ObjSegments/'
#             objPath = 'data/VOCdevkit/VOC2012/SegmentationObject/'
#             objects = os.listdir(objPath)
#             obj = Image.open(objPath + random.choice(objects))
#             # Resize object
#             rand = random.random()
#             if rand < 0.5:
#                 obj = np.array(obj.resize((32,32), Image.BILINEAR))
#             else:
#                 obj = np.array(obj)

#             # Determine whether to alpha-blend
#             if random.random() < 0.25: # 1/4 prob
#                 alpha = 1 ##### 0.5
#             else:
#                 alpha = 1

#             ### 1st frame ###
#             x1 = random.randint(144,176) # for minimum of 32 pixel shift
#             y1 = random.randint(64,192)

#             # alpha-blend (or erase) foreground
#             fr_1[y1:y1+len(obj), x1:x1+len(obj), :][obj>0] = fr_1[y1:y1+len(obj), x1:x1+len(obj), :][obj>0]*(1-alpha)
#             # Paste object (alpha-blended)
#             fr_1[y1:y1+len(obj), x1:x1+len(obj), :] = fr_1[y1:y1+len(obj), x1:x1+len(obj), :] + obj*alpha

#             ### 5th frame ###
#             x5 = random.randint(240,272) # for minimum of 32 pixel shift
#             y5 = random.randint(64,192)

#             # alpha-blend (or erase) foreground
#             fr_5[y5:y5+len(obj), x5:x5+len(obj), :][obj>0] = fr_5[y5:y5+len(obj), x5:x5+len(obj), :][obj>0]*(1-alpha)
#             # Paste object (alpha-blended)
#             fr_5[y5:y5+len(obj), x5:x5+len(obj), :] = fr_5[y5:y5+len(obj), x5:x5+len(obj), :] + obj*alpha

#             ### 3rd frame ###
#             x3 = (x1+x5)//2
#             y3 = (y1+y5)//2

#             # alpha-blend (or erase) foreground
#             fr_3[y3:y3+len(obj), x3:x3+len(obj), :][obj>0] = fr_3[y3:y3+len(obj), x3:x3+len(obj), :][obj>0]*(1-alpha)
#             # Paste object (alpha-blended)
#             fr_3[y3:y3+len(obj), x3:x3+len(obj), :] = fr_3[y3:y3+len(obj), x3:x3+len(obj), :] + obj*alpha

#             ### 2nd frame ###
#             x2 = (x1+x3)//2
#             y2 = (y1+y3)//2

#             # alpha-blend (or erase) foreground
#             fr_2[y2:y2+len(obj), x2:x2+len(obj), :][obj>0] = fr_2[y2:y2+len(obj), x2:x2+len(obj), :][obj>0]*(1-alpha)
#             # Paste object (alpha-blended)
#             fr_2[y2:y2+len(obj), x2:x2+len(obj), :] = fr_2[y2:y2+len(obj), x2:x2+len(obj), :] + obj*alpha

#             ### 4th frame ###
#             x4 = (x3+x5)//2
#             y4 = (y3+y5)//2

#             # alpha-blend (or erase) foreground
#             fr_4[y4:y4+len(obj), x4:x4+len(obj), :][obj>0] = fr_4[y4:y4+len(obj), x4:x4+len(obj), :][obj>0]*(1-alpha)
#             # Paste object (alpha-blended)
#             fr_4[y4:y4+len(obj), x4:x4+len(obj), :] = fr_4[y4:y4+len(obj), x4:x4+len(obj), :] + obj*alpha

        # Convert frames back to PIL image
        fr_1 = Image.fromarray(fr_1)
        fr_2 = Image.fromarray(fr_2)
        fr_3 = Image.fromarray(fr_3)
        fr_4 = Image.fromarray(fr_4)
        fr_5 = Image.fromarray(fr_5)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)



class FO_all_Vimeo90K_train(Dataset):
    def __init__(self, path='data/vimeo_septuplet/', transform=None):
        self.txt_path = path + 'sep_trainlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Random reverse
        if random.random() < 0.5:
            frames.reverse()

        # Randome start for septuplet
        rand = random.random()
        if rand < 0.33:
            start_idx = 1
        elif rand < 0.66:
            start_idx = 2
        else:
            start_idx = 3

        # Read image
        fr_1 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx) + frames[0][3:]))
        fr_2 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+1) + frames[0][3:]))
        fr_3 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+2) + frames[0][3:]))
        fr_4 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+3) + frames[0][3:]))
        fr_5 = np.array(Image.open(septuplet + frames[0][0:2] + '%d' % (start_idx+4) + frames[0][3:]))



        ###### Add flying object ######

        objPath = 'data/VOCdevkit/VOC2012/ObjSegments/'
        objects = os.listdir(objPath)
        obj = Image.open(objPath + random.choice(objects))
        # Resize object
        rand = random.random()
        if rand < 0.5:
            obj = np.array(obj.resize((32,32), Image.BILINEAR))
        else:
            obj = np.array(obj)

        # alpha-blend
        alpha = 1
        
        ### 1st frame ###
        x1 = random.randint(144,176) # for minimum of 32 pixel shift
        y1 = random.randint(64,192)

        # alpha-blend (or erase) foreground
        fr_1[y1:y1+len(obj), x1:x1+len(obj), :][obj>0] = fr_1[y1:y1+len(obj), x1:x1+len(obj), :][obj>0]*(1-alpha)
        # Paste object (alpha-blended)
        fr_1[y1:y1+len(obj), x1:x1+len(obj), :] = fr_1[y1:y1+len(obj), x1:x1+len(obj), :] + obj*alpha

        ### 5th frame ###
        x5 = random.randint(240,272) # for minimum of 32 pixel shift
        y5 = random.randint(64,192)

        # alpha-blend (or erase) foreground
        fr_5[y5:y5+len(obj), x5:x5+len(obj), :][obj>0] = fr_5[y5:y5+len(obj), x5:x5+len(obj), :][obj>0]*(1-alpha)
        # Paste object (alpha-blended)
        fr_5[y5:y5+len(obj), x5:x5+len(obj), :] = fr_5[y5:y5+len(obj), x5:x5+len(obj), :] + obj*alpha

        ### 3rd frame ###
        x3 = (x1+x5)//2
        y3 = (y1+y5)//2

        # alpha-blend (or erase) foreground
        fr_3[y3:y3+len(obj), x3:x3+len(obj), :][obj>0] = fr_3[y3:y3+len(obj), x3:x3+len(obj), :][obj>0]*(1-alpha)
        # Paste object (alpha-blended)
        fr_3[y3:y3+len(obj), x3:x3+len(obj), :] = fr_3[y3:y3+len(obj), x3:x3+len(obj), :] + obj*alpha

        ### 2nd frame ###
        x2 = (x1+x3)//2
        y2 = (y1+y3)//2

        # alpha-blend (or erase) foreground
        fr_2[y2:y2+len(obj), x2:x2+len(obj), :][obj>0] = fr_2[y2:y2+len(obj), x2:x2+len(obj), :][obj>0]*(1-alpha)
        # Paste object (alpha-blended)
        fr_2[y2:y2+len(obj), x2:x2+len(obj), :] = fr_2[y2:y2+len(obj), x2:x2+len(obj), :] + obj*alpha

        ### 4th frame ###
        x4 = (x3+x5)//2
        y4 = (y3+y5)//2

        # alpha-blend (or erase) foreground
        fr_4[y4:y4+len(obj), x4:x4+len(obj), :][obj>0] = fr_4[y4:y4+len(obj), x4:x4+len(obj), :][obj>0]*(1-alpha)
        # Paste object (alpha-blended)
        fr_4[y4:y4+len(obj), x4:x4+len(obj), :] = fr_4[y4:y4+len(obj), x4:x4+len(obj), :] + obj*alpha

        # Convert frames back to PIL image
        fr_1 = Image.fromarray(fr_1)
        fr_2 = Image.fromarray(fr_2)
        fr_3 = Image.fromarray(fr_3)
        fr_4 = Image.fromarray(fr_4)
        fr_5 = Image.fromarray(fr_5)

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        sample['x2'] = x2
        sample['y2'] = y2
        sample['x4'] = x4
        sample['y4'] = y4

        return sample

    def __len__(self):
        return len(self.sep_list)





### Triplet train set ###
class Vimeo90K_train_triplet(Dataset):
    def __init__(self, path='data/vimeo_triplet/', transform=None):
        self.txt_path = path + 'tri_trainlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Random reverse
        if random.random() < 0.5:
            frames.reverse()

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (1) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (2) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (3) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)


### Triplet test set ###
class Vimeo90K_test_triplet(Dataset):
    def __init__(self, path='data/vimeo_triplet/', transform=None):
        self.txt_path = path + 'tri_testlist.txt'
        self.path = path + 'sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (1) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (2) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (3) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)

### Triplet test set ###
class Vimeo90K_test_SR_triplet(Dataset):
    def __init__(self, path='data/vimeo_triplet/', transform=None):
        self.txt_path = path + 'tri_testlist.txt'
        self.path = path + 'SR-sequences/'
        self.transform = transform
        self.sep_list = []

        f = open(self.txt_path, "r")
        trainlist = [line for line in f.readlines()]
        f.close()

        for line in trainlist:
            self.sep_list.append(self.path + line[:-1]+'/')

    def __getitem__(self, idx):
        septuplet = self.sep_list[idx]
        frames = os.listdir(septuplet)
        frames.sort()

        # Read image
        fr_1 = Image.open(septuplet + frames[0][0:2] + '%d' % (1) + frames[0][3:])
        fr_2 = Image.open(septuplet + frames[0][0:2] + '%d' % (2) + frames[0][3:])
        fr_3 = Image.open(septuplet + frames[0][0:2] + '%d' % (3) + frames[0][3:])

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sep_list)












######################################################
### SMBV test set ###
class SMBV_test_triplet(Dataset):
    def __init__(self, path='data/SMBV_test/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        '''
        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-2]:
                self.img_list.append(self.path + sequence+'/' + img)
        '''

        imgs = os.listdir(self.path + '/test_video_04/')
        imgs.sort()

        for img in imgs[:-2]:
            self.img_list.append(self.path + '/test_video_04/' + img)


    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)


class SMBV_test_quin(Dataset):
    def __init__(self, path='data/SMBV_test/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        
        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-4]:
                self.img_list.append(self.path + sequence+'/' + img)
        
        '''
        imgs = os.listdir(self.path + '/test_video_04/')
        imgs.sort()

        for img in imgs[:-4]:
            self.img_list.append(self.path + '/test_video_04/' + img)
        '''

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')
        fr_4 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+3) + '.png')
        fr_5 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+4) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)



######################################################
### GoPro test set ###
class GoPro_test_triplet(Dataset):
    def __init__(self, path='data/GOPRO_Large_all/test_reduced/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-2]: # 30 entries
                self.img_list.append(self.path + sequence+'/' + img)
                

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)



class GoPro_test_quin(Dataset):
    def __init__(self, path='data/GOPRO_Large_all/test_reduced/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-4]: # 30 entries
                self.img_list.append(self.path + sequence+'/' + img)
                

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')
        fr_4 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+3) + '.png')
        fr_5 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+4) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)





######################################################
### Samsung test set ###
class samsung_test_triplet(Dataset):
    def __init__(self, path='data/samsung_selected/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-2]: # 30 entries
                self.img_list.append(self.path + sequence+'/' + img)
                

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)



class samsung_test_quin(Dataset):
    def __init__(self, path='data/samsung_selected/', transform=None):
        self.path = path
        self.transform = transform
        self.img_list = []

        dir_seq = os.listdir(self.path)
        dir_seq.sort()

        for sequence in dir_seq:
            imgs = os.listdir(self.path + sequence+'/')
            imgs.sort()

            for img in imgs[:-4]: # 30 entries
                self.img_list.append(self.path + sequence+'/' + img)
                

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # Read image
        fr_1 = Image.open(img_path)
        fr_2 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+1) + '.png')
        fr_3 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+2) + '.png')
        fr_4 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+3) + '.png')
        fr_5 = Image.open(img_path[:-7] + '%03d' % (int(img_path[-7:-4])+4) + '.png')

        sample = {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)



######################################################
#Transforms

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3, fr_4, fr_5 = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample['frame5']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_LEFT_RIGHT)
            fr_2 = fr_2.transpose(Image.FLIP_LEFT_RIGHT)
            fr_3 = fr_3.transpose(Image.FLIP_LEFT_RIGHT)
            fr_4 = fr_4.transpose(Image.FLIP_LEFT_RIGHT)
            fr_5 = fr_5.transpose(Image.FLIP_LEFT_RIGHT)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}


class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3, fr_4, fr_5 = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample['frame5']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_TOP_BOTTOM)
            fr_2 = fr_2.transpose(Image.FLIP_TOP_BOTTOM)
            fr_3 = fr_3.transpose(Image.FLIP_TOP_BOTTOM)
            fr_4 = fr_4.transpose(Image.FLIP_TOP_BOTTOM)
            fr_5 = fr_5.transpose(Image.FLIP_TOP_BOTTOM)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3, 'frame4': fr_4, 'frame5': fr_5}


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image or numpy.ndarray): Images to be converted to tensor.

        Returns:
            Tensor: Converted images.
        """
        fr_1, fr_2, fr_3, fr_4, fr_5 = sample['frame1'], sample['frame2'], sample['frame3'], sample['frame4'], sample['frame5']
        pics = [fr_1, fr_2, fr_3, fr_4, fr_5]

        num = 0
        for pic in pics:
            if isinstance(pic, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                # backward compatibility
                pic = pic.float().div(255)

            if accimage is not None and isinstance(pic, accimage.Image):
                nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                pic.copyto(nppic)
                pic = torch.from_numpy(nppic)

            # handle PIL Image
            if pic.mode == 'I':
                img = torch.from_numpy(np.array(pic, np.int32, copy=False))
            elif pic.mode == 'I;16':
                img = torch.from_numpy(np.array(pic, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                pic = img.float().div(255)
            else:
                pic = img

            pics[num] = pic
            num += 1

        return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2], 'frame4': pics[3], 'frame5': pics[4]}







######################################################
#Transforms for triplet
class RandomHorizontalFlip_tri(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_LEFT_RIGHT)
            fr_2 = fr_2.transpose(Image.FLIP_LEFT_RIGHT)
            fr_3 = fr_3.transpose(Image.FLIP_LEFT_RIGHT)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}


class RandomVerticalFlip_tri(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be flipped.

        Returns:
            PIL.Image: Randomly flipped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        if random.random() < 0.5:
            fr_1 = fr_1.transpose(Image.FLIP_TOP_BOTTOM)
            fr_2 = fr_2.transpose(Image.FLIP_TOP_BOTTOM)
            fr_3 = fr_3.transpose(Image.FLIP_TOP_BOTTOM)

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}


class ToTensor_tri(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image or numpy.ndarray): Images to be converted to tensor.

        Returns:
            Tensor: Converted images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']
        pics = [fr_1, fr_2, fr_3]

        num = 0
        for pic in pics:
            if isinstance(pic, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(pic.transpose((2, 0, 1)))
                # backward compatibility
                pic = pic.float().div(255)

            if accimage is not None and isinstance(pic, accimage.Image):
                nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
                pic.copyto(nppic)
                pic = torch.from_numpy(nppic)

            # handle PIL Image
            if pic.mode == 'I':
                img = torch.from_numpy(np.array(pic, np.int32, copy=False))
            elif pic.mode == 'I;16':
                img = torch.from_numpy(np.array(pic, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                pic = img.float().div(255)
            else:
                pic = img

            pics[num] = pic
            num += 1

        return {'frame1': pics[0], 'frame2': pics[1], 'frame3': pics[2]}


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __call__(self, sample):
        """
        Args:
            sample (PIL.Image): Images to be cropped.

        Returns:
            PIL.Image: Cropped images.
        """
        fr_1, fr_2, fr_3 = sample['frame1'], sample['frame2'], sample['frame3']

        w, h = fr_1.size
        th = 256
        tw = 448

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        fr_1 = fr_1.crop((x1, y1, x1 + tw, y1 + th))
        fr_2 = fr_2.crop((x1, y1, x1 + tw, y1 + th))
        fr_3 = fr_3.crop((x1, y1, x1 + tw, y1 + th))

        return {'frame1': fr_1, 'frame2': fr_2, 'frame3': fr_3}