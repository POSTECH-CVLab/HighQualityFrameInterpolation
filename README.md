# High-quality Frame Interpolation via Tridirectional Inference

This is an official source code of Choi et al., High-quality Frame Interpolation via Tridirectional Inference, WACV 2021 paper.

<p align="center">
  <img width="100%" src="https://github.com/POSTECH-CVLab/HighQualityFrameInterpolation/blob/master/figures/overview.jpg?raw=true" />
</p>


# Features
- Simple and effective video frame interpolation approach using **three consequent video frames**.
- Data-driven approach to interpolate moving objects.
- Generalizes well to high-resolution contents.

# Requirements
We tested our approach using the below environment. We recommend using docker and anaconda to avoid any hassles. (Tested with 4x Nvidia Titan X (12GB))

- Cuda 10.0
- cudatoolkit=10.0
- pytorch==1.0.1
- torchvision==0.2.2
- visdom==0.1.8.9
- [PWCNet](https://github.com/NVlabs/PWC-Net) (included).
- For more information, please check environment.yml file.


## Interpolating your own video with the pre-trained network
1. Check the file of the pretrained network. ```trained_models/FullNet_v5.pth```
2. Put your video file. For instance:
```
data/custom/13_1.mp4
```
3. Make a image sequence from a video using ```python vid2frames.py```
4. Run ```run_seq.py```.  For instance:
```
CUDA_VISIBLE_DEVICES=0 python run_seq.py --cuda --n_iter 1 --in_file data/custom/13_1/ --out_file data/custom/13_1_out/
```
5. Check the output folder to see interpolated frames. ```xx_1.png``` indicates interpolated frames, and ```xx_0.png``` indicates original video frames.


## Training the Network
1. Compile PWCNet in correlation package using the following script: 
```
cd models/PWCNet/correlation_package_pytorch1_0/ && ./build.sh
``` 
- If compilation is not working, consider modifying ```models/PWCNet/correlation_package_pytorch1_0/setup.py``` by changing the line ```'-gencode', 'arch=compute_75,code=sm_75'``` with the proper version by referencing [this](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

2. Download [Vimeo-90k Septuplet Dataset](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) in ```data``` folder (for providing plentiful video contents).
3. Download [PASCAL VOC 2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) in ```data``` folder (for augmenting flying objects).
4. Uncompress the downloaded files, and organize the data folder like below:

```
    data
        ├── vimeo_setuplet
        │   └── sequences
        ├── VOCdevkit
            └── VOC2012
                ├── :
                ├── SegmentationObject
                ├── ObjSegments
                └── :
    :
    train.py
    :
```
6. To add flying objects onto downloaded videos, run ```python seg_obj.py```. It crops image patches of objects from PASCAL VOC and save them into ```data/VOCdevkit/VOC2012/ObjSegments/```
7. Run Visdom ```python -m visdom.server -port [port_number]```
8. Train the network. For instance, to use 4 GPUs:
```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cuda --n_epoch 20 --decay_epoch 10 --batchSize 4 --n_cpu 8 --modelname './trained_models/FullNet_v5.pth```
*consider decrease ```--batchSize``` if the training crashes due to the small GPU memory.

# Trying with other datasets
(will be updated)
1. Download [SMBV dataset](http://www.cvg.unibe.ch/media/data/datasets/video/jin/slow-motion.zip) (Utilized four test videos)
2. Download [GoPro dataset](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view?usp=sharing) (We utilized eleven videos)

## Citation
Please cite our work if you use High-quality Frame Interpolation.
```bib
@article{Choi2021HQFrameInterpolation,
  title   = {{High-quality Frame Interpolation via Tridirectional Inference}},
  author  = {Jinsoo Choi and Jaesik Park and In So Kweon},
  journal = {Winter Conference on Applications of Computer Vision (WACV)},
  year    = {2021}
}
```