# MA-UNet
A Multiattention UNet with simAM, MSA, CAM, SAM

Quick Start Examples
========================
Install
------------------------
1. For single GPU training, please install the environment in "requirement for training with singe GPU.txt".<br>
2. For multi-GPU training, please install the environment in "requirement for training with multi-GPU.txt".<br>
3. Besides, Python3.7 is recommended.

Preparation of datasets
------------------------
1. All data should be placed in directory “VOCdevkit/VOC2007/”. <br>
   * The name of original image and its corresponding label must be consistent, their format can be different(important) <br>
      `Image: cat_00001.jpg ; Label: cat_00001.png`
2. Put all the original images in folder “JPEGImages” and all the labels in folder SegmentationClass.<br>
   * The default format is that the original image is ".jpg" format, the label is ".png" format. <br>
   * If your dataset format is different, please modify the format in the 82 and 83 lines of code in folder "utils/dataloader.py", as follow: <br>
       `jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))` <br>
       `png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))`
3. Run "vocdevkit/voc2unet" to divides training sets and test sets. <br>
   * If your label is not in png format, modify the code in line 17, as follow: <br>
       `if seg.endswith(".png"):`
   * You can also modify the 12 lines of code to divide the training set and test set according to other proportions, as follow. The default ratio is 8:2. <br>
       `train_percent = 0.8`
   * Finally, four text files will be generated in the "VOCdevkit/VOC2007/ImageSets/" directory.

Training
------------------------
1. Training with Multi-GPU. （recommended） <br>
    `python -m torch.distributed.launch --nproc_per_node=num_gpu train_multi_GPU.py` <br>
    If the memory is not released after training， use `pgrep python | xargs kill -s 9` <br>
   * It is worth noting that in the hyperparameters, num_classes should be set to the number of categories plus 1. <br>
   For example, if you want to segmentation cat and dog in the images, although there are only two categories, <br>
   you need to set it to 3, because the label of the background is 0. 
   * You can modify the hyperparameters in "train_multi_GPU.py" after line 150 to get better training results. <br>
   * Some tips for training: <br>
   (1) For 256x256 resolution input images, when the batchsize is 16, 10GB memory of gpu is required. <br>
   (2) If your memory of single GPU is not large(<12), it is recommended to use mixed accuracy training and sync_bn. <br>
   (3) If your memory of single GPU is large(>16), set the "input_shape" as large as possible, such as 512, 800, 1024. <br>
   (4) The hyperparameters "batch_size" is total batchsize for all GPU you used, If the batch of a single gpu is less than 4, use sync_bn. <br>
   * The model after each epoch of training will be saved in the "logs" folder. <br>
 
2. Training with single GPU. <br>
    `python train.py`
   * It is worth noting that in the hyperparameters, num_classes should be set to the number of categories plus 1. <br>
   * You can modify the hyperparameters in "train.py" after line 15 to get better training results. <br>
   * The model after each epoch of training will be saved in the "logs" folder. <br>

Prediction and Validation
------------------------
1. Prediction
   * You need to modify the 16 to 20 lines of code in the "unet.py" file, as follows：<br>
     `_defaults = {
        "model_path": 'MA-UNet.pth',
        "num_classes": 7,
        "input_shape": [256, 256],
        "blend": False,
        "cuda": True,
    }`
   * You can also modify the line 29 of code in "unet.py" to change the color in the prediction images. <br>
    `self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]`
   * The hyperparameters "input_shape" should be consistent with that set during training. <br>
   * Put the images which to be predicted in "imgs" folder, and run predict.py, and the results will be saved to the "save" folder.<br>

2. Validation
   * Reproduce and validate the results of two datasets in our paper. <br>
   (1) According to the parameters in our paper, using multiple GPUs to train two datasets for 200 epochs directly can easily reproduce our results.<br>
   (2) After modifying the model path in the "unet.py" file, run the following two codes:<br>
    `python get_miou_for_WHDLD.py` or `python get_miou_for_DLRSD.py`<br>
It is worth noting that since these two datasets do not have any background, but the label starts from 1, it is different from verifying other custom datasets, so our verification code for these two datasets eliminates the calculation of background.<br>
   * Validate custom datasets. <br>
   (1) The following contents in "get_miou_for_custom_datasets.py" need to be changed:<br>
      `line26 num_classes = 7` categories+1 <br>
      `line30 name_classes= ["background","1","2","3","4","5","6"]` categories names and add "background" <br>
      `line53 image_path= os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")` Change to your images format <br>
   (2) Run "get_miou_for_custom_datasets.py". 

Details of MA-UNet
------------------------
Please cite our papers if the code is useful for you. Thank you! <br>
 * MDPI and ACS Style <br>
Sun, Y.; Bi, F.; Gao, Y.; Chen, L.; Feng, S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry 2022, 14, 906. https://doi.org/10.3390/sym14050906 <br>

 * AMA Style <br>
Sun Y, Bi F, Gao Y, Chen L, Feng S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry. 2022; 14(5):906. https://doi.org/10.3390/sym14050906 <br>

 * Chicago/Turabian Style <br>
Sun, Yu, Fukun Bi, Yangte Gao, Liang Chen, and Suting Feng. 2022. "A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images" Symmetry 14, no. 5: 906. https://doi.org/10.3390/sym14050906 <br>

Reference
------------------------
https://github.com/bubbliiiing/unet-pytorch  <br>
https://github.com/ggyyzm/pytorch_segmentation  <br>
https://github.com/bonlime/keras-deeplab-v3-plus  <br>
