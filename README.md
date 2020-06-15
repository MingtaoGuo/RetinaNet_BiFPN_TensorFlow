# RetinaNet_BiFPN_TensorFlow
Simply implement the RetinaNet with BiFPN (EfficientDet) by tensorflow

### BiFPN
![](https://github.com/MingtaoGuo/RetinaNet_BiFPN_TensorFlow/blob/master/IMGS/BiFPN.jpg)

# How to use
### Dataset
Pascal Voc: http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
### Training phase
1. Downloading the pre-trained model of ResNet50, and put it into the folder **resnet_ckpt** 
   
   Address: http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

2. According to your own dataset, modify the **config.py** by yourself
3. Executing **train.py** 
### Testing phase
1. Changing the IMG_PATH or VIDEO_PATH in **test.py** for testing
2. Executing **test.py**

# Requirement
1. python
2. tensorflow
3. pillow
4. numpy
5. cv2
# Results
|Total Loss|Class Loss|Box Loss|
|-|-|-|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/total_loss.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/class_loss.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/box_loss.jpg)|

![](https://github.com/MingtaoGuo/RetinaNet_BiFPN_TensorFlow/blob/master/IMGS/1.jpg)

# Reference
[1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
# Author
Mingtao Guo
