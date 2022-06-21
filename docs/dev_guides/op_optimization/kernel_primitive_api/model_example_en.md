# Model
## Description
+ On the GPU platform, the operator written by the kernel primitive API is used by default.
+ Operators written using the kernel primitive API on the XPU platform need to enable FLAGS\_run\_kp\_kernel environment variables.

### XPU kernel primitive API model example

Take resnet50 as an example to show the basic process of XPU2 KP model operation.</br>
+ 1. Install PaddlePaddle XPU2 KP package. Currently, only python3.7 is supported.</br>
```
pip install https://paddle-wheel.bj.bcebos.com/2.3.0/xpu2/kp/paddlepaddle_xpu-2.3.0-cp37-cp37m-linux_x86_64.whl
```

+ 2. Download model library and install</br>
```
git clone -b develop https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
python -m pip install -r requirements.txt
```

+ 3. Downloading datasets <br>

```
cd dataset
rm -rf ILSVRC2012
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
tar xf whole_chain_CIFAR100.tar
ln -s whole_chain_CIFAR100 ILSVRC2012
cd ILSVRC2012
mv train.txt train_list.txt
mv test.txt val_list.txt
```

+ 4. Model operation</br>
```
cd ../..
export FLAGS_selected_xpus=0
export FLAGS_run_kp_kernel=1
export XPUSIM_DEVICE_MODEL=KUNLUN2
nohup python tools/train.py \
-c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
-o Global.device=xpu > ResNet50_xpu2.log &
```
+ 5. Creenshot is as follows: </br>
![Model](./images/example_model.png)

### XPU2 Kernel Primitive API Model List
Number | Model Name | Category
-- | -- | --
1 | resnet50 | Image Classification
2 | deepfm | Recommendation Network
3 | wide&deep | Recommendation Network
4 | yolov3-darknet53 | Object Detection
5 | ssd-resnet34 | Object Detection
6 | orc-db |  Text Detection
7 | bert-base | Natural Language Processing
8 | transformer | Natural Language Processing
9 | gpt-2 | Natural Language Processing
10 | unet | Image Segmentation
