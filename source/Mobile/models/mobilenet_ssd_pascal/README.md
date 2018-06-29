# Mobilenet SSD  

We offer the mobilenet(1.0) ssd model trained on PASCAL VOC0712 dataset. This model can be deployed on embedded system
and you can modify the network to adapt to your own application.

## run the demo
1. Install PaddlePaddle(see:  [PaddlePaddle installation instructions](http://paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/index_en.html))

2. Download the [parameters](https://pan.baidu.com/s/1o7S8yWq) trained on PASCAL VOC0712.

3. `python infer.py`


## train on your own dataset
You can modify the network to adapt to your own application. PaddlePaddle provides a detailed document to show how to train your model with SSD, refer the document [here](https://github.com/PaddlePaddle/models/tree/develop/ssd).
