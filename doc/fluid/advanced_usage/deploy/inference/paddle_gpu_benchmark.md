# TensorRT库性能测试

## 测试环境
- CPU:Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz GPU:Tesla P4
- TensorRT4.0, CUDA8.0, CUDNNV7
- 测试模型 ResNet50，MobileNet，ResNet101, Inception V3.

## 测试对象
**PaddlePaddle, Pytorch, Tensorflow**   

- 在测试中，PaddlePaddle使用子图优化的方式集成了TensorRT, 模型[地址](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models)。
- Pytorch使用了原生的实现, 模型[地址1](https://github.com/pytorch/vision/tree/master/torchvision/models)、[地址2](https://github.com/marvis/pytorch-mobilenet)。
- 对TensorFlow测试包括了对TF的原生的测试，和对TF—TRT的测试，**对TF—TRT的测试并没有达到预期的效果，后期会对其进行补充**， 模型[地址](https://github.com/tensorflow/models)。


### ResNet50 
 
|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|4.64117 |16.3|10.878|
|5|6.90622| 22.9 |20.62|
|10|7.9758 |40.6|34.36|

### MobileNet
|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1| 1.7541 | 7.8 |2.72|
|5| 3.04666 | 7.8 |3.19|
|10|4.19478 | 14.47 |4.25|

### ResNet101
|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|8.95767| 22.48 |18.78|
|5|12.9811 | 33.88 |34.84|
|10|14.1463| 61.97 |57.94|


### Inception v3
|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|15.1613 | 24.2 |19.1|
|5|18.5373 | 34.8 |27.2|
|10|19.2781| 54.8 |36.7|




