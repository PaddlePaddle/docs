# Performance test of TensorRT library

## Test Environment
- CPU:Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz GPU:Tesla P4
- TensorRT4.0, CUDA8.0, CUDNNV7
- Test model ResNet50，MobileNet，ResNet101, Inception V3.

## Test Objects
**PaddlePaddle, Pytorch, Tensorflow**   

- In test，subgraph optimization is used to integrate TensorRT in PaddlePaddle.model [address](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models).
- Native implementation is used in Pytorch. Model [address 1](https://github.com/pytorch/vision/tree/master/torchvision/models),[address 2](https://github.com/marvis/pytorch-mobilenet).
- Test for TensorFlow contains test for native TF and TF—TRT.**Test for TF—TRT hasn't reached expectation wihch will be complemented later**. Model [address](https://github.com/tensorflow/models).


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




