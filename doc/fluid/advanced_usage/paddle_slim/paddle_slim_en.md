# Model Compression Toolset

<div align="center">
  <h3>
      Model Compression Tool Library
    <span> | </span>
    <a href="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md">
      Introduction to Algorithmic Principle
    </a>
    <span> | </span>
    <a href="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md">
      Use Document
    </a>
    <span> | </span>
    <a href="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/demo.md">
      Example Document
    </a>
    <span> | </span>
    <a href="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/model_zoo.md">
      Model Zoo
    </a>
  </h3>
</div>

## Introduction

PaddleSlim is a sub-module of PaddlePaddle, which is initially released in the 1.4 version PaddlePaddle. PaddleSlim has implemented mainstream compression strategies including pruning, quantization and distillation, which are mainly applied to compression of image processing models. In subsequent versions, more compression strategies will be incorporated to support models in NLP field.

## Main Features

Paddle-Slim tool library has the following features:

###  Simplified Interface

- Manage configurable parameters intensively by the method of configuring files to facilitate experiment management.
- Model compression can be realized by adding merely a few codes to common scripts of training models.

Refer to [Use Examples](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/demo.md) for more details

### Good Performance

- For MobileNetV1 model with less redundant information, the convolution core pruning strategy can still reduce the size of the model and maintain as little accuracy loss as possible.
- The distillation strategy can increase the accuracy of the origin model dramatically.
- Combination of the quantization strategy and the distillation strategy can reduce the size ande increase the accuracy of the model at the same time.

Refer to [Performance Data and ModelZoo](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/model_zoo.md) for more details

### Stronger and More Flexible functions

- Automate the pruning and compression process
- The pruning and compression strategies support more network structures
- The distillation strategy supports multipy ways, and users can customize the combination of losses
- Support rapid configuration of multiple compression strategies for combined use

Refer to [Use Introduction](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md) for more details.

## Introduction to Framework

The overall principle of the model compression tool is briefly introduced here to make you understand the use process.
**figure 1** shows the architecture of the model compression tool, which is a API-relying relation from top to bottom. The distillation, quantization and pruning modules all rely on the bottom paddle framework indirectly. At present, the model compression tool serves as a part of PaddlePaddle framework, so users who have installed paddle of ordinary version have to download and install paddle supporting model compression again for the compression function.

<p align="center">
<img src="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/images/framework_0.png?raw=true" height=252 width=406 hspace='10'/> <br />
<strong>figure 1</strong>
</p>

As **figure 1** shows, the top purple module is user interface. When calling the model compression function in Python scripts, we only need to construct a Compressor object. Please refer to [Use Document](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md) for more details.

We call every compression algorithms compression strategies, which are called during the process of training models iteratively to complete model compression, as is shown in **figure 2**. The logic is packaged in the tool for model compression, and users only need to provide network structures, data and optimizers for model training. Please refer to [Use Document](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md) for more details.

<p align="center">
<img src="https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/images/framework_1.png?raw=true" height=255 width=646 hspace='10'/> <br />
<strong>figure 2</strong>
</p>

## Function List


### Pruning

- Support sensitiveness and uniform ways
- Support networks of VGG, ResNet and MobileNet types
- Support users to customize the pruning range

### Quantification training

- Support dynamic and static quantization training
  - Dynamic strategy: count the activated quantizaiton papameters dynamically in inference process
  - Static strategy: use the same quantization parameters counted from training data for different inputs in inference process
- Support overall and Channel-Wise quantization for weights
- Support to save models in the format compatible with Paddle Mobile

### Distillation

- Support to add combined loss in any layer of the teacher network and the student network
  - Support FSP loss
  - Support L2 loss
  - Support softmax with cross-entropy loss

### Other Functions

- Support to manage hyper-parameters of compression tasks by configuration files
- Support to use combination of multiple compression strategies

## Brief Experiment Results

Some experiment results of the PaddleSlim model compression tool library are listed in this section. To download more experiment data and pre-training moels, please refer to [Specific Experiment Results and ModelZoo](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/model_zoo.md)

### Quantification Training

The data set to assess the experiment is the ImageNet 1000 classes of data, and the top-1 accuracy is token as measurable indicator:

| Model | FP32| int8(X:abs_max, W:abs_max) | int8, (X:moving_average_abs_max, W:abs_max) |int8, (X:abs_max, W:channel_wise_abs_max) |
|:---|:---:|:---:|:---:|:---:|
|MobileNetV1|89.54%/70.91%|89.64%/71.01%|89.58%/70.86%|89.75%/71.13%|
|ResNet50|92.80%/76.35%|93.12%/76.77%|93.07%/76.65%|93.15%/76.80%|

### Convolution Kernel Pruning

data: ImageNet 1000 classes

model: MobileNetV1

size of the original model: 17M

original accuracy（top5/top1）: 89.54% / 70.91%

#### Uniform Pruning

| FLOPS |Model Size| Accuracy Loss（top5/top1）|Accuracy（top5/top1） |
|---|---|---|---|
| -50%|-47.0%(9.0M)|-0.41% / -1.08%|89.13% / 69.83%|
| -60%|-55.9%(7.5M)|-1.34% / -2.67%|88.22% / 68.24%|
| -70%|-65.3%(5.9M)|-2.55% / -4.34%|86.99% / 66.57%|

#### Iteration Pruning based on Sensitiveness

| FLOPS |Accuracy（top5/top1）|
|---|---|
| -0%  |89.54% / 70.91% |
| -20% |90.08% / 71.48% |
| -36% |89.62% / 70.83%|
| -50% |88.77% / 69.31%|

### Distillation

Data: ImageNet 1000 classes

Model: MobileNetV1

|- |Accuracy(top5/top1) |Benifits(top5/top1)|
|---|---|---|
| Single Training| 89.54% / 70.91%| - |
| ResNet50 Distillation Training| 90.92% / 71.97%| +1.28% / +1.06%|

### Combination Experiment

Data: ImageNet 1000 classes

Model: MobileNetV1

|Compression Strategy |Accuracy(top5/top1) |Model Size|
|---|---|---|
| Baseline|89.54% / 70.91%|17.0M|
| ResNet50 distillation|90.92% / 71.97%|17.0M|
| ResNet50 distillation training + quantification|90.94% / 72.08%|4.2M|
| pruning-50% FLOPS|89.13% / 69.83%|9.0M|
| pruning-50% FLOPS + quantification|89.11% / 69.70%|2.3M|

## Export Format of Models
the compression framework support to export models of following format:

- **the model format of Paddle Fluid:** can be loaded and used by Paddle framework.
- **the model format of Paddle Mobile:** can only be used in quantification training strategy situation, compatible with [Paddle Mobile](https://github.com/PaddlePaddle/paddle-mobile) model format.
