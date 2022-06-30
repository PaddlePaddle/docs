# SLIM Quantization-aware training (QAT) for INT8 OneDNN

This document is based on [QAT INT8 MKL-DNN in Paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/contrib/slim/tests), that describes how to use [Paddle Slim](https://paddlepaddle.github.io/PaddleSlim/index.html) to convert a quantization-aware trained model (Quant model) into INT8 MKL-DNN quantized model and run it.

In **Release 1.5**, we have released the first approach to the MKL-DNN-based quantization of Quant models, called Quant1. It enabled the `conv2d` and `mul` INT8 MKL-DNN kernels for Quant trained models (GoogleNet, MobileNetV1, MobileNetV2, ResNet50, ResNet101, VGG16, and VGG19) with 0.05% accuracy diff.

In **Release 1.6**, a new approach was introduced, called Quant2, which adds support for more performance optimizations and more INT8 MKL-DNN kernels. INT8 MKL-DNN models obtained using Quant2 have much better inference performance than using Quant1, with only a little bit bigger accuracy diff.

In **Release 1.7**, a support for [Ernie (NLP) Quant trained model](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn) was added to the Quant2.

In **Release 2.0**, further optimizations were added to the Quant2: INT8 `matmul` kernel, inplace execution of activation and `elementwise_add` operators, and broader support for quantization aware strategy from PaddleSlim.

In **Release 2.2**, a support for `fusion_lstm` INT8 kernel and LSTM model.

In **Release 2.3.0**, a support for more quantizable operator: `depthwise_conv2d`, `elementwise_mul`, `nearest_interp` and `slice`.

In **Release 2.3.1**, rewriting Quant2 Python approach to C++ with adding options to the API `EnableMkldnnInt8` to AnalysisConfig.

In this document we focus on the Quant2 approach only.


## Prerequisites
* PaddlePaddle in version 2.0 or higher is required. For instructions on how to install it see the [installation document](https://www.paddlepaddle.org.cn/install/quick).

* MKL-DNN and MKL are required. The highest performance gain can be observed using CPU servers supporting AVX512 instructions.
* INT8 accuracy is best on CPU servers supporting AVX512 VNNI extension (e.g. CLX class Intel processors). A linux server supports AVX512 VNNI instructions if the output of the command `lscpu` contains the `avx512_vnni` entry in the `Flags` section. AVX512 VNNI support on Windows can be checked using the [`coreinfo`]( https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo) tool.

## 1. Introduction

The procedure for converting the model from fake-quant to OneDNN INT8 model is now implemented in Python and in C++.
- The method in Python is constructed in such a way that the model is first converted with Python script and saved to disk, and then it can be run. A procedure on how to transform an FP32 model into a Quant model supported by the Quant2 Python approach is described in [this document](./Python.md).
- The second implementation in C++ is a new functionality and it is available from **Release 2.3.1**. This approach can be used from the Python and C++ API. The model can be loaded into memory, converted to OneDNN INT8 and launched immediately. A detailed example of how to use it is described in [this document](./C%2B%2B.md).

Both of these approaches converts models exactly the way and this process will be described in the following chapters.

## 2. How to turn a Quant model into an INT8 MKL-DNN model?

A Quant model can be transformed into an INT8 quantized model if it contains enough information about quantization scales for every quantized operator in the graph. The process of quantization is done :
- in Python by the `Quant2Int8MkldnnPass` class,
- in C++ `quant_dequant_mkldnn_pass` and `compute_propagate_scales_mkldnn_pass` passes.

Each of them consists of the following steps.

### Gathering scales

The information about the quantization scales is collected from two sources:

1. the `out_threshold` attribute of quantizable operators - it contains a single value quantization scale for the operator's output,
2.  fake quantize/dequantize operators - they imitate quantization from FP32 into INT8, or dequantization in reverse direction, but keep the quantized tensor values as floats.

There are three types of fake quantize/dequantize operators:

* `fake_quantize_moving_average_abs_max` and `fake_quantize_range_abs_max` - used before quantized operator (e.g. `conv2d`), gather single value scale information for the op's input,
* `fake_dequantize_max_abs` and `fake_channel_wise_dequantize_max_abs` - used after quantized operators, contain scales used for the operators' weights dequantization; the first one collects a single value scale for the weights tensor, whereas the second one collects a vector of scales for each output channel of the weights,
* `fake_quantize_dequantize_moving_average_abs_max` - used after a quantized operator to get the scale value for the op's output; imitates immediate quantization and dequantization.

Scale values gathered from the fake quantize/dequantize operators have precedence over the scales collected from the `out_threshold` attributes.

Notes:

1. As the next steps describe, quantization will be applied later to an optimized FP32 model. It means that quantization scales for inputs and outputs of each quantized operator have to be gathered for tensors which are inputs and outputs of already optimized or fused operators. For example, if a model contains the following sequence of tensors and operators in the graph
   ```... → input1 → conv2d → output1 → batch_norm → output2 → relu → output3 → ...```
   and we want to quantize the `conv2d` op, then after applying FP32 optimizations the sequence will become
   ```... → input1 → conv2d → output3 → ...```
   and the quantization scales have to be collected for the `input1` and `outpu3` tensors in the Quant model.
2. Quantization of the following operators is supported: `conv2d`, `depthwise_conv2d`, `mul`, `fc`, `matmul`, `pool2d`, `reshape2`, `transpose2`, `concat`.
3. The longest sequence of consecutive quantizable operators in the model, the biggest performance boost can be achieved through quantization:
   ```... → conv2d → conv2d → pool2d → conv2d → conv2d → ...```
   Quantizing single operator separated from other quantizable operators can give no performance benefits or even slow down the inference:
   ```... → swish → fc → softmax → ...```

### Removing fake operators

All the `fake_quantize_*` and `fake_dequantize_*` operators are being removed from the graph.

### Dequantizing weights

Weights of `conv2d`, `depthwise_conv2d` and `mul` operators are assumed to be fake-quantized (with integer values in the `int8` range, but kept as `float`s) in Quant models. Here, the information about the scale from `fake_dequantize_max_abs` and `fake_channel_wise_dequantize_max_abs` operators is used to fake-dequantize the weights back to the full float range of values. At this moment the model becomes an unoptimized clean FP32 inference model.

### Optimizing FP32 graph

A series of standard optimization passes are being applied to the FP32 graph. This gives us an optimized FP32 inference model and we can proceed with INT8 quantization.

### Computing weight scales

After optimization fuses, the weight tensors of `conv2d` or `fc` operators are likely to have different values and require new quantization scales. The weights are static, i.e. they do not change during the inference process, and the scales can be calculated simply as a maximum of absolute values from the tensor. To improve the inference accuracy we calculate the scales for each output channel separately, getting an array of quantization scales for a weight tensor.

### Taking activations into account

The basic datatype used during INT8 inference is signed INT8, with possible values from -128 to 127. However, if `conv2d` or `fc` operator has `relu` or `relu6` activation integrated in it, the output of the operator is known to have non-negative values. In that case we use unsigned INT8 datatype for output tensors, with a wider range for positive values (0 to 255), improving the inference accuracy further.

### Propagation of scales

Some of the operators (e.g. `reshape2`, `transpose2`, `pool2d` with max pooling) transform the data without changing the quantization scale. For this reason we propagate the quantization scale values through these operators without any modifications. We propagate the quantization scales also through the `scale` operator, updating the quantization scale accordingly. This approach lets us minimize the number of fake quantize/dequantize operators in the graph, because the information about the scales required for the quantization process to succeed spreads between quantized operators.

### Applying quantization passes

Having gathered all the data needed for quantization we apply the `cpu_quantize_pass` which quantizes the graph, and the `cpu_quantize_squash_pass` which optimizes the INT8 graph.
