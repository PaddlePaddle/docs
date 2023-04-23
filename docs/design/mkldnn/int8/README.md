# Design Doc: oneDNN quantization

There are two approaches to quantization supported in PaddlePaddle: [post-training quantization](./PTQ/README.md) (PTQ) and [quantization-aware training](./QAT/README.md) (QAT). While using PTQ or QAT a user can convert models into INT8 OneDNN models and than run inference on CPU. PTQ is more automatic and requires less model preparation. However, QAT usually gives better accuracy with similar performance.

Both approaches have same list of operators that can be quantized, that is why performance for both procedures give same performance results.

List of operators that can be quantized:
> "concat",
  "conv2d",
  "depthwise_conv2d",
  "fused_conv2d",
  "fused_conv3d",
  "fused_matmul",
  "elementwise_add",
  "elementwise_mul",
  "elementwise_sub",
  "fc",
  "matmul",
  "matmul_v2",
  "nearest_interp",
  "nearest_interp_v2",
  "pool2d",
  "prior_box",
  "reshape2",
  "transpose2",
  "fusion_gru",
  "fusion_lstm",
  "multi_gru",
  "slice",
  "split

## More details about PTQ and QAT:

- ### [post-training quantization](./PTQ/README.md)
- ### [quant-aware training](./QAT/README.md)

## Accuracy and Performance benchmark

This section contain Quant2 MKL-DNN accuracy and performance benchmark results measured on the following server:

* Intel(R) Xeon(R) Gold 6271 (with AVX512 VNNI support),

Performance benchmarks were run with the following environment settings:

* The benchmark threads were assigned to cores by setting

  ```bash
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_BLOCKTIME=1
  ```

* Turbo Boost was set to OFF using the command

  ```bash
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  ```

### Image classification models benchmark results
#### Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 Top1 Accuracy | INT8 Quant Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 Quant Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

#### Performance

Image classification models performance was measured using a single thread. The setting is included in the benchmark reproduction commands below.


>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 (images/s) | INT8 Quant (images/s) | Ratio (INT8/FP32)  |
| :----------: | :-------------: | :-----------------: | :---------------:  |
| MobileNet-V1 |      74.05      |       196.98        |      2.66          |
| MobileNet-V2 |      88.60      |       187.67        |      2.12          |
|  ResNet101   |      7.20       |       26.43         |      3.67          |
|   ResNet50   |      13.23      |       47.44         |      3.59          |
|    VGG16     |      3.47       |       10.20         |      2.94          |
|    VGG19     |      2.83       |       8.67          |      3.06          |

Notes:

* Performance FP32 (images/s) values come from [INT8 MKL-DNN post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) document.

### NLP models benchmark results

#### Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|     Model    |  FP32 Accuracy | Quant INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |      80.20%            |        79.44%        |  -0.76%  |


#### Performance


>**Intel(R) Xeon(R) Gold 6271**

|  Model  |     Threads  | FP32 Latency (ms) | Quant INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:---------:|:---------:|
| Ernie | 1 thread     |       237.21        |     79.26    |   2.99x    |
| Ernie | 20 threads   |       22.08         |     12.57    |   1.76x    |
