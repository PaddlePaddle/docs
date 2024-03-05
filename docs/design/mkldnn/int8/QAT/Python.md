##  How to reproduce the results using Python approach (Quant2Int8MkldnnPass)

> **_This approach will be deprecated in Paddle new version. [C++ EnableMkldnnInt8](./C%2B%2B.md) option is recommended_**

The steps below show, taking ResNet50 as an example, how to reproduce the above accuracy and performance results for Image Classification models.
To reproduce NLP models results (Ernie), please follow [How to reproduce Ernie Quant results on MKL-DNN](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn/README.md).

### Prepare dataset

Download the dataset for image classification models benchmarking by executing:

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

### Prepare models

Run the following commands to download and extract Quant model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QUANT_MODEL_NAME=ResNet50
export QUANT_MODEL_ARCHIVE=${QUANT_MODEL_NAME}_qat_model.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${QUANT_MODEL_ARCHIVE}
mkdir ${QUANT_MODEL_NAME} && tar -xvf ${QUANT_MODEL_ARCHIVE} -C ${QUANT_MODEL_NAME}
```

To download other Quant models, set the `QUANT_MODEL_NAME` variable in the above commands to one of the values: `ResNet101`, `MobileNetV1`, `MobileNetV2`, `VGG16`, `VGG19`.

Moreover, there are other variations of these Quant models that use different methods to obtain scales during training, run these commands to download and extract Quant model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QUANT_MODEL_NAME=ResNet50_qat_perf
export QUANT_MODEL_ARCHIVE=${QUANT_MODEL_NAME}.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${QUANT_MODEL_ARCHIVE}
mkdir ${QUANT_MODEL_NAME} && tar -xvf ${QUANT_MODEL_ARCHIVE} -C ${QUANT_MODEL_NAME}
```

To download other Quant models, set the `QUANT_MODEL_NAME` variable to on of the values: `ResNet50_qat_perf`, `ResNet50_qat_range`, `ResNet50_qat_channelwise`, `MobileNet_qat_perf`, where:
- `ResNet50_qat_perf`, `MobileNet_qat_perf` with input/output scales in `fake_quantize_moving_average_abs_max` operators, with weight scales in `fake_dequantize_max_abs` operators
- `ResNet50_qat_range`, with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes, with weight scales in `fake_dequantize_max_abs` operators
- `ResNet50_qat_channelwise`, with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes, with weight scales in `fake_channel_wise_dequantize_max_abs` operators

Download clean FP32 model for accuracy comparison against the INT8 model:

```bash
cd /PATH/TO/DOWNLOAD/MODEL/
export FP32_MODEL_NAME=resnet50
export FP32_MODEL_ARCHIVE=${FP32_MODEL_NAME}_int8_model.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/${FP32_MODEL_ARCHIVE}
mkdir ${FP32_MODEL_NAME} && tar -xzvf ${FP32_MODEL_ARCHIVE} -C ${FP32_MODEL_NAME}
```

To download other FP32 models, set the `FP32_MODEL_NAME` variable to on of the values: `Res101`, `mobilenetv1`, `mobilenet_v2`, `VGG16`, and `VGG19`.

### Run benchmark

#### Accuracy benchmark commands

You can use the `quant2_int8_image_classification_comparison.py` script to reproduce the accuracy result of the INT8 Quant models. The following options are required:

* `--quant_model` - a path to a Quant model that will be transformed into INT8 model.
* `--fp32_model` - a path to an FP32 model whose accuracy will be measured and compared to the accuracy of the INT8 model.
* `--infer_data` - a path to the validation dataset.

The following options are also accepted:
* `--ops_to_quantize` - a comma-separated list of operator types to quantize. If the option is not used, an attempt to quantize all quantizable operators will be made, and in that case only quantizable operators which have quantization scales provided in the Quant model will be quantized. When deciding which operators to put on the list, the following have to be considered:
  * Only operators which support quantization will be taken into account.
  * All the quantizable operators from the list, which are present in the model, must have quantization scales provided in the model. Otherwise, quantization of the operator will be skipped with a message saying which variable is missing a quantization scale.
  * Sometimes it may be suboptimal to quantize all quantizable operators in the model (cf. *Notes* in the **Gathering scales** section above). To find the optimal configuration for this option, user can run benchmark a few times with different lists of quantized operators present in the model and compare the results. For Image Classification models mentioned above the list usually comprises of `conv2d` and `pool2d` operators.
* `--op_ids_to_skip` - a comma-separated list of operator ids to skip in quantization. To get an id of a particular operator run the script with the `--debug` option first (see below for the description of the option), and having opened the generated file `int8_<some_number>_cpu_quantize_placement_pass.dot` find the id number written in parentheses next to the name of the operator.
* `--debug` - add this option to generate a series of `*.dot` files containing the model graphs after each step of the transformation. For a description of the DOT format see [DOT]( https://graphviz.gitlab.io/_pages/doc/info/lang.html). The files will be saved in the current location. To open the `*.dot` files use any of the Graphviz tools available on your system (e.g. `xdot` tool on Linux or `dot` tool on Windows, for documentation see [Graphviz](http://www.graphviz.org/documentation/)).


```bash
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/quant2_int8_image_classification_comparison.py --quant_model=/PATH/TO/DOWNLOADED/QUANT/MODEL --fp32_model=/PATH/TO/DOWNLOADED/FP32/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.01 --ops_to_quantize="conv2d,pool2d"
```

> Notes: Due to a large amount of images in the `int8_full_val.bin` dataset (50 000), the accuracy benchmark may last long. To accelerate accuracy measuring, it is recommended to set `OMP_NUM_THREADS` to the maximum number of physical cores available on the server.

#### Performance benchmark commands

To reproduce the performance results, the environment variable `OMP_NUM_THREADS=1` and `--batch_size=1` option should be set.

1. Transform the Quant model into INT8 model by applying the `Quant2Int8MkldnnPass` pass and save the result. You can use the script `save_quant_model.py` for this purpose. It also accepts the option `--ops_to_quantize` with a list of operators to quantize.

   ```bash
   cd /PATH/TO/PADDLE/build
   python ../python/paddle/fluid/contrib/slim/tests/save_quant_model.py --quant_model_path=/PATH/TO/DOWNLOADED/QUANT/MODEL --int8_model_save_path=/PATH/TO/SAVE/QUANT/INT8/MODEL --ops_to_quantize="conv2d,pool2d"
   ```

2. Run the C-API test for performance benchmark.

   ```bash
   cd /PATH/TO/PADDLE/build
   OMP_NUM_THREADS=1 paddle/fluid/inference/tests/api/test_analyzer_quant_image_classification ARGS --enable_fp32=false --with_accuracy_layer=false --int8_model=/PATH/TO/SAVED/QUANT/INT8/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
   ```
