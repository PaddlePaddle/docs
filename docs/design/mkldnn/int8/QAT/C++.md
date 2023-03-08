##  How to reproduce the results using C++ approach (EnableMkldnnInt8 and enable_mkldnn_int8)

Tutorial how to reproduce the results for old Python Quant2 approach can be found [here](./Python.md)


From release **Release 2.3.1** there is new approach to run QAT models with OneDNN.
Main idea is to run just one API function `EnableMkldnnInt8` from `AnalysisConfig`.

The steps below show, taking ResNet50 as an example, how to reproduce the above accuracy and performance results for Image Classification models.

### Prepare dataset

Download the dataset for image classification models benchmarking by executing:


```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

### Download models

Run the following commands to download and extract Quant model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QUANT_MODEL_NAME=ResNet50
export QUANT_MODEL_ARCHIVE=${QUANT_MODEL_NAME}_qat_model.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${QUANT_MODEL_ARCHIVE}
mkdir ${QUANT_MODEL_NAME} && tar -xvf ${QUANT_MODEL_ARCHIVE} -C ${QUANT_MODEL_NAME}
```

To download other Quant models, set the `QUANT_MODEL_NAME` variable in the above commands to one of the values: `ResNet50`, `ResNet101`, `MobileNetV1`, `MobileNetV2`, `VGG16`, `VGG19`.

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


### Model convertion

To run this quantiozation approach, first you need to set `AnalysisConfig` first and use `EnableMkldnnInt8` function that converts fake-quant model to INT8 OneDNN one.
Examples:

> C++
```C++
AnalysisConfig cfg;
cfg.SetModel(model_path);
cfg.SwitchIrOptim(true);
cfg.SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
cfg.EnableMKLDNN();
cfg.EnableMkldnnInt8();
```

> Python
```Python
config = AnalysisConfig(model_path)
config.switch_ir_optim(True)
config.set_cpu_math_library_num_threads(num_threads)
config.enable_mkldnn()
config.enable_mkldnn_int8()
```

There is available an option to set operators that should be quantized. You can pass a list of strings that will represent operators name to quantize as an argument to `EnableMkldnnInt8` function. Example:
```C++
cfg.EnableMkldnnInt8({"conv2d", "fc", "matmul"});
```

### Performance benchmark commands

To reproduce the performance results, the environment variable `OMP_NUM_THREADS=1` and `--batch_size=1` option should be set.
For image classification models you can use `paddle/fluid/inference/tests/api/analyzer_quant_image_classification_tester.cc` tester with flag `enable_quant_int8=True`.


   ```bash
   cd /PATH/TO/PADDLE/build
   OMP_NUM_THREADS=1 paddle/fluid/inference/tests/api/test_analyzer_quant_image_classification ARGS --enable_fp32=false --with_accuracy_layer=false --int8_model=/PATH/TO/SAVED/QUANT/INT8/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1 --enable_quant_int8=True
   ```
