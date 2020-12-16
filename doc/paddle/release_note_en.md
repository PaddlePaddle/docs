# Release Note

## Release note

The Paddle framework 2.0-RC1 version has the following updates:

- **Installation environment** Official release of the binary package supporting CUDA11(experimental) ; Official release of the binary package supporting Baidu Kunlun chip (experimental)
- **API function** Support numpy-compatible `paddle.Tensor` indexing and slicing operations(basic indexing); removes the axis parameter in some APIs, support numpy-compatible broadcast semantics; add some new APIs, improve some APIs' functions, and fix some API bugs
- **Dynamic to static conversion** Support more python syntax for dynamic to static graphs, and support for marking functions that do not perform dynamic to static conversion by running `paddle.jit.not_to_static`
- **Framework function** Support multiple executions of `paddle.Tensor.backward()` to accumulate the gradient. The effect is equivalent to the gradient calculated after increasing the batch size. By default, the C++ error stack is hidden, and the error reporting format is optimized. The distributed training supports the heterbox training
- **Framework performance** The mixed precision training supports pure FP16 mode. The ResNet50 model V100 single card training performance reaches up to 1400+ samples/sec. The performance of the distributed training is optimized

## Forward-looking preview
- The Paddle Framework plans to drop the support for python2 and python3.5 from a certain version in the future. It is recommended that you upgrade python to V3.8 for Paddle
- The Paddle Framework plans to drop the support for CUDA 9.0 from a certain version in the future. It is recommended that you upgrade the CUDA for Paddle

##  Training framework

### Basic API (including the distributed)

#### New APIs
- Add the paddle.log2
- Add the paddle.log10
- Add the paddle.nn.initializer.set_global_initializer
- Add the paddle.median
- Add the paddle.broadcast_shape. You can calculate the shape of two tensor shapes after broadcast calculation
- Add the paddle.vision.ops.deform_conv2d, paddle.vision.ops.DeformConv2d
- Add the paddle.subtract
- Add the paddle.optimizer.lamb
- Add the Tensor related APIs, Tensor.cpu, Tensor.cuda(idx), Tensor.pin_memory, Tensor.is_leaf, Tensor.clone


#### Fix and improve APIs
- In the paddle.multiply, remove the axis
- In the paddle.pow, remove the type promotion
- The paddle.add, paddle.subtract, paddle.multiply, paddle.divide, paddle.matmul, paddle.reshape, paddle.transpose, paddle.kron, paddle.trace, and paddle.sum support complex64 and complex128 data types
- Remove the axis parameter from the paddle.maximum and paddle.minimum
- In the multiplex, support the dynamic graphs
- In the CrossEntropyLoss, add the soft_label and axis, modify shape and improve performance
- The paddle.nn.functional.interpolate size parameter supports the input in the Tensor format
- In the paddle.nn.functional.pad, add the padding for N and C dimensions in constant mode
- In the paddle.optimizer.momentum, support the resume training
- Fix the error when converting a BatchNorm to a SyncBatchNorm using paddle.nn.SyncBatchNorm.convert_sync_batchnorm after specifying the weight_param name before conversion
- paddle.to_tensor supports direct input of other Tensor's place when selecting devices
- Optimize the performance of Tensor.detach, share memory with the original Tensor, reduce one memory copy, without keeping in the original computational graph
- In static graph mode, add the acquisition of the learning rate by paddle.optimizer.get_lr()
- Fix the exceeding-range ID error exception in the use of GPU in the paddle.Embedding


####  Remove API (including aliases)
- Remove the api under complex module: paddle.complex.matmul, paddle.complex.reshape, paddle.complex.transpose, paddle.complex.kron, paddle.complex.trace, paddle.complex.sum, paddle.complex.elementwise_add, paddle.complex.elementwise_sub, paddle.complex.elementwise_mul, paddle.complex.elementwise_div
- Remove the sigmoid_cross_entropy_with_logits in the paddle.nn.functional


### High-level API
- Add api paddle.callbacks.ReduceLROnPlateau
- Add api paddle.callbacks.LRScheduler
- Add api paddle.vision.datasets.FashionMnist
- In the paddle.io.DataLoader, change the places parameter to an optional parameter. When the default value is None, paddle.CPUPlace() or paddle.CUDAPlace(0) is automatically selected, and the places parameter will be deleted in later versions
- paddle.io.DataLoader supports disabling the DataLoader automatic group batch function by setting batch_size=None
- Add the api paddle.io. ComposeDataset for stitching multiple datasets into one dataset by field
- Add the api paddle.io. ChainDataset to integrate multiple datasets into one dataset by sample
- Add the api paddle.io. WeightedRadnomSampler for random sampling with the specified weights
- Add the api paddle.vison.ops.yolo_loss and paddle.vision.ops.yolo_box
- Add the api paddle.flops
- Add the api paddle.callbacks.EarlyStopping
- Update the api model.save. The saved file format is consistent with the bottom
- Fix the bug of saving prediction model when input dtype in the api dynamic graph is non-float32 and inputs are not provided in the Model initialization
- The paddle. metric. Accuracy supports input multi-dimensional Tensor, supports the label whose rank is 1 and the label represented by one-hot


### Function optimization (including distributed)
#### Dynamic graph basic functions
- Support Tensor and Scalar for correct type improvement when using operators for operations
- Fix the bug of the interference with each other in the switching between multiple model train/eval models.Dynamic graph Layer.eval() is decoupled from no_grad, Tracer will not automatically record the reverse after calling Layer.eval() before the change, but will still automatically record the reverse after calling Layer.eval() after the change. If the reverse is needed, you can use paddle.no_grad
- Support the change of Tensor data by index or slice
- Add inplace reverse detection module to detect whether the forward inplace operation will affect the correctness of the gradient calculation
- Add that in the Tensor.backward() automatic derivation, the gradient will be added to the previous gradient. This can increase the "batch_size"
- Enabled SE-ResNext oneDNN dygraph training


#### Dynamic graph to static graph

**New syntax**

- Add the support for using the isinstance syntax in the dynamic to static loop
- Add the support for dynamic to static syntax for assigning shape to tuples, such as a, b, c, d = tensor.shape
- Python's and/or statements have sequential execution of the left and right operands. If the result of the left operation can determine the logical value, the right operand will not be executed.In the past, logical_and/logical_or in dynamic to static graphs had problems in handling this case.This support is added
- Add the support for the case where the function signature contains **kwargs
- Support the use of jit.not_to_static decorative function. The function is not converted in the dynamic to static process
- Support python dictionary syntax dict.pop()

**Bug fixing**

- Fix the bug of model storage failure when a variable representing drop_state is not initialized in the dynamic to static storage lstm interface
- Fix the bug of nested loops in the variable analysis
- Fix the bug of return in some special cases
- Fix the bug of if-else in the handling of list generation and variable analysis
- Fix the bug of iterative variables in some special cases
- Fix the bug of inconsistent behavior of transpose API in dynamic and static graphs, and make it support dynamic to static
- Fix the bug of inconsistent behavior of concat API in dynamic and static graphs, and make it support dynamic to static
- Optimize some dynamic to static error messages, so that the error location is more accurate
- Fix the bug that convert_call will be repeatedly called recursively under special circumstances
- Fix the dynamic to static bug caused by different judgments of out.dtype in 2.0 API
- Fix the bug that x.shape == y.shape is judged to be equal to list in the dynamic graph and returns True/False, but will be re-loaded to elementwise in the static graph, and the elementwise result will be reduced after such conversion to static graph
- Fix the bug that param_guard does not cover hook
- Fix the bug of having some parameter variables in the init running in the static graph can not be assigned because the type is not static graph variables
- Fix the bug of the value of non-parameter type variables being defined by users in \__init__ function cannot be modified and updated correctly
- Fix the bug of wrongly converting third-party library logging in the dynamic to static process
- Fix the bug of incorrect transcription of AST in the for-enumerate syntax
- Fix the bug that some warning information is displayed multiple times in a loop

#### Mixed precision training
- Support more aggressive FP16 training mode (i.e., pure FP16 training).To ensure the convergence of the model in Momentum optimizer, add the new `multi_precision` and `rescale_grad` attributes. The `multi_precision` mainly indicates that the optimizer needs to maintain a copy of master weights
- Use the pure FP16 training. The ResNet50 model can reach 1400+ samples/sec on a single card with 16GB video memory on V100

####  Model quantization
- Dynamic graph quantization supports skip to specify the Layer
- Dynamic graph quantization supports 2.0 API Conv and Linear

####  Distributed training optimization

- Support the distributed low-order APIs such as `all_gather` using `paddle.distibuted.spawn` interface
- Support the heterbox heterogeneous training
- Pipeline supports Executor.run interface in parallel to improve the usability
- Launch interface is upgraded, support for specifying the number of processes of a single node
- Sharding supports multi-card training for 10 billion parameter models


#### Model saving and loading

- Support multiple methods declaring that Layers overridden by `paddle.jit.to_static` can still be loaded by `paddle.jit.load` after being stored by `paddle.jit.save`, and multiple methods overridden by `paddle.jit.to_static` can still be used
- Support that Layers loaded by `paddle.jit.load` can still be stored correctly by `paddle.jit.save` after fine-tune or used as sub-Layers of other Layers
- Expand `paddle.jit.save` to support storing the `paddle.DataParallel` model
- Optimize `paddle.static.load_program_state` interface experience. In the scenarios that do not specify to load `var_list`, only a warning is given when loading a directory with interfering files and no error is reported
- Support `paddle.jit.save` to handle InputSpec of dict type
- Support `paddle.onnx.export` to export dynamic model to ONNX file type


#### Performance optimization (including the distributed)
- Improve the performance of RNN class OP on CPU (LSTM, GRU, SimpleRNN). Compared with version 2.0-rc, the forward performance and backward performance of the LSTM, GRU, SimpleRNN have been significantly improved
- Optimize the FastThreadedSSAGraphExecutor scheduling. Fix the performance of the 4-engine 32-card resnet50 that is improved by about 0.3% in the communication synchronization scenario without the overlapping of the communication calculation
- Optimize the paddle. fleet amp distributed performance. Fix the performance of the 4-engine 32-card fp16 that is improved by about 0.5% in the case that the last communication and calculation are not overlapping
- Optimize the performance of the distributed communication component Communicator. In the GEO-400 mode, the W2V model throughput rate, Simnet-Bow model performance have been significantly improved. In the Async mode, compared to the Paddle Framework 1.8, the throughput rate of W2V model is improved by 11% and the performance of CTR-DNN model is improved by 14%
- Optimize the performance when the Worker is a GPU device in parameter server mode, reduce the copy time of Embedding table query. Significantly improve the training throughput rate in the CTR-DNN model
- The distributed GPU dynamic graph realizes the computation and communication overlap, and support the user fine-grained configuration of gradient fuse group size and other options. On the two models ResNet152 and Bert, the multi-node performance improvement is more than 5%.The performance of the ResNet50 is also improved by more than 3%
- Improve the performance of cumsum on GPU
- mproved performance of Resnet50 oneDNN dygraph training. Currently Resnet50 oneDNN drgraph training is 6.4X faster than Native CPU training
- Add the support of cudnn on the GRU and SimpleRNN


#### Debug analysis

- Optimize the alignment of the error exception type on the Paddle Python side with Python native error type
- Hide the C++ error stack by default, optimize the error reporting format after hiding the C++ stack, remove the demarcation flag `Error Message Summary`, and align with the native Python error reporting format
- Optimize some static module APIs in non-static graph mode, including 9 APIs such as static.append_backward, static.gradients, static.scope_guard, static. Print, static.nn.embedding, static.nn.data_norm, static.nn.multi_box_head, static.nn.nce, and static.nn.py_func
- Optimize the error message when the pass-in Tensor is None under the dynamic graph model
- Further optimize the print tensor format of the dynamic graph


### Compile and install

#### New support
- (experimental) Release the binary package supporting cuda11
- Mirror the Paddle of cuda10.1 or later and NCCL to version 2.7.8 in the CI system images
- Release the binary package supporting xpu
- Release the binary package supporting jetpack and C++ prediction library supporting nv_jetson

#### Experience optimization
- Fix the build strategy, separately release the gpu package containing tensorrt, to avoid the error of no tensorrt when users install other GPU versions of the package
- Remove installation dependencies: scipy, rarfile, prettytable, pathlib
- Installation documentation optimization


### Bug fixing

- Fix the bug that GPU card 0 occupies more video memory than other cards during multi-card training
- Fix the bug of wrong shape derivation in the tile op calculation
- Fix the bug of the large number of warning messages of invalid escape sequence in the use of paddle
- Fix the bug when paddle. full is set to INF, NAN, NINF, etc.
- Fix the bug that multiple-nccl comm settings of paddle. fleet do not take effect, and add the non-overlapping warning of multi-nccl comm communication in synchronous mode
- Fix the bug that the paddle. framework.seed in TruncatedNormal initialization does not meet the expectation
- Fix the inconsistent behavior of AvgPool related API dynamic to static exclusive parameters; fix the MaxPool related API ceil_mode transmission parameter problem
- Fix the bug that paddle. topk result is incorrect under GPU
- option in the fluid.layers.nn.gather dynamic graph API
- Fix the bug that the Window-based terminal does not recognize CUDA_VISIBLE_DEVICES as null character, and the frame can be executed in CPU mode by setting the null string
- Fix the bug that the recursive saving and loading of optimizer.state_dict/set_dict fails when LinearLrWarmup recursively contains Learning Rate Scheduler
- Fixed the ptb lm training performance decrease issue
- Fix the bug of gradient calculation when softmax_with_cross_entropy uses ignore_index
- Fix the bug that the parameter to be decayed is empty in the second acquisition after the first execution of AdamW


## Inference

###  Paddle Inference


#### Function upgrade
- In Paddle V2.0, add or upgrade some operators. Starting from this version, the forward operator versioning rules are defined by compatibility constraints. Through the alignment of operator versions between frameworks, ensure consistent definition and behavior of the same operator version in different frameworks, thus enhancing the overall robustness of the framework
- Add the TryShrinkMemory interface to reduce the application display/memory consumption by releasing temporary tensor. For the demo example, refer to [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/test/shrink_memory)
- Paddle-TRT supports clip op. Support the classification model GhostNet running under Paddle-TRT
- Paddle-TRT int8 prediction support models containing channelwise quantization of mul op. Support the PaddleOCR detection and recognition of PaddleSlim quantization model running under Paddle-TRT int8
- `load_inference_model` and `save_inference_model` APIs are migrated to `paddle.static` to improve ease of use and compatibility with old interfaces
- Add six APIs such as `serialize_program`, `deserialize_program`, `serialize_persistables`, `deserialize_persistables`, `save_to_file`, `load_from_ file` six APIs for users to perform serialize/deserialize program, serialize/deserialize params, and save models/parameters to file, or load models/parameters from files
- Enabled BF16 inference for models: resnet50, googlenet, mobilenetv1 and mobilenetv2
- Added oneDNN operators version compatibility support

#### Performance optimization
- When TenorRT is enabled, ERNIE models add the support for variable-length inputs, resulting in the performance improving by 147%.In software versions cuda10.1, cudnn 7.6, tensorrt 6.0, [OSS 7.2.1](https://github.com/NVIDIA/TensorRT/tree/7.2.1), model ernie-base-2.0, dataset QNLI, the performance on Nvidia Telsa T4 improves from 905 sentences/s to 2237 sentences/s when input BatchSize = 32.Example code: [Paddle-Inference-Demo/c++](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++)
- Improved oneDNN INT8 GRU performance. The GRU INT8 model has 1.65X speed-up compared with NativeConfig inference. (with thread=1, batch_size=50)
- Added oneDNN batchnorm + activation fuse, hence improved pvanet_ocr model performance by 2.8%


#### Bug fixing
- Fix the bug that models with avg pooling or global pooling have wrong computation results, error popups or hang
- Fix the bug that the shape of TensorRT subgraph output Tensor ended with x1 will be deleted incorrectly when using the TensorRT dynamic shape inference
- Fix the bug that config.pass_builder()->DeletePass() is not effective when the TensorRT inference is used
- Fix the issue that some models performance depends on the matmul ops' weights
- Fix the issue that CPU oneDNN predictin many models will report error or cause performance regression

## Model upgrade

### PaddleDetection
- Upgrade dynamic graph models:
  - Faster RCNN, Faster FPN, Mask RCNN, Mask FPN, Cascade RCNN, Cascade Mask, YOLOv3 model accuracy flattening static graphs
    - Support the dynamic to static function. Enable the Paddle Inference. The precision speed flattens the static graphs
- Release the SOLOv2, a real-time instance segmentation model. Compared to competing models, it is improved by 2.4% in accuracy and 31.2% in prediction speed. The training speed is as fast as 2.4 times of the competing models
- Add the Android mobile detection demos, including SSD and YOLO series models
- Add the PACT new quantification strategy. Compared to the ordinary quantification, YOLOv3-Mobilenetv3 on COCO dataset is improved by 0.7%

### PaddleSlim

- Support the dynamic graph compression function
  - Add the dynamic graph cropping and quantization training function
  - Add the cropping of the channel quantity alignment function, so that the output model is more easily accelerated by the prediction library
  - PACT quantization training method is changed to built-in method. It is convenient for users to call directly
- Add the OFA model compression technology. The TinyERNIE is accelerated by 40% after compression, with no loss of accuracy

### PaddleSeg

-  Newly release 1.0-rc version, fully upgraded to dynamic graph. It supports 13 segmentation models, 4 backbone networks, and 3 datasets:
  - Segmentation models: ANN, BiSeNetV2, DANet, DeeplabV3, DeeplabV3+, FCN, FastSCNN, Gated-scnn, GCNet, OCRNet, PSPNet, UNet, and U^2Net
  - Backbone networks: ResNet, HRNet, MobileNetV3, and Xception
  - Datasets: Cityscapes, ADE20K, and Pascal VOC
  - Loss: CrossEntropy Loss、BootstrappedCrossEntropy Loss、Dice Loss、BCE Loss
- Provide 40+ high quality pre-trained models based on Cityscapes and Pascal Voc datasets
- Support multi-card GPU parallel evaluation. This provides the efficient index calculation function. Support multiple evaluation methods such as multi-scale evaluation/flip evaluation/sliding window evaluation

### PaddleClas

- Newly released 2.0-rc1, fully upgraded to dynamic graph. It supports 23 series of classification network structures and 135 image classification pre-training models. Among them, 14 practical SSLD distillation models are included, and the effect is generally improved by more than 3% compared with the benchmark model. Three new series of ResNeSt, RegNet and GhostNet models are added
- Based on dynamic graph, provide the mixed precision training method and DALI-based training method
- Provide the off-line predictive deployment, service-oriented deployment and end-side deployment based on the dynamic graphs

### PaddleOCR

- Newly released 2.0-rc1. PP-OCR series models are upgraded to dynamic graphs. Provide 8.1M ultra-lightweight Chinese and English OCR models, universal Chinese and English OCR models and better multilingual recognition models (pure English numbers, French, German, Japanese, Korean). Support the offline predictive deployment and service-oriented deployment
- Release the Style-Text universal text data synthesis tool
- Release the PPOCRLabel text data annotation tool

### PaddleRec

- Release models: gru4rec, deepfm, mmoe, dnn, LR supporting dynamic graph

### PaddleGAN

- Release models: Pixel2Pixel, CycleGAN, PSGAN, UGATIT, ESRGAN, CGAN, DCGAN
- Provide 10 pre-trained models for style migration, makeup migration, coloring, super score, character and scene animation, etc.

### PaddleNLP

- Release 2.0-beta version: support all-around dynamic graph models; provide the PaddleNLP core library, with deeply integrating with higher-order APIs; support the pip installation; provide developers with best practices in the text domain of PaddlePaddle 2.0.
- Add the text graph learning model ERNIESage, generative pre-training model ERNIE-Gen, open domain dialogue generation model PLATO-2, semantic matching model SentenceTransformer, time sequence prediction model TCN, and so on.
- Enrich the pre-training language models further, including a total of 22 pre-training models such as ERNIE, BERT, RoBERTa, and ELECTRA (containing 11 Chinese pre-training models).
- Add 8 common text task evaluation metrics such as Perplexity, BLEU, Rouge-L, and so on, adapted to the PaddlePaddle 2.0 Metrics API system to improve ease of use.
- Add 25 new datasets for text classification, sequence annotation, machine translation, reading comprehension, and so on, adapted to the PaddlePaddle 2.0 Dataset API system, with fast loading by pressing one key.
- Add the Embedding API function, including 38 Chinese word vectors, supporting fast loading and word granularity semantic distance calculation.

### Parakeet

- Release 2.0-alpha version: provide Parakeet core library; improve Chinese documentation; support pip installation.
- Upgrade the text-to-speech model framework to unify the text front-end interface. The model is fully upgraded to Paddle 2.0 API, including TransformerTTS, Waveflow, Wavenet model, and new Tacotron2 model.
- Provide more reusable networking modules. This facilitates the combination of model flexibly. Optimize the data processing and loading process. This improves the training speed.
- Add the experiment module to standardize the experiment process. This facilitates the experiment management and secondary development. The sample codes for experiments are provided for existing models.

## Utility Component

### PaddleHub
- Release 2.0-rc version: fully migrate the dynamic graph programming mode. It is more convenient for model development and debugging. The finetune interface is more flexible and easy to use.
- Upgrade the visual class task migration learning capability fully, supporting a variety of tasks such as image classification, image coloring, and style migration.
- Upgrade Transformer class models such as BERT, ERNIE and RoBERTa to dynamic graph. Support the Fine-Tune capability for text classification.
- Optimize the Serving capability for service-oriented deployment, supporting multi-card prediction and automatic load balancing. The performance is improved greatly.
- Add the Auto Augment (automatic data augment capability). This allows the efficient search for the proper combination of data augment policies for the datasets.

### X2Paddle
- Release version 1.0.0-rc0: It fully supports PaddlePaddle dynamic graph API.
- Add the PyTorch model conversion: supports the conversion between Tracing and Scripting.
- Add the support of conversion from Caffe/ONNX/Tensorflow to Paddle2.0 dynamic graph.
- Add the Optimizer module, mainly including op fusions and op elimination functions, to improve the readability of the converted model code and the prediction performance of the model.

## [Kunlun hardware](https://cloud.baidu.com/product/kunlun.html)

###  Models adapted to Kunlun hardware
- Resnet50, mobilenetv3, deeplabv3, bertbase, DQN static graphs model adapted to Kunlun hardware  
