# Release Note

## 2.0.2 Release Note

## Important Updates
This version fixed some function and performance issues of PaddlePaddle 2.0.1, and optimized some function. The important updates are as following:

- Add the `use_softmax` parameter to `paddle.nn.functional.cross_entropy`, which controls whether to perform softmax operation before calculating the cross entropy; add the deprecated mark to `paddle.nn.functional.softmax_with_cross_entropy`, for this API will be deprecated in the future version.
- Fix multiple issues of distributed training in parameter server mode。
- Upgrade Paddle's oneDNN version to 2.2, which improves the inference performance of multiple models.

## Training Framework

### Function Optimization

#### API
- Add `paddle.io.random_split` and `paddle.io.Subset`. ([#32090](https://github.com/PaddlePaddle/Paddle/pull/32090))

### Bug Fixes

#### API
- Fix the issue that the `stride` and `padding` of `paddle.nn.MaxPool3D` and `paddle.nn.AvgPool3D` do not have default values. ([#32014](https://github.com/PaddlePaddle/Paddle/pull/32014))
- Fix the issue that when RNN supporting cudnn creates parameters, repeated creations are reported. ([#31916](https://github.com/PaddlePaddle/Paddle/pull/31916))
- Fix the issue that when the `soft_label` of `paddle.nn.functional.cross_entropy` is True, and the `weight` parameter is specified, an error will be reported; add the `use_softmax` parameter to `paddle.nn.functional.cross_entropy`, which controls whether to perform softmax operation before calculating the cross entropy; add the deprecated mark to `paddle.nn.functional.softmax_with_cross_entropy`, for this API will be deprecated in the future version. ([#31953](https://github.com/PaddlePaddle/Paddle/pull/31953), [#32105](https://github.com/PaddlePaddle/Paddle/pull/32105), [#32035]( https://github.com/PaddlePaddle/Paddle/pull/32035))
- Fix the issue of `paddle.nn.ClipByNorm` generating NaN values as the gradients are all zero, which will lead to non-convergence when using mixed precision training. ([#32038](https://github.com/PaddlePaddle/Paddle/pull/32038))
- Fix the issue of accessing array out of bounds in `paddle.stack`. ([#32005](https://github.com/PaddlePaddle/Paddle/pull/32005))

#### Distributed Training

- Fix the issue that in parameter server mode the calculation graph segmentation supports GradClip strategy.([#31945](https://github.com/PaddlePaddle/Paddle/pull/31945))
- Fix the initialization of truncated gaussian distribution in parameter server mode.([#31945](https://github.com/PaddlePaddle/Paddle/pull/31945))
- Fix the issue of incorrectly printing the Profiler's multi-threaded information in parameter server mode.([#31945](https://github.com/PaddlePaddle/Paddle/pull/31945))
- Fix the Python3 incompatibility issue when data are read by Dataset and output by zip.([#31945](https://github.com/PaddlePaddle/Paddle/pull/31945))
- Clean up redundant log information and optimize the output format of `exe.train_from_dataset`.([#32009](https://github.com/PaddlePaddle/Paddle/pull/32009))

## Inference Deployment

### Paddle Inference

#### Function Upgrades
- Paddle-TRT adapts to the ERNIE/BERT model trained and saved by PaddlePaddle 2.0.([#31959](https://github.com/PaddlePaddle/Paddle/pull/31959))

#### Performance Optimization
- Upgrade onednn to version 2.2, which has improved many models inference performance. ([#31270](https://github.com/PaddlePaddle/Paddle/pull/31270))
- Add hard_swish oneDNN support and conv + hard_swish fusion, which has improved ocr_det model inference performance by 18% on SkyLake. ([#31870](https://github.com/PaddlePaddle/Paddle/pull/31870))

## 2.0.1 Release Note

## Important Updates

This version fixed some function and performance issues of PaddlePaddle 2.0.0, and optimized some function. The important updates are as following:

- The new scheme that operators can be customized outside the framework. The process of customized operators’ writing and inference deployment, is simplified.
- `paddle.save/paddle.static.save` supports users to choose the pickle version, which can improve the efficiency of saving models under Python 3.
- At the stage of inference, users can apply [DLA](http://nvdla.org/) of NVIDIA while using TensorRT.
- PaddlePaddle inference APIs of C++ and Python support XPU, which is aligned with training supported by PaddlePaddle to XPU.


## Training Framework

### Function Optimization

#### API
- Add `aligned` in `roi_align`, and  `pixel_offset` in `generate_proposals、distribute_fpn_proposals` to improve performance.
- `paddle.nn.functional.cross_entropy` supports float type label in XPU accelerator.
- Add label error checks and optimized error message of `paddle.nn.functional.softmax_with_cross_entropy`.
- `paddle.nn.LayerList` supports `paddle.nn.LayerList([None])` .

#### Dynamic Graph to Static Graph
- Add the support of `tuple` as loop variable in for-loop.
- Add `Tensor` support to be indexed by unspecific start and stop variables,  such as `x[:], x[2:]`.
- Now `Tensor` supports slicing with lvalue in static graph. In dynamic graph, `Tensor` uses slicing can correctly turn into static graph. `Tensor` can be modified by indexing or slicing. `Python.Int`、`Tensor`、`Python.slice` can be used for indexing. The stride could be 1, greater than 1 or negative. `NumPy.array`, `Tensor` types could be used as rvalue.

#### Mixed Precision Training
- Mixed precision training of dynamic graph supports `paddle.nn.LayerNorm` , improving efficiency by reducing the number of `cast`.

#### Distributed Training Optimization
- `paddle.distributed.fleet.DistributedStrategy` amp adds pure fp16 strategy.
- `paddle.distributed.ProbabilityEntry` and `paddle.distributed.CountFilterEntry` are added for sparse parameters training.
- Optimized the number of communications in parallel pipeline.
- In parameter server mode, fields like `count/unseen_day` could be saved into model.
- Add the elimination strategy of sparse parameters in parameter server mode.

#### Model Saving and Loading
- `paddle.save` and `paddle.static.save` allow users to select the pickle version, and the default version is 2. For Python 3, users can choose Pickle version 4+. In this way,  saving speed could be increased and single file could over 4G. But, please notice that models saved this way must be loaded and used under Python 3.
- Add `paddle.static.normalize_program` to obtain the pruned computation graph.

#### Complex Number Operation
- `paddle.abs` supports  Complex64 and Complex128 types.

#### Customized Operator
- Offered the new scheme of custom operators outside the framework, simplify the writing and using process of custom operators, support two installation and calling methods, and support Linux and Window at the same time; custom operators by using the new scheme can be used in dynamic graphs, static graphs, dynamic-to-static and inference scenarios; for specific instructions, please refer to the file: [Customizing External Operators](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html).

#### Distributed Training
- Fixed Entry Config has no effect issue in parameter server mode.
- Fixed the saved parameters could not be loaded issue in parameter server mode.
- Fixed Profiler abnormal issue in parameter server mode.
- Fixed training abnormal issue when data type category is higher than INT32 in parameter server mode.
- Fixed long stringIP cannot be bounded issue in parameter server mode.
- Fixed the issue of too much log outputs issue, in distributed training caused by lower level LOG config.
- Fixed the issue of inconsistent parameters of each devices, when if else control flow is used in dynamic graph distributed training.
- Fixed the issue that FLAGS setting of multi-host distributed training is not consistent with single host distributed training.

### Bug Fixes
#### API
- Fixed the `muti_precision` function of `paddle.optimizer.AdamW` to ensure the master weights in FP32 type, which are regularized, in order to prevent possible diverge.
- Fixed the issue when the input of `paddle.nn.ELU` is nan, the output is nan.
- Fixed gradient calculation error of using `Tensor.backward()` for gradient accumulation, in dynamic graph mulit-card training.
- Fixed the integer overflow issue when `paddle.nn.functional.softmax_with_cross_entropy` processes a `Tensor` with over 2^31 elements.
- Fixed crash bug during the for-loop traversal of `paddle.nn.Sequential`.
- Fixed Wrong error message of dynamic graph slicing.
- Fixed the issue that batch_size=-1 cannot be used, when `paddle.nn.functional.local_response_norm` is used in static graph or dynamic graph to static graph converting.
- Fixed `paddle.nn.LayerNorm` computation error when data type is float64.

#### Others

- Fixed the error message of `metric_learning finetune` under PaddlePaddle/models.
- Fixed weight asynchrony issue caused by lack of operators, when XPU's static graph multi-card is used.

## Inference Deployment

### Model Quantification
- Support the quantification inference of TRT, which uses per-layer to quantize.

### Paddle Inference
#### API
- Add API— `paddle_infer::Config::EnableTensorRtDLA()`.  At the stage of inference, users can apply [DLA](http://nvdla.org/) of NVIDIA while using TensorRT.
- Paddle-TRT will check inputs of model, If input shape is variant, the error messages are optimized and Paddle-TRT will hint users to use dynamic_shape.

#### Function Upgrades
- Support inference and deployment models that have the operators customized by users, and provide [User Documentation](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/custom-operator).
- PaddlePaddle inference APIs of C++ and Python support XPU, which is aligned with training supported by PaddlePaddle to XPU.

#### Performance Optimization
- Paddle-TRT supports `group_norm` op, and speed up `solov2_r50_fpn_1x` as following: compared with TRT v2.0.0, on T4, CUDA11, cuDNN8.1 and TRT7.1.3, the performance of TRT FP32 improves by 13%, from 87.019ms to 75.13ms, and the performance of TRT FP16 improves by 65%, from 72.9253ms to 44.149ms.

#### Bug Fixes
- Fix some operator problems in TensorRT v7.1+, like TensorRT’s inference of ERNIE.
- Fix some issues of using Python pass_builder API.
- Due to limited memory of Jetson, `auto_growth` is regarded as default distribution policy of memory,  tackling problems that some models cannot run with limited memory.
- Avoid the problem of cuDNN8.0’s memory leaks to ensure the availability, and this will not influence other versions of cuDNN.
- Fixed MakeCipher’s symbol absence issue in inference dynamic library.
- Fixed wrong predicting results issue of `mask_rcnn_r50_1x_coco` model when this static graph model is converted from dynamic graph.
- Fixed the inference failure of the segmentation models, caused by adaptive pooling is not fully supported by oneDNN,
- Fixed the issue that oneDNN’s OCR model inference will be incorrect when batch_size>1.
- Fixed freeze_model inference failure due to ReLU CPU’s implementation error.
- Fixed the incompatibility issue that BF16’s images cannot change into binary script for Python3.


## Environment Adaptation
### Training Framework

- Upgrade the GCC from V4.8.2 to V5.4 in Paddle docker images of CUDA9.0 and CUDA10.0
- Add the Windows Paddle develop version wheel package. Windows users now can run `pip --pre` for installation.

### Paddle Inference
- Fixed the problem that develop docker image cannot compile with TensorRT, and replace TensorRT7 of powerpc architecture with TensorRT6 of x86-64 architecture.
- Upgrade the name of Paddle Inference library: the name of dynamic link library changes from `libpaddle_fluid.so` to `libpaddle_inference.so`.

## 2.0.0 Release Note

## **Update**

The PaddlePaddle framework V2.0.0 has the following updates:

- Programming Paradigm: Enable dynamic graph mode for model development and training by default, and perform the model deployment and training acceleration through the dynamic to static mode.If you need to use static graph programming paradigm, you can switch to static graph mode by running paddle.enable_static().
- API system: The API has been supplemented and the directory structure has been adjusted to make it easier to use, please see [API documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html) for more details. A high-level API is provided to simplify the process. See [PaddlePaddle High-Level API Usage Guide](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/high_level_api/high_level_api.html) for more details.
- Framework features: Data loading, dynamic graph execution, OP performance, mixed precision training, distributed training, dynamic-static conversion, etc. have been enhanced and optimized.
- Environment adaptation: Supported ARM-based CPU. Added support for Python 3.8, CUDA 10.1/10.2. Released the installation package (experimental) supporting CUDA11, and released the installation package (experimental) supporting [Baidu Kunlun](https://cloud.baidu.com/product/kunlun.html) chip. For details, see [Start](https://www.paddlepaddle.org.cn/install/quick).
- Model zoo and development kits: The official model zoo and kits for PaddlePaddle have been upgraded to PaddlePaddle framework V2.0.0.
  - [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)： Support dynamic graph V2.0. Fully migrate the dynamic graph programming mode, make model development and debugging more convenient. The finetune interface is more flexible and easy to use.
  - [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection): Support dynamic graph V2.0. Cover the mainstream algorithm of detection direction (PP-YOLO, Faster-RCNN, SOLOv2), support dynamic-static conversion, hit the inference deployment, and provide a more modular way of networking.
  - [PaddleClas](https://github.com/PaddlePaddle/PaddleClas): Support dynamic graph V2.0. Provide 29 series of classification algorithms and 134 pre-training models, provide an optimization scheme based on SSLD knowledge distillation, and generally improve the precision of classification models by more than 3%.
  - [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg): Support dynamic graph V2.0. Provide 50+ high quality pre-training models, support 15+ mainstream segmentation networks, and provide the industry's SOTA model OCRNet, which well enhances the usability of the product.
   - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR):  Support dynamic graph V2.0. PPOCR system, text detection models (DB, EAST, SAST) and text recognition models (Rosetta, CRNN, StarNet) , and complete the adaptation of dynamic graph V2.0.
  - [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)：Support dynamic graph V2.0. Nine models, including style migration, video enhancement, lip migration, face animation and others are developed based on dynamic graph.
  - [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)： Support dynamic graph V2.0. The installation-free and unified dynamic and static networking are provided, convenient for user's research and going online. Release the classic dataset of the recommendation system.
  - [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)：Support dynamic graph V2.0. Provide 25+ pre-training models and easy-to-use API way to enhance the efficiency of text modeling.
  - [Parakeet](https://github.com/PaddlePaddle/Parakeet)：Support dynamic graph 2.0. The released acoustic models and vocoder well support dynamic graph version.
  - [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)：Support dynamic graph V2.0. The video classification and video motion positioning direction models are included, such as TSN, TSM, SlowFast, AttentionLSTM, BMN models and featured application pre-training models VideoTag and FootballAction.
  - [AmazonDJL](http://docs.djl.ai/paddlepaddle/index.html): Easy-to-use Java inference interface which supports various operating system platforms (Mac / Windows / Linux) and Paddle pre-training model loading. Please refer to [AmazonDJL](http://docs.djl.ai/paddlepaddle/index.html) for more information.

## **Forward-looking Preview**

- The PaddlePaddle Framework plans to drop the support for python2 and python3.5 from a certain version in the future. It is recommended that you upgrade python to V3.8 for PaddlePaddle.
- The PaddlePaddle Framework plans to drop the support for CUDA 9.0 from a certain version in the future. It is recommended that you upgrade the CUDA for PaddlePaddle.

## **Training Framework**

### **Compatibility instructions**

- Programming paradigm: PaddlePaddle 2.0.0 has the imperative programming paradigm (dynamic graphs) enabled by default, but still retains support for static graphs. static graph code (including static graph code from version 1.8) can be executed by running paddle. enable_static().
- API: The PaddlePaddle Framework Version 2.0.0 recommends users to use the API located in the paddle root directory, while all the APIs from version 1.x are retained in the paddle.fluid directory, retaining support for the API system of earlier versions. Therefore, the static graph training code version 1.x can run normally on version 2.0.0 by running paddle.enable_static(). The model saved by training of version 1.x can be used for inference in version 2.0.0.
- A [table](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_others_information/api_mapping_cn.html) of correspondence from version 1.8 API to version 2.0 API is prepared.
- We provide a migration tool to facilitate the migration of codes based on earlier version to codes of version 2.0.0. See Version Migration Tool.

### **dynamic graph mode**

By default, the dynamic graph mode is enabled for model development and training, which allows you to perform model deployment and training acceleration in the dynamic-to-static mode.For details, please see [dynamic graph](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/dynamic_graph/dynamic_graph.html), [Dynamic-to-static graph](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/index_cn.html).

###  **API system**

- Basic APIs
  - API directory structure adjustment: The API V1.x is mainly located in the paddle.fluid directory. In this version, the API directory structure is adjusted so that the classification can be more reasonable. For the specific adjusted directory, see the [API documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html).  
  - Added 186 new APIs. Fixed and revised 260 APIs. See Release Notes of 2.0.0 pre release version and [API documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html).
  - Added the distributed basic communication class API to paddle.distributed:broadcast, all_reduce, reduce, all_gather, scatter, barrier; dynamic graph multi-card training startup API spawn, init_parallel_ env, dynamic-static unified startup method fleetrun
  -  Networking class API for dynamic and static unification: supports running in both dynamic graph mode and static graph mode.
- High-level API
  - Added PaddlePaddle high-level API, and encapsulated the common operations such as networking, training, evaluation, prediction, access, etc. in the process of model development, to achieve low code development. See [PaddlePaddle high level API instructions](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/high_level_api/high_level_api.html).
  - Added distributed high-level API paddle.distributed.fleet. Supported multiple optimization strategy combinations and automatic parallelism, distributed metrics calculation, InMemoryDataset by configuring the DistributedStrategy.

### **Function optimization (including distributed)**
#### **dynamic graph basic functions**

- Usability optimization:
  - Tensor function enhancements: Added Tensor copy interface Tensor.clone(), and more than 120 Tensor computation interfaces (e.g. Tensor.cos(), etc.). Added the new function to modify the Tensor function by using index or slice. Added the new function of automatic type boost in case of Tensor and Scalar operation. Optimized the printing information of dynamic graph Tensor. The display form remains the same as Numpy.
  - Layer function enhancement: Added the new Layer deep copy interface Layer.deepcopy(). Added the new Layer property and function to view interface Layer.dir(). From this version, the Trace function still records reverse operation automatically after the invoking of Layer.eval(). If you don't need to record reverse, you need to explicitly call paddle. no_grad().
  - Added a set_lr() interface for Optimizer so that users can flexibly adjust a learning rate in dynamic graph mode.
  - Added a new set_global_initializer() interface to define global parameter initialization methods.
 - Simplified the code for multi-card operation without needing to explicitly call scale_loss and apply_collective_grads.
- Performance optimization:
  - Supported the function of gradient updating by using sparse parameters for APIs (for example, embedding) in case of multi-card training.
  - For dynamic graph training and inference, added the support for Intel acceleration library oneDNN (former MKL-DNN). The speed of Resnet50 model in CPU training scenario can improve by 6 times.
  - New dynamic graph Inplace calculation function: The Tensor storage space can be reused, reducing the occupation of video memory. Added the new View method. You can change the Tensor description in case of shared underlying storage.
  - [Incompatible upgrade] new dynamic graph gradient accumulation function, with disguised "expand BatchSize" role. By default, the gradient of backward() interface is not clear, with needing to explicitly call optimizer.clear_grad() to clear the gradient.
- Fixing bugs:
  - Fixed the bug of train and eval interference with each other when switching between them in multiple models.

#### **Dynamic-to-static graph**

- **Added the grammar support for dynamic-to-static conversion**
  - Added the support for the return grammar. In the if-elif-else or loop conditions, the loop can return earlier, with return different types of tensor or None.
  - Added support for the **kwargs parameter contained in the signature function.
  - Added the grammar support of “for”, “for enumerate” traversing Tensor and TensorList, more flexible operation for traversing Tensor.
  - Added the support for more python grammars, such as print, assert, cast, isinstance, tuple, dict.pop(), etc.
- **Optimized the usability of dynamic-static conversion**
  - Changed the return type of dynamic-to-static from callable function to Class. The code and main_program interfaces invoking the Class can obtain the converted static graph information more easily.
  - The dynamic-to-static decorator to_static is added with the directly decorating model instances, such as to_static (model, input_spec).
  - Added the jit.not_to_static decorator. The function is not converted in the dynamic-to-static process.
  - Added set_verbosity() and set_code_level() interfaces. You can set different levels to view the log or intermediate state code of the dynamic to static process.
  - Added InputSpec. You can specify the shape and data type of input Tensor variables in the dynamic to static process.
  - Error message optimization: Locate the specific wrong line of code in the original dynamic graph and hide the user-unrelated error message.
  - Support break point debugging by using pdb.set_trace().
- **Optimized deployment of model storage and loading APIs**
  - Added paddle.jit.save interface for storing dynamic-to-static models: The interface is compatible with and used to store both the Layer object not transcribed by paddle.jit.to_static and paddle.DataParallel models. Remove the old interface ProgramTranslator. save_ inference_model.
  - Added the paddle.jit.load interface for loading prediction models stored in static graph format, including models saved by paddle.jit.save and paddle.io.save_inference_model. This can be used for model inference or model training optimization under dynamic graph after loading.
  - Added the program method for opaddle.jit. TransLatedLayer for obtaining the program of the paddle.jit.load loading model. It is for understanding of the model structure.
  - [Incompatible upgrade] changed the meaning of the interface parameter model_path of paddle.jit.save and paddle.jit.load: That is, changed to the prefix of storage files instead of that of directory.

#### **Mixed precision training**
- Mixed precision policy upgrade: In addition to the black and white list policy (hereinafter referred to as "O1 policy"), "Almost FP16 (hereinafter referred to as O2 policy)" is added. That is, use FP16 for calculation as much as possible.
  - Added the FP16 Guard function (`paddle.static.amp.fp16_guard`): Support users to freely control whether a single Op in the model chooses FP16 calculation type.
  - User can customize `custom_black_list` to control a certain type of Op to keep FP32 computation.
  - Using the O2 policy: Resnet50 and Bert base can be trained at 1400 images/s and 590 sequences/s, respectively, on a single card V100.
- Usability optimization:
  - Use the `paddle.static.amp` package to manage the interfaces related to static graph mixed precision training in a unified manner.
  - Provide the simplified name `CustomOpLists` for `AutoMixedPrecisionLists`: That is, users can customize the AMP black and white list Op list by using `CustomOpLists`.

#### **Optimization of the distributed training**

- Integrated communication All Reduce
  - Support mixed parallel training of 100 billion language models: support pipeline parallel training based on the executor interface, with sharding-DP strategy, GradientMerge+AMP strategy, Recompute+Offload strategy, and megatron strategy.
  - Support dynamic graph: support multi-stream communication strategy, automatic rebuild group strategy, high performance sparse parameter communication, and multi-card gradient sequential consistency strategy.
- Parameter server PS
  - Upgraded the large-scale sparse function: Upgrade large-scale sparse PS-API, and abstract communication component/parameter table/optimizer base class. It is convenient for users to carry out secondary development in a subclass derivation mode. Meanwhile, it also supports 100 billion features streaming training, including feature access, exit, incremental training, distributed metrics prediction, etc. The communication mode switches from GRPC to BRPC.
  - Open source heterogeneous parameter server: Support both traditional pure CPU machine PS, and pure GPU machine PS based on three levels of storage (SSD/memory/video memory). It also supports CPU machine + GPU machine/Kunlun machine mixing distributed PS, with completing the minute-level training of trillions of parameter hit rate prediction models
- Support of new training mechanism:
  - Support control flow-based multitasking distributed training: The performance is improved by more than 50% compared to the Intag-based multitasking.
- Optimization of the distributed startup method
  - Supported distributed low-order APIs such as all_gather using the `paddle.distibuted.spawn` interface
  - Upgraded the `paddle.distributed.launch` interface: Support specifying the number of processes in a single node with simplifying as `fleetrun`.
  - Optimized `gen_nccl_id`: Removed the grpc dependency, added some fault tolerance, and improved the stability of starting distributed tasks.
  - Supported the startup of multi-CPU in the integrated communication in the Gloo method

#### **Model saving and loading**
- Standardized the set_dict method name of APIs such as Layer and Optimzier: That is, changed to set_state_dict in a unified manner.
- Enhanced paddle.load compatibility: support the loading of Layer's state_dict from storage results of interfaces such as fluid.io.save_inference_model and fluid.io.save_params/persistables.
- Modified the paddle. save/load interface behavior: For the paddle.save, A suffix is not added to the storage results. In each loading, paddle.load returns only one result. Standardize the interface semantics.
- Removed paddle.SaveLoadConfig: For the interface compatibility loading scenarios of paddle.jit.save, paddle.jit.load, and paddle.load, use **kwargs to pass in additional configuration to simplify the use of the interface.
- Moved the original static graph APIs such as paddle.io.save, paddle.io.load, paddle.io.save_inference_model, and paddle.io.load_inference_model to the paddle.static module.
- Optimized the paddle.static.load_program_state interface experience. In the scenarios without specifying the loading var_list, only a warning (instead of error report) is given when there is an interference file in the loading of a directory.

#### **Plural computation**

- Extended the dynamic static graph execution engine: Support the plural neural network training and plural gradient accumulation.
-  Added Op such as mul, div, matmul, kron, and abs for supporting the plural computation.

#### **ONNX function upgrade**

- Added API: `paddle.onnx.export` for supporting the conversion from Paddle2.0 dynamic graph to ONNX protocol.
- Added PPOCR, PPYOLO, FasterRCNN, and ERNIE for model conversion.
- Richer Paddle op coverage: Support 88 Paddle OP operators. Support the export as different versions of ONNX 1~12 operator sets.

#### **Performance optimization (including the distributed)**

- dynamic graph performance optimization:
  - Optimized the data read performance: Simplify the DataLoader underlying implementation logic in dynamic graph mode, reduce the thread reading overhead, and further improve the data reading efficiency and the overall model training speed. The overall training speed of MobileNetV1 in a scenario of single card V100 and BatchSize = 128 is improved by 34%.
  - Upgraded and performance optimization of dynamic graph networking API: A large number of dynamic graph APIs directly call an automatically generated Pybind API. As a result, the performance is improved significantly.
  - Improved the training performance of Resnet50 oneDNN dynamic graph. The dynamic graph training speed of the current CPU scenario Resnet50 oneDNN is improved by 6.4 times.
- OP performance optimization:
  - argsort: The number of elements of the input Tensor is optimized as the number equal to its `axis` dimensional length. In this way, the forward speed is improved by 34 times, and the reverse speed is improved by 10 times.
  - dropout: Optimized GPU performance. The FP32 performance is improved by 20%. The FP16 performance is improved by 50%.
  - cast: Optimized GPU performance. The performance is improved by 10% to 20%.
  - softmax: Optimized GPU performance in case of axis=-1. The performance is improved by 3 times to 96 times for different shapes.
  - Performance optimization of other OPs: Significantly improved the performance of other OPs such as cumsum, reshape, Flatten, IndexSelect, Roll, elementwise_add, AdamW and RNN class (LSTM, GRU, SimpleRNN).
- Optimization strategy:
  - Added fused_bn_add_act fusion strategy: Performed the automatic fusion acceleration for the combined pattern of batch_norm+elementwise_add+activation.
  - Added inplace addto strategy for gradient aggregation: Support in-situ gradient accumulation. Improve the performance by 6.3% in ResNet-50 mixed precision training.

- Optimized FastThreadedSSAGraphExecutor scheduling: Fixed the bug that the communication calculation does not overlap in the communication synchronization scenario. The performance of 4 machines and 32 cards resnet50 is improved by about 0.3%.

- Distributed performance optimization:
  - Optimized lars strategy: The time2train index of 16k batch size in the ResNet50 distributed multi-card training is smaller than 10 minutes.
  - Optimized the paddle.fleet amp distributed performance: Fixed the bug that the last communication and calculation are not overlapping. The performance of the 4-machine 32-card FP16 is improved by about 0.5%.
  - Optimized paddle. fleet.gradient_merge distributed performance: Aggregate gradients before communication. The multi-machine performance can be improved by 20%-40% to achieve linear acceleration ratio.
  - Optimized the performance of the parameter server communication component Communicator. In case of GEO-400batch communication once, the W2V model throughput rate and Simnet-Bow model performance are significantly improved. In the Async mode, compared to the PaddlePaddle Framework 1.8, the throughput rate of W2V model is improved by 11% and the performance of CTR-DNN model is improved by 14%

#### **Debugging analysis**

- Uniformly changed the wording of LOG(FATAL) throw exception at just 100 points to PADDLE_THROW: Optimize the error format and content caused by non-support of a framework behavior.
- Improved the Signal Handler implementation within the framework. Optimized the error format and content when system signal error occurs during the execution.
- Optimized the framework error stack format: In the compiling, the python error stack is moved below the native error stack to improve error message reading experience.
- An accumulative total of about 1500 error type and prompt copywritings of check errors within the framework. This enhances the overall debugging usability of the framework.
- Enhanced dynamic graph error messages: Error messages on the Pybind layer under a dynamic graph are systematically enhanced to improve user experience.
- Optimized exception types of Paddle Python side error report: Align with Python native error report types.
- Hide the C++ error stack by default: Optimized the error format after hiding the C++ stack, removed the demarcation flag `Error Message Summary`, and aligned with the native Python error format.
- Optimized the error prompts of APIs in non-static graph mode in some static modules, including 9 APIs such as static. append_backward, static.gradients, static.scope_guard, static.Print, static.nn.embedding, static.nn. data_norm, static.nn.multi_box_head, static.nn.nce, and static.nn.py_func.
- Optimized the error message when passing in Tensor as None under dynamic graph model.
- Optimized the printing information of Layers, and supported printing the relationship of each hierarchy in Layers.

## **Inference Deployment**

#### **Model quantification**

- Enhanced the quantification function in case of the training of dynamic graphs: Added the quantification function of dynamic graphs for the `ImperativeQuantAware` class in the unified manner. Currently, it supports quantification of weighted layers such as Conv2D, Linear, etc. Support the obtaining the channel-based quantification parameters of weighted layers, quantification of weightless layers such as ReLU, Tanh, and Layer quantification specified by skip.
- Added the function to obtain the output scale parameter for the model layer during the training of dynamic graph quantification, for the deployment of quantification inference on the Server side.
- dynamic graph quantitative model supports inference deployment using Paddle-Lite.
- For the offline quantification function, support the advance fusion of conv+bn and output LSTM quantitative models. Remove the function of saving sampled data to temporary files.
- For the static graph quantification, support Conv2d_tranpose quantification. Support Linear quantification in the form of per-channel.

### Paddle Inference

The default naming of inference library is changed from fluid_inference to paddle_inference.

#### API

- The inference C++ API is upgraded fully. The new APIs are recommended. The old APIs remain temporarily. There is warning reported in the use of old APIs. The old APIs are planned to be deleted in the future. The new APIs include changes of naming standardization and simplification of usage method, including:
  - A new `paddle_infer` namespace for the C++ interface, containing inference-related interfaces.
  - Renamed `ZeroCopyTensor` to `Tensor` as the default input/output representation of the inference interface.
  - Simplify `CreatePaddlePredictor` to `CreatePredictor`, with keeping the support for only `AnalysisConfig`. Other multiple Configs are not supported.
  - Added service-related utility classes such as `PredictorPool`, which can be used when multiple predictors are created.

#### **Function upgrade**
-  Operator-related version information
  - Some operators are newly added or upgraded in Paddle V2.0. Starting from this version, the forward operator version is defined with compatibility constraints. Through the alignment of operator versions between frameworks, ensure consistent definition and behavior of the same operator, thus enhancing the overall robustness of the framework.
  - Added the registration mechanism for inference forward operator versions and included the incompatible upgrade behavior of operators for the statistics.
  - Added the operator version information for the prediction models. Through the model file, the inference library is able to identify the definition of the operator corresponding to this model, so as to avoid calculation errors caused by different definitions.
- Model interface
  - The `load_inference_model` and `save_inference_model` APIs are migrated to `paddle.static` to improve the usability, with compatibility with the old interfaces.
  - Added six APIs such as `serialize_program`, `deserialize_program`, `serialize_persistables`, `deserialize_persistables`, `save_to_file`, and `load_from_file` for users to perform serialize/deserialize programs, serialize/deserialize params, and saved models/parameters to file, or loaded models/parameters from files.

- Inference-related NV GPU
  - Added the adaptive support for TRT 7.1.
  - Added the adaptive support for Jetson Nx hardware.
  - Paddle-TensorRT enhances the support for the PaddleSlim quantitative model. Cover multiple tasks such as detection, classification, and segmentation on CV.
  - Paddle-TRT supports clip op, and supports the classification model GhostNet running on the Paddle-TRT.
  - Paddle-TRT supports mul op models with channelwise quantification, and supports the PaddleOCR detection. Identified the quantitative models running in the Paddle-TRT int8.
  - Paddle-TRT dynamic shape function supports PaddleSlim quantification Int8 models.
- X86 CPU-related inference
  - Added the support for oneDNN BF16: support the computation of conv2d and gru bf16. It currently supports BF16 prediction for resnet50, googlenet, mobilenetv1 and mobilenetv2 models.
  - Added support for quantification and inverse quantification of scales with bias in oneDNN INT8 quantification strategy.
  - Added version compatibility support for some oneDNN operators.
  - Added the kernel support for `elementwise_add` and `elementwise_mul` INT8 oneDNN on the CPU side.
  - Improved the usability of CPU-side test quantification models. Supported the comparative test of original models and quantitative models at the same time.

- Custom OP
  - Added the support for user-defined Ops on Python-side inference.
-  Memory/GPU memory correlation
  - Added the TryShrinkMemory interface. Reduced the occupation of application's memory/video memory by releasing temporary tensors. For the demo, see [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/test/shrink_memory).
- dynamic graph quantitative model support
  - X86 inference supports dynamic graph quantitative models.
  - NVIDIA GPU inference supports dynamic graph quantitative model.
- Error message:
  - In the Compiling, when enabling ON_INFER, FLAGS_call_stack_level is on by default. The error message indicates that the stack is invoked.

#### **Performance optimization**
- Improved the transformation and optimization of quantitative models.
- NV GPU correlation
  - Optimized the ArgMin and ArgMax OP of CUDA so that the binary system size of the OP is decreased from 60 M to 1.3 M.
  - For the ERNIE model on T4 with using the Paddle-TRT FP16 inference, the performance is improved by 15%.
  - The ERNIE model adds the support for variable-length inputs when TenorRT is enabled. The performance is improved by 147%.In software versions cuda10.1, cudnn 7.6, tensorrt 6.0, [OSS 7.2.1](https://github.com/NVIDIA/TensorRT/tree/7.2.1), model ernie- base-2.0, dataset QNLI, the performance on Nvidia Telsa T4 improves from 905 sentences/s to 2237 sentences/s when input BatchSize = 32.Sample code: [Paddle-Inference-Demo/c++](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++).
- X86 CPU related
  - Added the conv + affine_op pass. The MASK-RCNN fp32 single-threaded performance is improved by 26% on machine 6248.
  - Added the fc + gru pass and enable oneDNN (former MKL-DNN) GRU fp32 kernel, speeding up GRU fp32 model inference on 4 CRU threads by 20% on machine Intel Xeon 6248.
  - By supporting oneDNN INT8 GRU, the GRU INT8 model is about 1.65 times faster compared to NativeConfig inference (threads = 1, batch_size = 50).
  - Added the fuse support for oneDNN batchnorm + activation. The pvanet_ocr model performance is improved by 2.8% as a result.
  - Added the oneDNN FC + Gelu, FC + Sigmoid and FC + tanh operator fusion. The BERT inference model is improved by 4.5%.
  - Added oneDNN inplace support for partial Op
  - Optimized oneDNN LRN op (speedup 1% for the GoogleNet fp32 model).
  - With oneDNN upgraded to 1.6, Ernie Large oneDNN inference on Skylake (Intel Core 6148) is about 2.7x faster (i.e. unit test test_analyzer_ernie_large).
  - Added the interpolate oneDNN forward operator support. Now ocr_det model inference performance improved by 2.04x compared to CPU Native inference alone.

## Paddle Lite
End-side inference engine Paddle Lite v2.8 is adapted to the main framework v2.0

## **Environment Adaptation**

### **Compile and install**

#### **Training Framework Paddle**
- Released the installation package supporting the use of x86 CPUs and the use of Kunlun chips under the FT CPU.
- Added the support for python3.8 in the installation package.
- Added the installation package for cuda10.1 and cuda 10.2.
- (experimental) Released the installation package for cuda11.
- Upgraded the Paddle image of cuda 10.1 and later, and the NCCL version in the CI system image to V2.7.8
- Upgraded oneDNN (former MKL-DNN) from V1.3 to V1.5.
- Added the pre-installed openssl-dev dependencies to the image.
- Removed installed dependencies: nltk, opencv, scipy, rarfile, prettytable, pathlib, matplotlib, graphviz, objgraph.
- Paddle's avx and no_avx are released separately. whl package is reduced by 40%. avx version is installed by default. Optimized installation error message. The system checks the user's CPU type and Paddle version, automatically prompting the corresponding installation error.
- Improved the pypi installation user experience for the Paddle develop version. Reduced the user installation path. You can run pip --pre for installation.

#### **Paddle inference engine**
- The inference library supports cuda10.2-cudnn8-trt7.1 version.
- Release the installation package supporting jetpack and C++ inference library supporting nv_jetson.
- Newly release the joint compilation of two wheel packages for tensorrt, that is, cuda10.0-cudnn7.6-trt6.0.1.5-python36 and cuda10.0-cudnn7.6-trt6.0.1.5-python36.
- Fixed the joint compilation strategy, released the gpu package containing tensorrt separately to avoid the error of no tensorrt when users install the packages of other GPU versions.
- Fixed a bug of duplicate in the inference library packages.


### **Support of new hardware training**
- Kunlun chip: support single card training, static graph multi-card training. Release 10+ models.
- Centerm 910 chip: support single card training.

## **Known Issues**

- Due to cuDNN 8.0.x's own limitations, when using cuDNN 8.0.x to compile inference library and not using TensorRT acceleration, there is performance degradation on many models. This bug is to be fixed in cuDNN's subsequent versions. You can try to use TensorRT acceleration or use cuDNN 7.6.
- Due to cuDNN 8.0.x’s own limitation, memory leak occurs in some models when using cuDNN 8.0.x for inference. Currently, it is found that the problem occurs when the convolutionBiasActivationForward of cuDNN is used. You can try to disable conv_elementwise_add_act_fuse_pass and conv_elementwise_add_act_fuse_pass by using the inference config file config. pass_builder()->DeletePass().If there is still leakage, you can try cuDNN7.6 and send us the model where you found the problem by issue for analysis.
