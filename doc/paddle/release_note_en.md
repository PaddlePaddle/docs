# Release Note

## Important Update
This version is the beta version of PaddlePaddle Framework v2.0. The most important change is the full upgrade of the API system and the comprehensive improvement on the imperative programming (dynamic graph) capability. This version systematically optimizes the directory structure of PaddlePaddle basic APIs, comprehensively fixes relevant issues left over from the past, fully supplements APIs, and especially provides the better high-level API functions. It also provides support for the quantitative training and mixed precision training under a dynamic graph. Perfect syntax support is implemented in the dynamic-to-static conversion. The usability is improved substantially. Dynamic graph-related functions tend to be perfect. In addition, the C++ APIs for the inference library are upgraded and optimized. Both the support of the inference library for quantitative models and the inference performance are fully enhanced.

## Training Framework

### Basic APIs

#### Compatibility Description

For Version Paddle 2.x, users are recommended to use APIs in the paddle root directory. In addition, all the APIs of Version Paddle 1.x are reserved in the paddle.fluid directory. Codes for Version Paddle 1.x training are not changed according to the design, that is, models saved for Version Paddle 1.x training can run on Version Paddle 2.x normally and inference can be performed using Version Paddle 2.x.

#### Directory Structure Adjustment
- Based on the 2.0-alpha version, this version has made some adjustments to the directory structure. The latest adjusted directory structure is as follows:

  | Directory | Functions and Included APIs |
  | :--- | --------------- |
  | paddle.* | The aliases of commonly used APIs are reserved in the paddle root directory, which currently include all the APIs in the paddle.tensor and paddle.framework directories |
  | paddle.tensor | APIs related to tensor operations such as creating zeros, matrix operation matmul, transforming concat, computing add, and finding argmax |
  | paddle.nn | Networking-related APIs such as Linear, Conv2d, loss function, convolution, LSTM，and activation function |
  | paddle.static.nn | Special APIs for networking under a static graph such as input placeholder data, fully connection fc and control flow while_loop/cond |
  | paddle.static | APIs related to the basic framework under a static graph such as Variable, Program, and Executor |
  | paddle.framework | Universal APIs and imprerative mode APIs such as to_tensor |
  | paddle.optimizer | APIs related to optimization algorithms such as SGD, Adagrad, and Adam |
  | paddle.optimizer.lr_scheduler | APIs related to learning rate attenuation |
  | paddle.metric | APIs related to evaluation index computation such as accuracy and auc |
  | paddle.io | APIs related to data input and output such as Dataset, and DataLoader |
  | paddle.device | APIs related to device management such as CPUPlace and CUDAPlace |
  | paddle.distributed | Distributed related basic APIs |
  | paddle.distributed.fleet | Distributed related high-level APIs |
  | paddle.vision | Vision domain APIs such as datasets, data processing, and commonly used basic network structures like resnet |
  | paddle.text | NLP domain APIs such as datasets, data processing, and commonly used basic network structures like transformer |

#### API Alias Rules
- For the convenience of users, APIs will create aliases in different paths, such as `paddle.add -> paddle.sensor.add`. Users are recommend to use the shorter path `paddle.add`.

- All the APIs in the framework and tensor directories are aliased in the paddle root directory. Except for a few special APIs, all other APIs have no aliases in the paddle root directory.

- All the APIs in the paddle.nn directory, except those in the functional directory, have aliases in the paddle.nn directory. All the APIs in the functional directory have no aliases in the paddle.nn directory.

- The following are some special alias relations. It is recommended to use the names on the left.
  - paddle.sigmoid -> paddle.tensor.sigmoid -> paddle.nn.functional.sigmoid
  - paddle.tanh -> paddle.tensor.tanh -> paddle.nn.functional.tanh
  - paddle.remainder -> paddle.mod -> paddle.floor_mod
  - paddle.divide -> paddle.true_divide
  - paddle.rand -> paddle.uniform
  - paddle.randn -> paddle.standard_normal
  - Optimizer.clear_grad -> Optimizer.clear_gradients
  - Optimizer.set_state_dict -> Optimizer.set_dict
  - Optimizer.get_lr -> Optimizer.current_step_lr
  - Layer.clear_grad -> Layer.clear_gradients
  - Layer.set_state_dict -> Layer.set_dict

#### Name Change of Commonly Used APIs

- This version uses tensor representation data, creates tensor APIs, and changes paddle.fluid.dygraph.to_variable to paddle.to_tensor
- Addition, subtraction, multiplication, and division use full names only
- For the current element-by-element operation, no elementwise prefix is added
- For operating by a certain axis, no reduce prefix is added
- For Conv, Pool, Dropout, BatchNorm and Pad networking APIs, 1d, 2d, and 3d suffixes are added according to the input data type

  | Paddle 1.8    | Paddle 2.0-beta |
  | --------------- | ------------------------ |
  | paddle.fluid.layers.elementwise_add | paddle.add               |
  | paddle.fluid.layers.elementwise_mul | paddle.multiply          |
  | paddle.fluid.layers.elementwise_div | paddle.divide |
  | paddle.fluid.layers.elementwise_max | paddle.maximum             |
  | paddle.fluid.layers.elementwise_min | paddle.minimum |
  | paddle.fluid.layers.reduce_sum | paddle.sum |
  | paddle.fluid.layers.reduce_prod | paddle.prod |
  | paddle.fluid.layers.reduce_max | paddle.max        |
  | paddle.fluid.layers.reduce_min | paddle.min        |
  | paddle.fluid.layers.reduce_all | paddle.all        |
  | paddle.fluid.layers.reduce_any | paddle.any        |
  | paddle.fluid.dygraph.Conv2D | paddle.nn.Conv2d |
  | paddle.fluid.dygraph.Conv2DTranspose | paddle.nn.ConvTranspose2d |
  | paddle.fluid.dygraph.Pool2D | paddle.nn.MaxPool2d, paddle.nn.AvgPool2d |

#### Added APIs
- Added a total of 140 APIs. See [Link] (https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-New-API-List) and the API document
  - Added environment setting APIs: paddle.set_default_dtype, paddle.get_default_dtype, paddle.set_device, paddle.get_device, paddle.manual_seed
  - Added tensor operation APIs: numel, chunk, masked_select, isfinite, isinf, isnan, sort, topk, Flatten, dim, tile
  - Added networking APIs: Linear, Bilinear, Embedding, linear, bilinear, embedding
  - Added vision networking APIs: Conv1d, ConvTranspose1d, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, ReflactionPad1d, ReflactionPad2d, ReflactionPad3d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d, ZeroPad2d, ConstantPad1d, ConstantPad2d, ConstantPad3d, PixelShuffle, Upsample, UpsamplingNearest2d, UpsamplingBilinear2d, conv1d, conv_transpose1d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d, max_pool2d, max_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d, adaptive_avg_pool1d, adaptive_avg_pool3d
  - Added text processing networking APIs: SimpleRNN, LSTM, GRU, MultiHeadAttention, Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
  - Added activation APIs: ELU, Hardshrink, Hardtanh, PReLU, ReLU6, Tanh, Tanhshrink, Softmax
  - Added normalization APIs: BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, weight_norm, remove_weight_norm, batch_norm, instance_norm, layer_norm, normalize
  - Added dropout APIs: Dropout2d, Dropout3d, AlphaDropout, dropout, dropout2d, dropout3d
  - Added similarity and loss function APIs: CosineSimilarity, PairwiseDistance, CTCLoss, KLDivLoss, BCEWithLogitsLoss, MarginRankingLoss, SmoothL1Loss, consine_similarity, binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy, ctc_loss, l1_loss, mse_loss, margin_ranking_loss, nll_loss, smooth_l1_loss
  - Added distributed communication APIs: broadcast, all_reduce, reduce, all_gather, scatter, barrier
  - Added probability distribution APIs: Distribution, normal, bernoulli
  - Added optimizer-related APIs: step, AdamW
  - Added dataset-related APIs: Dataset, IterableDataset, TensorDataset, Sampler, RandomSampler, BatchSampler, DistributedBatchSampler

#### Fixing and Improving APIs
- Modified and improved a total of 155 APIs. See [Link] (https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-Upgraded-API-List) and the API document
- Fixed APIs related to random number generation including: seed setting paddle.rand, randn, randint, randperm, dropout, Uniform, and Normal
- Upgraded the codes of the underlying C++ operators corresponding to the following APIs to theoretically achieve compatibility without excluding slight incompatibility: linspace, concat, gather, gather_nd, split, squeeze, unsqueeze, clip, argmax, argmin, mean, norm, unique, cumsum, LeakyReLU, leaky_relu, hardshrink, embedding, margin_ranking_loss, grid_sample, affine_grid
- Added oneDNN support for the relu6 and Sigmoid activation functions

#### Multi-device/Distributed Training APIs

- Single-Machine Multi-Card Training Under a Dynamic Graph
  - Added paddle.distributed.spawn(func, args=(), nprocs=-1, join=True, daemon=False, **options)，which is used to start multi-card training under a dynamic graph.
  - Added paddle.distributed.init_parallel_env(), which is used to initialize the environment of multi-card training under a dynamic graph.
  - Added paddle.distribued.get_rank(), which is used to get the rank of the current process during the multi-card training.
  - Added paddle.distribued.get_world_size(), which is used to get the total number of processes participating in training during the multi-card training.

- Distributed Collective Communication
  - Added paddle.distributed.broadcast(tensor, src, group=0), which broadcasts a tensor of a specified process to all the processes.
  - Added paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0), which performs the reduce operation on specified tensors of all the processes and returns results to all the processes.
  - Added paddle.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=0), which performs the reduce operation on specified tensors of all the processes and returns results to specified processes.
  - Added paddle.distributed.all_gather(tensor_list, tensor, group=0), which gathers specified tensors of all the processes and returns results to all the processes.
  - Added paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0), which distributes tensors in a specified tensor list to all the processes.
  - Added paddle.distributed.barrier(group=0)，which synchronizes all the processes.

### High-level APIs

- Added PaddlePaddle high-level APIs to encapsulate common operations such as networking, training, evaluation, inference, and access so as to implement low code development. In the MNIST handwritten digit recognition task versus the imperative programming implementation mode, high-level APIs can reduce 80% of executable codes.

- **Data Management**
  - Unified data loading and usage method
    - Dataset definition, which is implemented by inheriting `paddle.io.Dataset`.
    - Multi-process data loading using `paddle.io.DataLoader`.
  - Added `paddle.io.IterableDataset`, which is used for a streaming dataset and supports its concurrent acceleration in `paddle.io.DataLoader`.
  - Added `paddle.io.get_worker_info` for dividing child process data in `paddle.io.IterableDataset`.

- **Model Networking**
  - Added the encapsulation of the common loss API `paddle.nn.loss.*` and metric API `paddle.metric.*`
  - Released 12 models based on high-level API implementations, including Transformer, Seq2seq, LAC, BMN, ResNet, YOLOv3, VGG, MobileNet, TSM, CycleGAN, Bert, OCR. The code can be found in [PaddlePaddle/hapi](https://github.com/PaddlePaddle/hapi).

- **Model Execution**
  - Added class API `paddle.Model`, which encapsulates the common model development methods:
    - API `Model.summary`   to view the network structure and the number of parameters of the dynamic graph networking.
    - API `Model.prepare`  to specify a loss function and an optimization algorithm.
    - API `Model.fit`  to implement training and evaluation, which can implement the execution of user-defined functions such as model storage by callback.
    - API `Model.evaluate`  to implement the computation of inference and evaluation indexes on the evaluation set.
    - API `Model.predict`  to implement specific test data inference.
    - API `Model.train_batch`  to implement training on a single batch of data.
    - API `Model.eval_batch`  to implement evaluation on a single batch of data.
    - API `Model.text_batch`  to implement testing on a single batch of data.
    - API `Model.save`/`Model.load` , which supports storing an inference model in dynamic graph training mode.
  - Added callback API `paddle.callbacks.*` as a model execution API, which performs logging and Checkpoint model saving, etc. Users can customize a callback by inheriting `paddle.callbacks.Callback`.

- **Domain APIs**
  - Added computer vision (CV) APIs `paddle.vision`
    - Added dataset API `paddle.vision.datasets.*`, which encapsulates common public datasets and supports random access to data.
    - Added 24 common data preprocessing APIs `paddle.vision.transforms.*` such as Resize, Normalize, etc.
    - Added image classification backbone network and pre-training parameters:
      - `paddle.vision.models.lenet` or `paddle.vision.lenet`
      - `paddle.vision.models.vgg` or `paddle.vision.vgg`
      - `paddle.vision.models.resnet` or `paddle.vision.resnet`
      - `paddle.vision.models.mobilenetv1` or `paddle.vision.mobilenetv1`
      - `paddle.vision.models.mobilenetv2` or `paddle.vision.mobilenetv2`
  - Added natural language processing (NLP)  APIs `paddle.text`.
    - Added dataset API `paddle.text.datasets.*`, which encapsulates commonly-used datasets and supports random access to data.
    - Added networking API `paddle.text.*`.
- **Automatic Breakpoint Restart**
  - Added API `train_epoch_range`, which implements the epoch-level `checkpoint` autosave and autoloading functions on a static graph and supports automatic breakpoint restart.

### Function Optimization (Including Distributed)

#### Dynamic Graph to Static Graph

- **Added Syntax Support for ProgramTranslator**

  - Added dynamic-to-static support for the return syntax so as to return in advance or to return different types of tensors or none in if-elif-else or loop conditions during the dynamic-to-static conversion.

  - Added dynamic-to-static support for the print syntax so that print (tensor) can also print out a tensor in the dynamic-to-static conversion.

    - Added dynamic support for “for traversing a tensor”, “for traversing a tensor using enumeration”, “for traversing a TensorList”, and “for traversing a TensorList using enumeration” syntaxes so that operations related to the circular processing of tensors can be flexibly used in the dynamic-to-static conversion.

    - Added dynamic-to-static support for the assert syntax to ensure that an assert tensor can be true (bool type) or non-0 (other data types) in the dynamic-to-static conversion.

    - Added support for the transfer of cast of data type so that type conversion of similar conversion statements of dynamic graph type such as float (tensor) and int (tensor) can also be performed in a static graph.

- **ProgramTranslator Usability Optimization Function**

  - Changed the dynamic-to-static return type to class StaticLayer from callable. This class can obtain converted static graph information more easily by calling .code，.main_program, and other APIs.

  - Added set_verbosity and set_code_level APIs so that users can set a log class to view a log in the dynamic-to-static running process or a converted code in intermediate state.

  - Added InputSpec to specify the shape and data type of an input tensor variable.

  - Optimized an error message displayed in case of error in the dynamic-to-static running so that codes with running error in the static graph after dynamic-to-static conversion can also be reported to the original error code line in the dynamic graph; deleted some dynamic-to-static errors from python stacks so that an error message is more related to user codes.

  - Support performing a breakpoint test using pdb.set_trace() during the dynamic-to-static conversion.

- **Optimized Deployment of Model Storage and Loading APIs**

  - Added paddle.jit.save API, which is used to save a dynamic-to-static model so that the API is easier to use; deleted an old API ProgramTranslator.save_inference_model.
  - Added paddle.jit.load API, which is used to load inference models including models saved by paddle.jit.save and paddle.io.save_inference_model. After being loaded, models can be used for model inference or model training optimization in a dynamic graph.

#### Mixed Precision Training
- Added the support for mixed precision of dynamic graphs. The ratio of the speed when the ResNet-50 model is trained on V100 using mixed precision to the speed using fp32 is 2.6.

#### Quantitative Training

- Added `ImperativeQuantAware` class. The dynamic graph quantitative training function is provided. Currently, the quantization of Conv2D, Linear, and other layers are supported. The supported model types include MobileNetV1/MobileNetV2/ResNet50.
- After dynamic graph quantitative training is performed on a model, inference deployment of any quantitative model saved using an `ImperativeQuantAware.save_quantized_model` API can be performed using a Paddle-Lite inference library.
- As for static graph quantization, Conv2d_tranpose quantization as well as Linear quantization in the form of per-channel is supported.

#### Performance Optimization (Including Distributed)

- Simplified the DataLoader underlying implementation logic in dynamic graph mode, reduced the thread reading overhead, and further improved the data reading efficiency and the overall model training speed.The overall training speed of MobileNetV1 in a scenario of single V100 card and BatchSize = 128 is increased by 34%.
- Upgrade and performance optimization of dynamic graph networking. A large number of dynamic graph APIs will directly call an automatically generated Pybind API, improving the performance.

#### Basic Functions for Dynamic Graph

- Support the function of updating the gradient using a sparse parameter by configuring embedding and other APIs.
- Added over 120 member functions of Tensor type, including Tensor().abs(), Tensor().add(), and Tensor().cos().
- Added dir() API for a layer to facilitate viewing the attributes and functions in the layer.
- Added an optimizer.set_lr() API so that users can flexibly adjust a learning rate in dynamic diagram mode.
- Added a global parameter initialization method API set_global_initializer to define a global parameter initialization method.
- Added oneDNN (former MKL-DNN) support for dynamic training and inference.Resent50 oneDNN dynamic training with minist dataset is enabled.
- Added oneDNN support for dynamic training and inference. Resent50 oneDNN dynamic training with minist dataset is enabled.

#### Debugging Analysis

- Uniformly changed the wording of LOG (FATAL) throw abnormal at just 100 points to PADDLE_THROW; optimized the error format and content caused by non-support of the framework for a behavior.
- Improved Signal Handler implementation within the framework; optimized the error format and content when system signal error occurs during the execution.
- Optimized the framework error stack format. The python error stack occurring during the compilation is moved to below the native error stack to improve error message reading experience.
- Further improved an accumulative total of about 1,300 error type and prompt copywritings of check errors within the framework to enhance the overall debugging usability of the framework.
- Enhanced dynamic graph error messages. Error messages on the Pybind layer under a dynamic graph are systematically enhanced to improve user experience.

### Bug Fixing

- Fixed the problem that AttributeError may unexpectedly occur when the add_parameter API is used on a layer under a dynamic graph; enhance the input check.
- Fixed the problem that tensors of int_8 and uint_8 types cannot be normally printed so that data can be normally output.

#### Dependency Library Upgrading
- Upgraded oneDNN (former MKL-DNN) to Version 1.5 from Version 1.3.
- Upgrade oneDNN from 1.3->1.5


## Inference

### Paddle Inference

#### API
- Fully upgraded the inference C++ APIs. The new version of the APIs is recommended. The original APIs are reserved tentatively, but give a warning during use, and are planned to be deleted in the future. The upgrade to the new version of the APIs mainly involves naming standardization and usage method simplification. The important changes include:
  - adding a `paddle_infer` naming space for the C++ APIs, containing inference-related APIs.
  - renaming `ZeroCopyTensor` to `Tensor` as the default input/output representation method for the inference APIs.
  - simplifying `CreatePaddlePredictor` to `CreatePredictor` and reserving the support for only `AnalysisConfig`, not for other Configs any more.
  - adding service-related utility classes such as `PredictorPool`, which can be used when multiple predictors are created.

#### Functional Upgrading
- Upgraded the operator version compatibility information registry to support more accurate Op version information and improve inferential compatibility.
- Added the adaptive support for Version TRT 7.1.
- Paddle-TensorRT enhances the support for the PaddleSlim quantitative model. Multiple tasks such as detection, classification, and segmentation on CV are covered.
- Added the support for user-defined operators for Python-side inference.
- Added the kernel support for `elementwise_add` and `elementwise_mul` INT8 oneDNN (former MKL-DNN) on the CPU side.
- Improved the usability of CPU-side test quantitative models. A simultaneous comparison test of original models with quantitative models is supported.
- Added the adaptive support for Jetson Nx hardware.

### Performance optimization
- Added conv + affine_op pass. The MASK-RCNN fp32 single thread performance is improved by 26% (1.26x) on machine 6248.
  - Added conv + affine_op pass, MASK-RCNN single thread performance is improved by 26% (1.26x) on machine 6248
- Added fc + gru pass and enabled oneDNN (former MKL-DNN) GRU fp32 kernel, speeding up GRU fp32 model inference on 4 CPU threads by 20% on machine Intel Xeon 6248.
  - Added fc + gru fuse pass and enabled oneDNN gru fp32 kernel, speeding up GRU fp32 model inference on 4 CPU threads by 20% (1.2x) on machine Intel Xeon 6248
- Added oneDNN inplace support for many operators (speedup 2% for the feature fp32 model).
  - Added support for oneDNN inplace support for many operators (speedup 2% for Feature model)
- Optimized oneDNN LRN operator (speedup 1% for the GoogleNet fp32 model).
  - Optimized LRN operator (speedup 1% for GoogleNet)
- Improved the transformation and optimization of quantitative models.
  -  Improved the transformation and optimization of quantized model
- Optimized the ArgMin, ArgMax operator of CUDA so that the binary system size of the operator is decreased to 1.3 M from 60 M.

#### Bug Fixing

- Fixed the mask-rcnn inference error under CPU inference.
  - Fixed mask-rcnn inference error under CPU inference
- Fixed the error occurring in the CPU multithread inference on quantitative models.
  - Fixed the CPU multithread inference on oneDNN quantized INT8 models
