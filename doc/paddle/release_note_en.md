# Release Note

## Important Update
Compared with the 2.0-beta, this version is further improved in the following aspects:

- Default mode: For the versions later than paddle 2.0-rc, the dynamic graph mode is enabled by default. To use the static graph programming mode, run paddle.enable_static() to switch to it.
- Framework APIs: Modify 50 commonly used API names, add 8 APIs, remove 220 APIs (including alias removal), add the second-order derivative calculation in 8 APIs, add the support of the Kunlun chips in more APIs, formalize the distributed FleetAPI and functionally enhance the high-level APIs.
- Framework features: Optimize the dynamic-to-static conversion usage, optimize model reading and loading, optimize mixed-precision training and quantization strategies, optimize distributed training strategies, and remove 6 compilation dependencies such as nltk; the installation package support for Python 3.8 and CUDA 10.1/10.2.
- Inference engine: Enhance the int8 quantitative capability, add operator version information, optimize the oneDNN performance.
- Fix a numbers of bugs in 2.0-beta.

## Training Framework

### Basic API (Including Distributed)

#### Name Change of Commonly Used APIs
1. Modified 58 API names. For details, see [link](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0rc-renamed-API-List)

#### Added APIs

1. Added paddle.emtpy API to return uninitialized memory
2. Added paddle.emtpy_like API to return uninitialized memory
3. Added paddle.mv API to return the matrix-vector multiplication result
4. Added paddle.multinomial multinomial distribution API
5. Added paddle.nn.LocalResponseNorm and paddle.nn.functional.local_response_norm
6. Added paddle.nn.Pad1D/Pad2D/Pad3D api, and supported constant, reflect, replicate and circular modes
7. Added paddle.add_n
8. Added dynamic graph mixing precision training API, paddle.amp.auto_cast and paddle.amp.GradScaler

#### Fixed and Improved APIs
1. paddle.reshape API supports bool type input
2. paddle.distribution.Categorical API is added with sample and log_prob methods
3. BatchNorm1D, BatchNorm2D, and BatchNorm3D are added with the support of the channel last data layout
4. Modified paddle.optimzier.Adam and paddle.optimizer.AdmaW parameter order
5. yolo_box supports the input feature graph where the H and W are not equal, that is, complete the prediction of a graph with unequal width and length
6. paddle.nn.function.interpolate supports the settings that the input type of scale_factor is list
7. Added the support of oneDNN of the adaptive pool2d operator
8. Added the support of oneDNN of dilated conv and dilated conv_transpose
9. unique supports the GPU device computing
10. paddle.multiply supports the input of non-variable and tensor data types
11. paddle.nn.AdaptiveMaxPool1D/2D/3D and paddle.nn.functional.adaptivemaxpool1d/2d/3d, refactor the implementation of PoolAPI on the python
12. paddle.set_printoptions support setting display options of dynamic graph Tensor
13. paddle.assign API, support array/tensor to tensor assignment
14. paddle.nn.functional.swish/paddle.nn.Swish, delete the beta parameter
15. paddle.nn.functional.thresholded_relu/paddle.nn.ThresholdedReLU, the default value of the threshold parameter is 1.0
16. paddle.norm, after upgrade it supports fro, inf, -inf, 0, 1, 2, and the p norm corresponding to any positive real number p
17. RNN classes (SimpleRNN, LSTM, and GRU) are optimized with the parameter order and the implementation of the base class RNNBase, and integrated with cudnn lstm
18. Fixed the GPU gradient anomaly of adaptive_pool op in special output cases
19. Added the Second-order Derivation Function: batch_norm、abs、log、expand、tile、squeeze、unsqueeze、matmul
20. Added more than 50 OPs to support Kunlun(XPU) training

#### Renamed APIs
1. Renamed 50 APIS of 2.0-beta, see [link](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0rc-renamed-API-List)

#### Removed APIs (Including Aliases)
1. Removed 220 APIs (including aliases), see [link](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0rc-removed-API-List)


#### Multi-device/Distributed Training APIs
1. fleet api is formalized to paddle.distributed.fleet in a unified manner as the Paddle universal distributed training unified entry
2. paddle.distributed.fleet.DistributedStrategy is exposed as Paddle unified parallel strategy definition entry
3. Added paddle.distributed.fleet.meta_optimizer.RecomputeOptimizer API to support the distributed re-computing mechanism
4. Added paddle.distributed.fleet.meta_optimizer.GradientMergeOptimizer API to support the distributed gradient summation mechanism
3. Added paddle.distributed.fleet.meta_optimizer.PipelineOptimizer API to support the distributed pipeline parallel mechanism
4. paddle.distributed.fleet.DistributedStrategy is added with the AMP optimization strategy to support the enabling of automatic blending precision mechanism in the distributed environment
5. paddle.distributed.fleet.DistributedStrategy is added with the dgc optimization strategy to support the enabling of deep gradient compression mechanism in the distributed environment
6. paddle.distributed.fleet.DistributedStrategy is added with the fp16_allreduce optimization strategy to support the enabling of fp16 allreduce communication mechanism in the distributed environment
7. paddle.distributed.fleet.DistributedStrategy is added with the lars optimization strategy to support the use of lars optimizer for large batch size training in the distributed environment
8. paddle.distributed.fleet.DistributedStrategy is added with the lamb optimization strategy to support the use of lamb optimizer for large batch size training in the distributed environment
9. paddle.distributed.fleet supports multi-optimization strategy combinations, including combinations of more than ten kinds of strategies such as amp+recompute, dgc+recompute, amp+recompute+lars, and so on
10. paddle.distributed.fleet.DistributedStrategy is added with the a_sync optimization strategy to support synchronous, asynchronous, GeoSGD, and heterogeneous parameter server optimization training by using the parameter servers in the distributed environment
11. paddle.distributed.fleet.DistributedStrategy is added with the auto experimental optimization strategy to support auto parallel for multi-strategy optimization in the distributed environment
12. Added fleetrun to start the distributed training task, to support Collective mode to start in the single-machine single-card, single-machine multi-card and multi-machine multi-card, support the parameter server mode to start under CPU cluster, GPU cluster, and heterogeneous cluster, and support the direct submission of the PaddleCloud cluster
13. paddle.distributed.fleet supports dynamic graph execution and supports the single-machine single-card, single-machine multi-card and multi-machine multi-card training of a dynamic graph in GPU mode
14. paddle.distributed.fleet is added with the communication collection function, to support all_reduce, all_gather and barrier functions
15. paddle.distributed.fleet is added with the distributed indicator calculation function, including auc, rmse, mae, and acc
16. In paddle.distributed.fleet, fleet.main_program and fleet.startup_program are removed to be replaced with paddle.static.default_main_program() and paddle.static.default_startup_program()
17. paddle.distributed.fleet supports heterogeneous parameter server mode, to implement the heterogeneous computing device training and cross-device collaborative distributed training through fleetAPI and user networking
18. Distributed collective communication API supports CPU devices
19. paddle.distributed.fleet.DistributedStrategy is added with the localsgd optimization strategy
20. paddle.distributed.fleet.DistributedStrategy is added with the adaptivelocalsgd optimization strategy to support the localsgd strategy to automatically calculate step in the distributed environment
21. Added paddle.distributed. InMemoryDataset and QueueDataset are added to support the distributed training by using Dataset

### High-level APIs
1. Added IterableDataset base class support streaming dataset. DataLoader supports multi-process acceleration of IterableDataset, and supports the getting of the child process state through paddle.io.get_worker_info() and the inter-process data division
2. The places parameter of paddle.io.DataLoader is updated to be optional. Places is not specified to use the default value
3. Added 10 map-style datasets such as CIFAR10, CIFAR100, Conll05st, and so on, to support automatic download of dataset and get data in map-style mode
4. Added the num_replicas and rank parameters of the DIstributedBatchSampler interface, for specifying the number of cards and the logical serial number of the current card
5. Added the support of reading tensor dataset of paddle.io.SensorDataset
6. Added paddle.io.Sampler base class, and SequenceSampler. RandomSampler is used for getting data in BatchSampler in order or random order
7. paddle.io.BatchSampler supports Sampler as input, and the original input parameter indices is deleted
8. Removed the original API in paddle.reader
9. The graph conversion operator in paddle.vision.transforms is added to process PIL backend
10. paddle.summary supports multi-input multi-output Layers
11. model.save is upgraded. When a dynamic graph saves a prediction model, a user does not need to call paddle.jit_to_static or add a decorator for the layer function (dynamic to static function).If inputs is passed in when initializing the model, the correct input shape is saved. Otherwise, the input shape of the model is saved according to the pass-in input shape when running the model

### Function Optimization  (Including Distributed)
#### Dynamic Graph
1. Added the clone interface of Tensor. An identical Tensor is copied while the clone Tensor continues to retain in the computation graph and supports the gradient return
2. Hided the scale_loss and apply_collective_grads methods of multi-card API DataParallel of the dynamic graphs. The two methods need not to be called when multi-card model codes are prepared. This can simplify the writing method and improve the usability
3. Supported the modification of Tensor through index or slice (inplace)
4. Optimized the dynamic graph Tensor printing and display, high-dimensional tensor data display mode alignment numpy. The abbreviation is supported
5. Optimized the `__call__` method of the initializer class. The pass-in of block is not required. This can prevent the user from perceiving the static graph block concept in the dynamic graph
6. Add the support for oneDNN dynamic graphs, and support Resnet50 model training and inference.

#### Dynamic Graph to Static Graph

1. For the dynamic graph to static graph, the related API interface is migrated in V2.0, simplifying the import route
2.Dynamic-to-static decorator to_static is added with the support of the direct decoration model instances, for example, to_static(model, input_spec)
3. Added the parsing mechanism for the default value of the name parameter in InputSpec. If no name is specified, the decorated function parameter name is used as name
4. StaticLayer is renamed to StaticFunction
5. Optimized the dynamic to static Debug log
6. Fixed the dynamic to static bug in some scenes

#### Mixed Precision Training
1. Re-constructed the gradient validity check and dynamic loss scaling logic in static graph mixed precision training, and removed some condition block logics

#### Model Quantization
1. Added the division channel quantization function for dynamic graphs, to support to quantize the division channel parameters of the weight of layer in Conv2D and Linear
2. Added the function of getting the output scale parameter on model layer during dynamic graph quantization training for Server-side quantization inference deployment

#### Distributed Training Optimization
1. Supported the pipeline training in parallel
2. Supported the heterogeneous distributed training in parameter server mode, supported PS+GPU, PS+Kunlun, PS+CPU, PS+CPU+GPU (Kunlun) and other devices for training, a single GPU/Kunlun machine + 10 cpu machines to complete click-through rate model training of hundreds of billions of parameters of millions of data in one minute
3. Upgraded the massive sparse function, to support for the sparse ID in int64 range, and support for sparse table self-growth, configuration access conditions and incremental model preservation function
4. Distributed support for control flow multitasking. The performance is improved by over 50% than that in instag multitasking


#### Model Saving and Loading
1. Supported paddle.jit.save interface to store the Layer object without paddle.jit.to_static transcription, to expand the interface usage scenarios
2. Standardized the set_dict method name of the APIs such as Layer and Optimzier, to rename to the set_state_dict in the unified method to standardize the interface name
3. Supported the loading of state_dict of Layer from the result stored in the fluid.io.save_inference_model interface by paddle.load
4. Supported the loading of state_dict of Layer from the default result stored in the fluid.io.save_params/persistables interface by paddle.load, to enable the interface system and improve the usability
5. Modified the paddle.save/load interface behavior. paddle.save does not add the suffix for the stored results. paddle.load returns only one result in each loading to standardize the interface semantics
6. paddle.jit.TransLatedLayer is added with the program method, to get the program of the paddle.jit.load loading model to facilitate the understanding of the model structure
7. Removed paddle.SaveLoadConfig. For paddle.jit.save, paddle.jit.load, paddle.load and other interface-compatible loading scenarios, use **kwargs to pass in additional configuration to simplify the use of the interface
8. Updated the meaning of model_path of the paddle.jit.save and paddle.jit.load interface parameter. The user input string is used as a prefix to the stored file, instead of a directory
9. Original static graph APIs such as paddle.io.save, paddle.io.load, paddle.io.save_inference_model, and paddle.io.load_inference_model are moved to the paddle.static module

#### Performance Optimization  (Including Distributed)
1. Improved the performance of Argsort OP when the number of input Tensor elements is equal to its `axis` dimensional length. The forward speed is improved by 34 times and the reverse speed is improved by 10 times
2. Optimized lars strategy, ResNet50 distributed multi-card training 16k batch size with the time2train index smaller than 10 minutes
3. Added fused_bn_add_act OP, with the integration of batch_norm, elementwise_add and activation OP
4. Added inplace addto strategy for gradient aggregation, to support in-situ gradient summation. The performance is improved by 6.3% in ResNet-50 mixed precision training

#### Debugging Analysis

1. Continued to improve about 1500 pieces of error checking hint texts in paddle, to improve the framework debugging and usability

### Compiling and Installation
1. Added the support for python3.8 in the installation package
2. Removed the installation dependency on matplotlib
3. Remove the installation dependency on graphviz
4. Removed the installation dependency on objgraph
5. Removed the installation dependency on netifaces
6. Remove the installation dependency on nltk
7. Removed the installation dependency on opencv
8. Added the installer support for cuda10.1 and cuda 10.2
9. The prediction library supports cuda10.2-cudnn8-trt7.1 version

### Bug Fixing
1. Fixed the bug of error reported by gradient clipping GradientClipByGlobalNorm used in network where Paddle default dtype is float64
2. Fixed the bug of Windows-based CUDA version 10.1/10.2 failed to load CUDA related dll
3. Fixed the bug of Tensor copy each other between CUDAPinnedPlace and other Place
4. Fixed the bug of error in paddle.jit.load loading Layer without parameter
5. Fixed the bug of calculation error in the large size input of paddle.diag, and fixed the bug of memory usage exception of paddle.diag in Windows Python 3.8 environment
6. Fixed the unreasonable shape problem of paddle.topk in static graph networking
7. Fixed the bug of exit with the direct report of error of paddle.io.DataLoader multi-process mode when started through paddle.distributed.spaw
8. Fixed the problem of device failure in some scenarios when the paddle.set_device interface is set with the runtime
9. Fixed the bug of the gradient calculation error caused by using the variable of forward calculation in paddle.static.nn.while_loop backward calculation
10. Fixed the bug of fleet not supporting paddle.optimizer
11. Fixed the bug that the Adam optimizer formula and thesis have diff
12. Fixed the problem of logsumexp causing too slow compilation on some machines
13. Fixed the ParamAttr missing type check problem
14. Fixed the calculation problem of average pooling core on CPU when AvgPool API ceil_mode=true
15. Fixed the dimension mismatch problem when paddle.distributed.fleet.init_server() is loaded with a model
16. Fixed the problem that the training node does not support GPU in paddle.distributed.fleet parameter server mode
17. Fixed the precision diff problem of paddle.allclose in float64 data type
18. Fixed the error of back propagation supporting grouped conv operators (conv2d grad op with groups)
19. Fixed the bug of failure to save the model when dynamic to static to_static decorative model is directly switched to the eval mode
20. Fixed the bug that matmul does not support fp16bug
21. Fixed the problem of poor performance of matmul reverse calculation and high memory consumption
22. Fixed the error when the bias_attr and weight_attr parameters of paddle.nn.Transformer are specified as bool, list/tuple
23. Fixed the problem that dynamic_decode prediction decoding doesn't end early correctly
24. Fixed the result error of paddle.unsqueeze when axis is Tensor
25. Fixed the problem of paddle.to_tensor caused by zero_copy in some scenarios, to temporarily disable the zero_copy behavior

## Inference

###  Paddle Inference

1. Changed the default name of prediction library from fluid_inference to paddle_inference

#### API


#### Function Upgrading
1. Paddle-TRT dynamic shape supports PaddleSlim quantization of Int8 models
2. Paddle Inference GPU Int8 supports conv2d_transpose quantization
3. Added operator version information for the prediction model
4. Add support for (de/re) quantization with shiftted scales in INT8 quantization strategy
5. Added the support for oneDNN BF16: support conv2d bf16 operator and gru bf16 op, and enabled resnet50 bf16 model inference


#### Performance Optimization
1. The inference performance of ERNIE model using Paddle-TRT FP16 on T4 is improved by 15%.
2. Through the comparison of the speed of supporting oneDNN FP32 GRU and oneDNN INT8 GRU, the speed of the GRU INT8 model is about 1.49 times faster than that of NativeConfig inference (thread = 1, batch_size = 50)
3. By upgrading oneDNN to 1.6, the speed of Ernie Large oneDNN inference on Skylake (Intel Core 6148) is improved about 2.7 times (that is, unit test test_analyzer_ernie_large)

#### Bug Fixing
1. Fixed the bug of memory leak under the variable length input when a user uses the Paddle Inference ZeroCopyRun interface to enable MKLDNN
2. Fixed the bug of prediction error when ERNIE model contains shared parameters
3. Fixed the bug of initialization error for the prediction library with the Paddle-TensorRT function in the environment when TensorRT is not installed
4. Fixed the bug of dimension calculation error when softmax op and layer_norm op use the Paddle-TRT prediction
5. Solved the problem of failing to improve the prediction performance (PaddleOCR repository) when increasing the number of cpu_math_library_num_threads_
6. Solved the problem of oneDNN concat reload data error
7. Solved the problem of error reported when enabling the oneDNN to infer the NHWC model
8. Solved the oneDNN prediction failure problem of the rec_r34_vd_tps_bilstm_attn model
9. Solved the prediction failure problem of deeplabv3p_xception oneDNN
