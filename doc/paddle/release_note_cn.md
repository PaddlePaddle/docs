# 版本说明

## 重要更新

本版本为飞桨框架v2.0的测试版，最重要的变化为API体系的全面升级以及命令式编程（动态图）能力的全面完善。本版本系统优化了飞桨基础API的目录结构，全面修复了历史遗留的相关问题，并对API做了充分补充，特别是提供了更为完善的高层API功能；同时提供了对动态图的量化训练、混合精度训练的支持，动静转换实现了完备的语法支持，并且易用性大幅提升，动态图相关功能趋于完善，推荐使用动态图模式。此外，推理库的C++接口也做了升级优化，推理库对量化模型的支持以及推理性能都有了全面增强。

## 训练框架

### 基础API

#### 兼容性说明

- Paddle 2.x版本推荐用户使用位于paddle根目录下的API，同时在paddle.fluid目录下保留了所有的Paddle 1.x版本的API。按照设计，Paddle 1.x版本训练的代码，不做任何修改，即可在Paddle 2.x版本上正常运行；Paddle 1.x版本训练保存的模型，可以使用Paddle 2.x版本进行推理。

#### 目录结构调整

- 在2.0-alpha版本的基础上，本版本对于目录结构进行了一些调整，调整完最新的目录结构如下：

  | 目录 | 功能和包含的API |
  | :--- | --------------- |
  | paddle.*          | paddle根目录下保留了常用API的别名，当前包括：paddle.tensor, paddle.framework目录下的所有API |
  | paddle.tensor     | 跟tensor操作相关的API，比如：创建zeros, 矩阵运算matmul, 变换concat, 计算add, 查找argmax等 |
  | paddle.nn         | 跟组网相关的API，比如：Linear, Conv2d，损失函数，卷积，LSTM等，激活函数等 |
  | paddle.static.nn  | 静态图下组网专用API，比如：输入占位符data/Input，控制流while_loop/cond |
  | paddle.static | 静态图下基础框架相关API，比如：Variable, Program, Executor等 |
  | paddle.framework  | 框架通用API和imprerative模式的API，比如：to_tensor, prepare_context等       |
  | paddle.optimizer  | 优化算法相关API，比如：SGD，Adagrad, Adam等                  |
  | paddle.optimizer.lr_scheduler  | 学习率衰减相关API                  |
  | paddle.metric     | 评估指标计算相关的API，比如：accuracy, auc等             |
  | paddle.io         | 数据输入输出相关API，比如：save, load, Dataset, DataLoader等 |
  | paddle.device     | 设备管理相关API，比如：CPUPlace， CUDAPlace等                |
  | paddle.distributed      | 分布式相关基础API                                                |
  | paddle.distributed.fleet      | 分布式相关高层API                                         |
  | paddle.vision     | 视觉领域API，比如，数据集，数据处理，常用基础网络结构，比如resnet             |
  | paddle.text       | NLP领域API, 比如，数据集，数据处理，常用网络结构，比如transformer |

#### API别名规则
- 为了方便用户使用，API会在不同的路径下建立别名，比如`paddle.add -> paddle.tensor.add`，推荐用户优先使用较短的路径`paddle.add`

- 所有framework, tensor目录下的API，均在paddle根目录建立别名；除少数特殊API外，其他API在paddle根目录下均没有别名。

- paddle.nn目录下除functional目录以外的所有API，在paddle.nn目录下均有别名；functional目录中的API，在paddle.nn目录下均没有别名。

- 以下为一些特殊的别名关系，推荐使用左边的名称：
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
#### 常用API名称变化

- 此版本使用Tensor表示数据，创建张量API， paddle.fluid.dygraph.to_variable修改为paddle.to_tensor
- 加、减、乘、除使用全称，不使用简称
- 对于当前逐元素操作，不加elementwise前缀
- 对于按照某一轴操作，不加reduce前缀
- Conv, Pool, Dropout, BatchNorm, Pad组网类API根据输入数据类型增加1d, 2d, 3d后缀

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

#### 新增API
- 共计新增140个API，具体参考[链接](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-New-API-List)和API文档
  - 新增环境设置API：paddle.set_default_dtype, paddle.get_default_dtype, paddle.set_device, paddle.get_device, paddle.manual_seed
  - 新增Tensor操作API：numel, chunk, masked_select, isfinite, isinf, isnan, sort, topk, Flatten, dim, tile
  - 新增组网API: Linear, Bilinear, Embedding, linear, bilinear, embedding
  - 新增视觉组网类API：Conv1d, ConvTranspose1d, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, ReflactionPad1d, ReflactionPad2d, ReflactionPad3d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d, ZeroPad2d, ConstantPad1d, ConstantPad2d, ConstantPad3d, PixelShuffle, Upsample, UpsamplingNearest2d, UpsamplingBilinear2d, conv1d, conv_transpose1d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d, max_pool2d, max_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d, adaptive_avg_pool1d, adaptive_avg_pool3d
  - 新增文本处理组网类API: SimpleRNN, LSTM, GRU, MultiHeadAttention, Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
  - 新增激活类API：ELU, Hardshrink, Hardtanh, PReLU, ReLU6, Tanh, Tanhshrink, Softmax
  - 新增归一化API：BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, weight_norm, remove_weight_norm, batch_norm, instance_norm, layer_norm, normalize
  - 新增Dropout类API：Dropout2d, Dropout3d, AlphaDropout, dropout, dropout2d, dropout3d
  - 新增相似度、损失函数类API：CosineSimilarity, PairwiseDistance, CTCLoss, KLDivLoss, BCEWithLogitsLoss, MarginRankingLoss, SmoothL1Loss, consine_similarity, binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy, ctc_loss, l1_loss, mse_loss, margin_ranking_loss, nll_loss, smooth_l1_loss
  - 新增分布式通信类API: broadcast, all_reduce, reduce, all_gather, scatter, barrier
  - 新增概率分布类API： Distribution, normal, bernoulli
  - 新增Optimizer相关API：step, AdamW
  - 新增数据集相关API：Dataset, IterableDataset, TensorDataset, Sampler, RandomSampler, BatchSampler, DistributedBatchSampler

#### 修复和完善API
- 共计修改和完善155个API，具体参考[链接](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0beta-Upgraded-API-List)和API文档
- 修复随机数生成相关的API，包括：种子设置paddle.rand, randn, randint, randperm, dropout, Uniform, Normal等
- 以下API对应的底层C++ OP进行了代码升级，理论上可以实现兼容，但不排除会出现少量不兼容的情况：linspace, concat, gather, gather_nd, split, squeeze, unsqueeze, clip, argmax, argmin, mean, norm, unique, cumsum, LeakyReLU, leaky_relu, hardshrink, embedding, margin_ranking_loss, grid_sample, affine_grid
- 增加了relu6和Sigmoid激活函数的 oneDNN支持

#### 多设备/分布式训练API
- 动态图单机多卡训练
	 - 新增paddle.distributed.spawn(func, args=(), nprocs=-1, join=True, daemon=False, **options)，用于启动动态图多卡训练。
	 - 新增paddle.distributed.init_parallel_env()，用于初始化动态图多卡训练的环境。
	 - 新增paddle.distributed.get_rank()，用于获取多卡训练时当前进程的rank。
	 - 新增paddle.distributed.get_world_size()，用于获取多卡训练时参与训练的总进程数。

 - 分布式集合通信
	 - 新增paddle.distributed.broadcast(tensor, src, group=0)，将指定进程上的tensor广播到所有进程。
	 - 新增paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0)，对所有进程的指定Tensor执行归约操作，结果返回给所有进程。
	 - 新增paddle.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=0)，对所有进程的指定Tensor执行归约操作，结果返回给指定进程。
	 - 新增paddle.distributed.all_gather(tensor_list, tensor, group=0)，聚合所有进程的指定Tensor，结果返回给所有进程。
	 - 新增paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0)，将指定进程Tensor列表中的Tensor分发到所有进程。
	 - 新增paddle.distributed.barrier(group=0)，同步所有进程。
### 高层API

- 新增飞桨高层API，对模型开发过程中常见的组网、训练、评估、预测、存取等操作进行封装，实现低代码开发，MNIST手写数字识别任务对比命令式编程模式实现方式，高层API可减少80%执行类代码。

- **数据管理**
	- 统一数据加载使用方式
		- 数据集定义，继承`paddle.io.Dataset`进行实现。
		- 多进程数据加载，使用`paddle.io.DataLoader`。
	- 新增`paddle.io.IterableDataset`用于流式数据集，并在`paddle.io.DataLoader`中支持对其进行并发加速。
	- 新增`paddle.io.get_worker_info`用于`paddle.io.IterableDataset`中划分子进程数据。
- **模型组网**
	- 新增常见Loss接口`paddle.nn.loss.*`和Metric接口`paddle.metric.*`的封装
	- 发布基于高层API实现的12个模型
		- Transformer，Seq2seq，LAC，BMN，ResNet，YOLOv3，VGG，MobileNet，TSM，CycleGAN，Bert，OCR
		- 发布于[PaddlePaddle/hapi](https://github.com/paddlePaddle/hapi)仓库
- **模型执行**
    - 新增Model类`paddle.Model`封装，封装模型开发过程中常用的基础功能，包括：
		- 提供`Model.summary`接口，用于查看动态图组网的网络结构与参数数量。
		- 提供`Model.prepare`接口，用于指定损失函数和优化算法。
		- 提供`Model.fit`接口，实现训练和评估，可通过callback方式实现训练过程中执行自定义功能，比如模型存储等。
		- 提供`Model.evaluate`接口，实现评估集上的预测和评估指标计算。
		- 提供`Model.predict`接口，实现特定的测试数据推理预测。
		- 提供`Model.train_batch`接口，实现单batch数据的训练。
		- 提供`Model.eval_batch`接口，实现单batch数据的评估。
		- 提供`Model.text_batch`接口，实现单batch数据的测试。
		- 提供`Model.save`/`Model.load`接口，支持动态图训练模式存储推理模型。
	- 新增Callback接口`paddle.callbacks.*`，用于模型执行接口，进行日志记录、Checkpoint模型存储等，用户可继承`paddle.callbacks.Callback`进行自定义。
- **领域API**
    - 新增视觉（CV）领域接口`paddle.vision`
    	- 新增Dataset接口`paddle.vision.datasets.*`，对常用数据集进行封装，支持数据的随机访问
		- 新增Resize, Normalize等24种常见的数据预处理接口`paddle.vision.transforms.*`
		- 新增图像分类骨干网络和预训练参数
			- `paddle.vision.models.lenet` 或 `paddle.vision.lenet`
			- `paddle.vision.models.vgg` 或 `paddle.vision.vgg`
			- `paddle.vision.models.resnet` 或 `paddle.vision.vgg`
			- `paddle.vision.models.mobilenetv1` 或 `paddle.vision.mobilenetv1`
			- `paddle.vision.models.mobilenetv2` 或 `paddle.vision.mobilenetv2`
	- 新增自然语言处理（NLP）领域接口`paddle.text`
		- 新增Dataset接口`paddle.text.datasets.*`，对常用数据集进行封装，支持数据的随机访问
		- 新增领域组网接口`paddle.text.*`
- **自动断点重启**
   -  新增接口 `train_epoch_range`:可以在静态图上实现基于epoch粒度的 `checkpoint` 自动保存和自动加载功能，支持自动断点重启。

### 功能优化（含分布式）

#### 动态图转静态图

- **ProgramTranslator新增语法支持**

	- 新增对return语法动转静支持，使得动转静时可以在if-elif-else或者循环条件中提前return，也能return不同类型的tensor或None。

	- 新增对print语法动转静支持，使得print(tensor)也能在动转静中打印出tensor。

	- 新增对for遍历Tensor，for enumerate遍历Tensor，for遍历TensorList，for enumerate遍历TensorList几种语法的动转静支持，使得循环处理Tensor的相关操作在动转静中能够灵活使用。

	- 新增对assert语法动转静支持，使得assert tensor也能在动转静中保证tensor为True（bool类型）或者非0（其他数据类型）。

	- 新增对数据类型cast的转写支持，使得float(tensor), int(tensor) 等类似的动态图类型转化语句也能在静态图中进行类型转化。

- **ProgramTranslator易用性优化功能**

	- 将动转静的返回类型从callable函数改为class StaticLayer，这个class可以调用.code，.main_program等接口更轻松获取转化后的静态图信息。

	- 增加 set_verbosity 和 set_code_level 接口，可以让用户设置log级别来查看动转静运行过程的log或者查看中间状态转化的代码。

	- 新增InputSpec，可以指定动转静时输入Tensor变量形状和数据类型。

	- 优化了动转静运行下如果出错显示的报错信息，使动转静后静态图运行错误的代码也能汇报到原动态图错误的代码行，并且删除python栈中动转静部分报错，使报错信息更多与用户代码相关。

	- 动转静支持用 pdb.set_trace() 进行断点调试。

- **优化部署模型存储载入接口**

	- 新增 paddle.jit.save 接口用于动转静模型的保存，使接口更加易用，删除旧接口ProgramTranslator.save_inference_model 。
	- 新增 paddle.jit.load 接口用于载入静态图格式存储的预测模型，包括paddle.jit.save和paddle.io.save_inference_model保存的模型，模型载入后可在动态图下用于模型推理或者模型训练调优。

#### 混合精度训练
- 增加了动态图混合精度的支持，ResNet-50模型在V100上使用混合精度相比于fp32训练加速比为2.6。

#### 量化训练

- 新增`ImperativeQuantAware`类，提供动态图量化训练功能，目前支持对Conv2D、Linear等层的量化，支持的模型类型包括MobileNetV1/MobileNetV2/ResNet50等。
- 模型经动态图量化训练后，使用`ImperativeQuantAware.save_quantized_model`接口保存的量化模型可利用Paddle-Lite推理库进行预测部署。
- 静态图量化支持Conv2d_tranpose量化，支持Linear使用per-channel形式量化。
#### 性能优化（含分布式）

- 简化动态图模式下DataLoader底层实现逻辑，降低读取线程开销，进一步提升数据读取效率，提升模型整体训练速度。经测试MobileNetV1在V100单卡、BatchSize=128的场景下整体训练速度提升34%。
- 动态图组网API升级和性能优化，大量动态图API将直接调用自动生成的Pybind接口，提升性能。

#### 动态图基础功能

- 支持多卡训练时配置Embedding等API使用稀疏参数梯度更新的功能。@威行
- 增加Tensor类成员函数，包括Tensor().abs()、Tensor().add()、Tensor().cos()等120余个。
- 增加Layer的dir()接口，可以方便地查看Layer中属性和函数。
- 增加optimizer.set_lr()接口，用户可以在动态图模式下中灵活调整学习率。
- 增加全局参数初始化方式的接口set_global_initializer，可定义全局的参数初始化方法。
- 增加了对动态训练和推理的oneDNN（原MKL-DNN）支持。Resent50 oneDNN动态训练可以使用（Minist数据集）
  - Added oneDNN support for dynamic training and inference. Resent50 oneDNN dynamic training with minist dataset is enabled.

#### 调试分析

- 将框架内仅100处使用LOG(FATAL)抛出异常的写法统一改为使用PADDLE_THROW，优化由于框架不支持某种行为而导致的报错格式与内容。
- 完善框架内Signal Handler实现，优化执行遇到系统Signal错误时的报错格式与内容。
- 优化框架报错栈格式，将编译时python报错栈移至原生报错栈下方，提升报错信息阅读体验。
- 累计进一步完善约1300余条框架内检查报错的错误类型与提示文案，提升框架整体调试易用性。
- 动态图报错信息增强，动态图下Pybind层的报错信息进行系统性增强，提升用户体验。

### Bug修复

- 修复动态图Layer使用add_parameter接口可能意外出现AttributeError的问题，增强输入检查。
- 修复无法正常打印int_8与uint_8类型的Tensor的问题，使数据可以正常输出。

#### 依赖库升级
- 升级oneDNN（原MKL-DNN）从1.3至1.5版本
  - Upgrade oneDNN from 1.3->1.5
## 推理

###  Paddle Inference

#### API
- 全面升级推理C++ API，推荐使用新版API。原API暂时保留，但使用时会报 warning，计划未来会删除；新版API主要是从规范命名、简化使用方法角度做的升级，重要变化包括：
	- C++ 接口新增 `paddle_infer` 命名空间，包含推理相关接口；
	- `ZeroCopyTensor` 更名为 `Tensor`，作为推理接口默认输入输出表示方式；
	- 简化 `CreatePaddlePredictor` 为 `CreatePredictor`，只保留 对`AnalysisConfig` 的支持，不再支持其他多种Config；
	- 新增服务相关的工具类，比如 `PredictorPool`，便于创建多个predictor 时使用。

#### 功能升级
- 升级算子版本兼容信息注册表以支持更精确的Op版本信息，提升推理兼容性。
- 新增对TRT 7.1版本的适配支持。
- Paddle-TensorRT增强对 PaddleSlim 量化模型的支持，涵盖CV上检测，分类，分割等多个任务。
- Python端推理新增对用户自定义OP支持。
- CPU端增加了`elementwise_add` 和`elementwise_mul` INT8 oneDNN（原MKL-DNN）内核支持。
- 提升了CPU端测试量化模型的易用性，支持同时对比测试原始模型和量化模型。
- 新增对Jetson Nx硬件的适配支持。
#### 性能优化
- 新增 conv + affine_op pass，在6248机器上，MASK-RCNN fp32单线程性能提高了26％。
  - Added conv + affine_op pass, MASK-RCNN single thread performance is improved by 26% (1.26x) on machine 6248
- 新增fc + gru pass和oneDNN（原MKL-DNN） GRU fp32内核，使得GRU fp32模型4线程推断速度在机器Intel Xeon 6248上提高 20％。
  - Added fc + gru fuse pass and enabled oneDNN gru fp32 kernel, speeding up GRU fp32 model inference on 4 CPU threads by 20% (1.2x) on machine Intel Xeon 6248
- 增加了对许多Op的oneDNN inplace支持（人脸feature fp32模型提速2％）
  - Added support for oneDNN inplace support for many operators (speedup 2% for Feature model)
- 优化的oneDNN LRN op，使得GoogleNet fp32模型提速1％
 - Optimized LRN operator (speedup 1% for GoogleNet)
- 升级了量化模型的转换和优化 @intel
  -  Improved the transformation and optimization of quantized model
- 优化了CUDA 的ArgMin, ArgMax OP，使得该OP的二进制大小从60M下降至1.3M

#### Bug修复

- 修复CPU下的mask-rcnn推断错误的问题
  - Fix mask-rcnn inference error under CPU inference
- 修复CPU多线程量化模型和推断过程中出现的错误
  - Fix the CPU multithread inference on oneDNN quantized INT8 models
