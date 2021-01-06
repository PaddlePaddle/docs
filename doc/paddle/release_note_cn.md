# Release Note

## 重要更新

飞桨框架2.0-RC1版本有如下重要更新：

- **安装环境** 官方发布支持CUDA11的安装包（experimental）；官方发布支持[百度昆仑芯片](https://cloud.baidu.com/product/kunlun.html)的安装包（experimental）
- **API功能** 支持numpy兼容的`paddle.Tensor` 索引和切片操作(基本索引)；去除部分API中的axis参数，支持numpy兼容的广播语义；新增了部分API，完善了部分API的功能，修复了部分API的bug
- **动静转换** 支持动态图转静态图的更多python语法，并支持通过 `paddle.jit.not_to_static ` 标识不进行动转静的函数
- **框架功能** 支持多次调用`paddle.Tensor.backward()` 进行累计梯度，效果等同于增加batch size后计算的梯度；默认隐藏了C++报错栈，并优化了报错格式；分布式训练支持heterbox异构训练
- **框架性能** 混合精度训练支持纯FP16模式，ResNet50模型V100单卡训练性能达1400+ samples/sec；分布式训练做了性能优化

## 前瞻性预告
- 飞桨框架计划在未来的某个版本起，放弃对python2和python3.5的支持，建议您升级python到3.8版本来使用飞桨
- 飞桨框架计划在未来的某个版本起，放弃对CUDA9.0的支持，建议您升级CUDA版本来使用飞桨

##  训练框架

### 基础API（含分布式）

#### 新增API
- 新增paddle.log2
- 新增paddle.log10
- 新增paddle.nn.initializer.set_global_initializer
- 新增paddle.median
- 新增paddle.broadcast_shape，可以计算两个tensor shape经过broadcast计算后的shape
- 新增paddle.vision.ops.deform_conv2d, paddle.vision.ops.DeformConv2d
- 新增paddle.subtract
- 新增paddle.optimizer.lamb
- 新增Tensor相关API，Tensor.cpu、Tensor.cuda(idx)、Tensor.pin_memory、Tensor.is_leaf、Tensor.clone


#### 修复和完善API
- paddle.multiply 去掉axis
- paddle.pow 去掉 type promotion
- paddle.add, paddle.subtract, paddle.multiply, paddle.divide, paddle.matmul, paddle.reshape, paddle.transpose, paddle.kron, paddle.trace, paddle.sum 支持complex64 和complex128 数据类型
- 移除paddle.maximum, paddle.minimum的axis参数
- multiplex支持动态图
- CrossEntropyLoss增加soft_label and axis，修改形状，并提升性能
- paddle.nn.functional.interpolate size参数支持Tensor格式输入
- paddle.nn.functional.pad添加在constant模式时，对N和C维度的padding
- paddle.optimizer.momentum支持恢复训练
- 修复转换前对BatchNorm指定weight_param名字，再使用paddle.nn.SyncBatchNorm.convert_sync_batchnorm 转换成SyncBatchNorm时报错
- paddle.to_tensor选择设备时，支持直接输入其他Tensor的place
- 优化Tensor.detach的性能，与原Tensor共享内存，减少1次内存拷贝，并且不保留在原计算图中
- 静态图模式下，新增支持通过paddle.optimizer.get_lr()获取学习率
- 修复paddle.Embedding在GPU下使用超范围ID报错异常


#### 移除API（包括别名）
- 移除complex module下的api:  paddle.complex.matmul, paddle.complex.reshape, paddle.complex.transpose, paddle.complex.kron, paddle.complex.trace, paddle.complex.sum, paddle.complex.elementwise_add, paddle.complex.elementwise_sub, paddle.complex.elementwise_mul, paddle.complex.elementwise_div
- 移除paddle.nn.functional下的sigmoid_cross_entropy_with_logits


### 高层API
- 新增api paddle.callbacks.ReduceLROnPlateau
- 新增api paddle.callbacks.LRScheduler
- 新增api paddle.vision.datasets.FashionMnist
- paddle.io.DataLoader中places参数变更为可选参数，当为默认值None时，自动选择paddle.CPUPlace()或paddle.CUDAPlace(0)，places参数将在后续版本删除
- paddle.io.DataLoader支持通过设置batch_size=None来禁用DataLoader自动组batch功能
- 新增api paddle.io.ComposeDataset 用于将多个数据集按字段拼接为一个数据集
- 新增api paddle.io.ChainDataset 用于将多个数据集按sample整合为一个数据集
- 新增api paddle.io.WeightedRadnomSampler 用于通过指定权重进行随机采样
- 新增api paddle.vison.ops.yolo_loss和paddle.vision.ops.yolo_box
- 新增api paddle.flops
- 新增api paddle.callbacks.EarlyStopping
- 更新api model.save，保存文件格式与底层保持一致
- 修复api 修复动态图input dtype为非float32且Model初始化不提供inputs时，保存预测模型报错的bug
- paddle.metric.Accuracy支持输入多维Tensor，支持rank为1的label和one-hot表示的label


### 功能优化（含分布式）
#### 动态图基础功能
- 支持Tensor和Scalar在使用运算符运算时进行正确的类型提升
- 修复了多个模型train/eval模型切换互相干扰的问题。动态图Layer.eval()与no_grad解耦，改动前调用Layer.eval()后Tracer不会自动记录反向，改动后调用Layer.eval()仍会自动记录反向，如果需要反向，可以使用paddle.no_grad
- 支持通过索引或切片修改 Tensor数据
- 增加 inplace 反向检测模块，检测是否前向inplace 操作会影响梯度计算的正确性
- 新增Tensor.backward()自动求导时，梯度会累加在之前的梯度上，可以实现变相扩大“batch_size”
- 支持了 SE-ResNext oneDNN 动态图训练


#### 动态图转静态图

**新增语法**

- 增加在动转静循环中使用isinstance语法的支持
- 添加对赋值shape给tuple的动转静语法支持，如a, b, c, d = tensor.shape
- python的 and/or 语句的左右操作数的执行是有先后顺序的，若左操作数的结果能够确定逻辑值，将不执行右操作数。过去动转静图中的logical_and/logical_or对这种情况处理有问题。增加了这种支持。
- 增加支持了函数 signature中含有**kwargs的情况
- 支持使用 jit.not_to_static 装饰函数，在动转静过程中，不转化该函数
- 支持python字典语法 dict.pop()

**bug修复**

- 修复动转静存储lstm接口时一个表示drop_state的变量没有初始化导致模型存储失败的问题
- 修复嵌套循环在变量分析上的问题
- 修复return在一些特殊情况的问题
- 修复if-else中处理列表生成式及变量分析上的问题
- 修复迭代变量在一些特殊情况的问题
- 修复transpose API 在动态图和静态图行为不一致问题，使之支持动转静
- 修复concat API 在动态图和静态图行为不一致问题，使之支持动转静
- 优化部分动转静报错信息，使报错位置更准确
- 修复convert_call在特殊情况下会重复递归调用问题
- 修复由于2.0 API对out.dtype判断不同导致的动转静问题
- 修复了x.shape == y.shape在动态图是判断list相等，返回True/False，但静态图下会被重载成elementwise的问题，这种转为静态图后对elementwise结果进行reduce。
- 修复了param_guard覆盖不到hook的问题。
- 修复了init运行动态图一些参数变量在静态图因为类型不是静态图变量不能赋值的问题
- 修复了用户在\__init__函数中定义的非参数类型变量值无法正确修改和更新的问题
- 修复了动转静过程中错误转化第三方库logging的问题
- 修复了for-enumerate语法AST转写有误的问题
- 修复了部分warning信息循环显示多次的问题

#### 混合精度训练
- 支持更为激进的FP16训练模式（即纯FP16训练）。为保证模型的收敛性在Momentum优化器中新增`multi_precision`和`rescale_grad`属性，`multi_precision`主要指示优化器需要维护一份master weights
- 使用纯FP16训练，ResNet50模型在配有16GB显存的V100上单卡训练性能可达1400+ samples / sec

#### 模型量化
- 动态图量化支持skip指定Layer
- 动态图量化支持2.0 API Conv 以及Linear

#### 分布式训练优化

- 支持使用`paddle.distibuted.spawn`接口启动`all_gather`等分布式低阶API
- 支持heterbox异构训练
- 流水线并行支持Executor.run接口，提升易用性
- Launch接口升级，支持指定单节点的进程数
- Sharding支持百亿参数模型多卡训练


#### 模型保存与载入

- 支持有多个方法声明由`paddle.jit.to_static`转写的Layer在使用`paddle.jit.save`存储后，仍然能够通过`paddle.jit.load`载入，并且由`paddle.jit.to_static`转写的多个方法仍然能够使用
- 支持由`paddle.jit.load`载入的Layer在fine-tune或者作为其他Layer的子Layer使用之后，仍然能够通过`paddle.jit.save`正确存储
- 拓展`paddle.jit.save`支持存储`paddle.DataParallel`模型
- 优化`paddle.static.load_program_state`接口使用体验，在不指定载入`var_list`的使用场景中，载入目录存在干扰文件时仅警告而不报错
- 支持`paddle.jit.save`处理dict类型的InputSpec
- 支持`paddle.onnx.export`将动态图模型导出为ONNX文件格式


#### 性能优化（含分布式）
- 提升RNN类OP在CPU上的性能（LSTM，GRU，SimpleRNN），对比2.0-rc版本，LSTM、GRU、SimpleRNN前向性能与后向性能均有显著提升
- 优化FastThreadedSSAGraphExecutor调度，修复通信同步场景下，通信计算不重叠的情况，4机32卡resnet50提升约0.3%
- 优化paddle.fleet amp分布式性能，修复最后一个通信和计算不重叠的情况，fp16 4机32卡性能提升约0.5%
- 优化分布式通信组件Communicator性能。GEO-400模式下，W2V模型吞吐率、Simnet-Bow模型性能均有显著提升。Async模式下，相较于飞桨框架1.8按本，W2V模型吞吐率提升11%，CTR-DNN模型性能提升14%
- 优化参数服务器模式下Worker为GPU设备时的性能，降低Embedding查表的拷贝耗时，在CTR-DNN模型中，训练吞吐率有显著提升
- 分布式GPU动态图实现计算和通信overlap，并支持用户细粒度配置梯度fuse的group大小等选项。在ResNet152、Bert两个模型上，多节点性能提升在5%以上。在ResNet50也有3%以上的提升
- 提升cumsum在GPU上的性能。
- 提高了Resnet50 oneDNN 动态图训练的性能。目前Resnet50 oneDNN drgraph训练比CPU训练快 6.4 倍
- 新增GRU和SimpleRNN的cudnn支持


#### 调试分析

- 优化Paddle Python端报错异常类型与Python原生报错类型对齐
- 默认隐藏C++报错栈，优化隐藏C++栈之后的报错格式，去掉分界标志`Error Message Summary`，与Python原生报错格式对齐
- 优化部分static模块下API在非静态图模式下使用报错提示，包括static.append_backward, static.gradients, static.scope_guard, static.Print, static.nn.embedding, static.nn.data_norm, static.nn.multi_box_head, static.nn.nce, static.nn.py_func共9个API
- 优化了动态图模型下传入Tensor为None时的报错信息
- 动态图print tensor的格式进一步优化


### 编译安装

#### 新增支持
- （experimental）发布支持cuda11的安装包
- 将cuda10.1及以上的Paddle镜像以及CI系统镜像中的NCCL版本到2.7.8
- 发布支持xpu的安装包
- 发布支持jetpack的安装包，以及支持nv_jetson的C++预测库。

#### 体验优化
- 修复联编策略，单独发布包含tensorrt的gpu包，避免用户在安装其他GPU版本的包出现没有tensorrt的报错
- 删除安装依赖包：scipy、rarfile、prettytable、pathlib
- 安装文档优化


### Bug修复

- 修复多卡训练时0号GPU卡显存占用多于其他卡的Bug
- 修复了tile op计算时shape推导错误的问题
- 修复了使用paddle时出现的大量invalid escape sequence的warning信息
- 修复了paddle.full设置INF、NAN、NINF等时的bug
- 修复paddle.fleet多nccl comm设置不生效的问题，添加同步模式下多nccl comm通信不重叠的警告
- 修复paddle.framework.seed在TruncatedNormal初始化不符合预期的问题
- 修复AvgPool 相关 API动转静exclusive参数行为不一致问题；修复MaxPool 相关 API ceil_mode传参问题
- 修复paddle.topk在GPU下结果不正确的bug
- 修复 fluid.layers.nn.gather 动态图API，缺少了 overwrite 选项 bug
- 修复Windows下终端不识别CUDA_VISIBLE_DEVICES为空字符的bug，通过设置空字符串可以使框架以CPU模式执行
- 修复当LinearLrWarmup中递归包含Learning Rate Scheduler时，optimizer.state_dict/set_dict时的无法递归保存加载的Bug
- 修复了ptb lm模型单机训练性能下降的问题
- 修复了softmax_with_cross_entropy使用ignore_index时梯度计算的bug
- 修复了AdamW第一次执行后再次获取要进行decay的参数为空的bug


## 推理

###  Paddle Inference


#### 功能升级
- Paddle 在 2.0 中新增或升级了部分算子。从本版本起，对前向算子版本规则进行定义与兼容约束。通过框架间算子版本的对齐，确保不同框架中同一算子版本的定义和行为一致，从而增强框架整体的健壮性
- 新增TryShrinkMemory接口，通过释放临时tensor的方式减少应用显/内存占用，demo示例可参考[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/test/shrink_memory)
- Paddle-TRT支持clip op，支持分类模型GhostNet在Paddle-TRT下运行
- Paddle-TRT int8预测支持含有channelwise量化的mul op的模型，支持PaddleOCR检测和识别的PaddleSlim量化模型在Paddle-TRT int8下运行
- `load_inference_model` 和 `save_inference_model` 两个API迁移到 `paddle.static` 下，提升了易用性，兼容旧接口。
- 新增 `serialize_program`, `deserialize_program`, `serialize_persistables`, `deserialize_persistables`, `save_to_file`,   `load_from_file` 六个API，用来满足用户执行序列化/反序列化 program，序列化/反序列化 params，以及将模型/参数保存到文件，或从文件中加载模型/参数的需求。
- 支持部分模型的BF16预测。目前支持resnet50，googlenet，mobilenetv1和mobilenetv2模型的BF16预测
- 添加了一些oneDNN 算子的版本兼容性支持

#### 性能优化
- ERNIE模型在开启TenorRT时增加变长输入的支持，带来性能提升147%。在软件版本cuda10.1、cudnn 7.6、tensorrt 6.0、[OSS 7.2.1](https://github.com/NVIDIA/TensorRT/tree/7.2.1)，模型ernie-base-2.0，数据集QNLI，输入BatchSize = 32时，Nvidia Telsa T4上的性能从905 sentences/s提升到2237 sentences/s。示例代码：[Paddle-Inference-Demo/c++](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++)
- 提高了oneDNN INT8 GRU性能。GRU INT8 模型的预测速度是原Paddle NativeConfig float32 模型的 1.65倍（线程= 1，batch_size = 50）
- 添加了oneDNN batchnorem + activation的fuse支持，pvanet_ocr模型性能因此提高了2.8％


#### Bug修复
- 修复含有avg pooling或global pooling的模型在jetson设备上出现计算结果错误、报错跳出或hang住的问题
- 修复使用TensorRT动态shape推理时，TensorRT子图输出Tensor的shape结尾是x1时会被错误的删除的问题
- 修复当使用TensorRT推理时，config.pass_builder()->DeletePass()不生效的问题
- 解决了某些模型的性能取决于 matmul 算子的 weights 数值的问题
- 修复了当CPU oneDNN加载多个模型预测时性能变慢的问题

## 模型升级

### PaddleDetection
- 升级动态图模型：
  - Faster RCNN, Faster FPN, Mask RCNN, Mask FPN, Cascade RCNN, Cascade Mask, YOLOv3模型精度打平静态图
    - 支持动转静功能，并打通Paddle Inference，精度速度打平静态图
- 发布实时实例分割模型SOLOv2，相较竞品精度提升2.4个点，预测速度提升31.2%， 训练速度为竞品2.4倍
- 新增Android移动端检测demo，包括SSD、YOLO系列模型
- 新增PACT新量化策略，YOLOv3-Mobilenetv3在COCO数据集上比普通量化相比提升0.7%。

### PaddleSlim

- 动态图压缩功能支持
  - 新增动态图剪裁、量化训练功能
  - 剪裁新增通道数对齐功能，使产出模型更容易被预测库加速
  - PACT量化训练方法改为内置方法，方便用户直接调用
- 新增OFA模型压缩技术，TinyERNIE经压缩后加速40%，精度无损

### PaddleSeg

- 全新发布2.0-rc版本，全面升级至动态图，支持15+分割模型，4个骨干网络，3个数据集，4种Loss：
  - 分割模型：ANN, BiSeNetV2, DANet, DeeplabV3, DeeplabV3+, FCN, FastSCNN, Gated-scnn, GCNet, HarDNet, OCRNet, PSPNet, UNet, UNet++, U^2Net, Attention UNet
  - 骨干网络：ResNet, HRNet, MobileNetV3, Xception
  - 数据集：Cityscapes, ADE20K, Pascal VOC
  - Loss：CrossEntropy Loss、BootstrappedCrossEntropy Loss、Dice Loss、BCE Loss
- 提供基于Cityscapes和Pascal Voc数据集的高质量预训练模型 40+
- 支持多卡GPU并行评估，提供了高效的指标计算功能。支持多尺度评估/翻转评估/滑动窗口评估等多种评估方式。

### PaddleClas

- 全新发布2.0-rc1，全面升级至动态图，支持23个系列分类网络结构，135个图像分类预训练模型。其中包含14个实用的SSLD蒸馏模型，效果普遍比基准模型提升3%以上，新增ResNeSt、RegNet和GhostNet三个系列模型。
- 基于动态图，提供混合精度训练方式和基于DALI的训练方式。
- 基于动态图，提供离线预测部署、服务化部署以及端侧部署三种部署方式。

### PaddleOCR

- 全新发布2.0-rc1，PP-OCR系列模型升级至动态图。提供8.1M超轻量中英文OCR模型，通用中英文OCR模型以及效果更优的多种语言识别模型（纯英文数字、法、德、日、韩），并支持离线预测部署和服务化部署两种部署方式。
- 发布Style-Text通用文本数据合成工具。
- 发布PPOCRLabel文本数据标注工具。

### PaddleRec

- 发布模型：gru4rec, deepfm, mmoe, dnn, LR 支持动态图

### PaddleGAN

- 发布模型：Pixel2Pixel, CyclGAN, PSGAN, UGATIT, ESRGAN, CGAN, DCGAN
- 提供风格迁移，妆容迁移，上色，超分，人物、场景动漫化等预训练模型10个

### PaddleNLP

- 发布2.0-beta版本，全面支持动态图模式，提供PaddleNLP核心库，与高阶API深入融合，支持pip安装，为开发者提供飞桨2.0文本领域的最佳实践。
- 新增文本图学习模型ERNIESage，生成式预训练模型ERNIE-Gen，开放域对话生成模型PLATO-2，语义匹配模型SentenceTransformer，时间序列预估模型TCN等。
- 预训练语言模型进一步丰富，包括ERNIE,  BERT,  RoBERTa, ELECTRA等共计22个预训练模型，其中包含11个中文预训练模型。
- 新增Perplexity, BLEU, Rouge-L等8种常用的文本任务评估指标，适配飞桨2.0 Metrics API体系，提升易用性。
- 新增文本分类、序列标注、机器翻译、阅读理解等共25个数据集，适配飞桨2.0 Dataset API体系，一键快速加载。
- 新增Embedding API功能，包含38个中文词向量，支持快速加载和词粒度语义距离计算。

### Parakeet

- 发布 2.0-alpha 版本，提供 Parakeet 核心库，完善了中文文档，支持 pip 安装。
- 语音合成模型框架全新升级，统一文本前端的接口使用，模型全面升级为 Paddle 2.0 API，包括TransformerTTS、Waveflow、Wavenet 模型，新增 Tacotron2 模型。
- 提供了更多可复用的组网模块，方便灵活搭建模型。优化数据处理及加载流程，提升训练速度。
- 新增 experiment 模块，标准化实验流程，方便实验管理和二次开发，对已有模型提供的实验样例代码。

## 工具组件

### PaddleHub
- 发布 2.0-rc版本，全面迁移动态图编程模式，模型开发调试更加方便，finetune接口更加灵活易用。
- 视觉类任务迁移学习能力全面升级，支持图像分类、图像着色、风格迁移等多种任务。
- BERT、ERNIE、RoBERTa等Transformer类模型升级至动态图，支持文本分类的Fine-Tune能力。
- 优化服务化部署Serving能力，支持多卡预测、自动负载均衡，性能大幅度提升。
- 新增自动数据增强能力Auto Augment，能高效地搜索适合数据集的数据增强策略组合。

### X2Paddle
- 发布 1.0.0-rc0版本，全面支持PaddlePaddle动态图API。
- 新增PyTorch模型转换，支持Tracing和Scripting两种方式进行转换。
- 新增Caffe/ONNX/Tensorflow到Paddle2.0 动态图的转换支持。
- 新增Optimizer模块，主要包括op融合、op消除功能，提升转换后模型代码的可读性以及模型的预测性能。

## [昆仑硬件](https://cloud.baidu.com/product/kunlun.html)

###  模型适配昆仑硬件
- Resnet50, mobilenetv3, deeplabv3, bertbase, DQN 静态图模型适配昆仑硬件
