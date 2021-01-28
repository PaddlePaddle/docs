# Release Note

## 重要更新

飞桨框架2.0.0版本有如下重要更新：

- 编程范式：默认开启动态图模式进行模型开发和训练，通过动转静的方式进行模型部署和训练加速。如果需要使用静态图编程范式，可以通过paddle.enable_static()来切换到静态图模式。
- API体系：对API进行了补充，对目录结构进行了调整，使得更加易用，详情请见：[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/index_cn.html)，同时，提供高层API简化使用流程；详情请见： [飞桨高层API使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/tutorial/quick_start/high_level_api/high_level_api.html)。
- 框架功能：对数据加载、动态图执行，OP性能，混合精度训练，分布式训练，动静转换，等进行了功能增强和性能优化。
- 环境适配： 提供了对ARM架构CPU的支持，增加了对Python 3.8、CUDA 10.1/10.2的支持，发布支持CUDA11的安装包（experimental），发布支持[百度昆仑芯片](https://cloud.baidu.com/product/kunlun.html)的安装包（experimental），详情请见：[开始使用](https://www.paddlepaddle.org.cn/install/quick)。
- 模型库及开发套件：飞桨的官方模型库和套件已经完成绝大部分模型升级至飞桨框架2.0.0版本。
  - [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)：支持2.0动态图，全面迁移动态图编程模式，模型开发调试更加方便，finetune接口更加灵活易用。
  - [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection): 支持2.0动态图，覆盖检测方向主流算法（PP-YOLO、Faster-RCNN、SOLOv2），支持动静转换，打通预测部署，提供了更加模块化的组网方式。
  - [PaddleClas](https://github.com/PaddlePaddle/PaddleClas): 支持2.0动态图，提供了29个系列的分类算法和134个预训练模型，提供了基于SSLD知识蒸馏的优化方案，将分类模型的精度普遍提升3%以上。
  - [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg): 支持2.0动态图，提供了50+的高质量预训练模型，支持15+主流分割网络，提供了业界的SOTA模型OCRNet，很好的提升了产品易用性。
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR):  支持2.0动态图，PPOCR系统、文字检测模型（DB，EAST，SAST）与文字识别模型（Rosetta，CRNN，StarNet）完成2.0动态图适配。
  - [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)：支持2.0动态图，所有模型，包括风格迁移、视频增强、唇形迁移、人脸动漫化等九种模型均基于动态图开发。
  - [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)：支持2.0动态图，免安装，动静组网统一，方便用户的调研和上线，同时整理发布了推荐系统经典数据集。
  - [PaddleNLP](https://github.com/PaddlePaddle/models/PaddleNLP)：支持2.0动态图，提供25+预训练模型和易用的API方式提升文本建模效率。
  - [Parakeet](https://github.com/PaddlePaddle/Parakeet)：支持2.0动态图，已发布的声学模型及声码器均良好支持动态图版本。
  - [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)：支持2.0动态图，包含了视频分类和视频动作定位方向模型，包括: TSN、TSM、SlowFast、AttentionLSTM、BMN模型以及特色应用预训练模型VideoTag和FootballAction。


## 前瞻性预告

- 飞桨框架计划在未来的某个版本起，放弃对python2和python3.5的支持，建议您升级python到3.8版本来使用飞桨。
- 飞桨框架计划在未来的某个版本起，放弃对CUDA9的支持，建议您升级CUDA版本来使用飞桨。

## 训练框架

### 兼容性说明

- 编程范式：飞桨2.0.0默认开启了命令式编程范式（动态图），但仍然保留对静态图的支持，静态图代码（包括1.8版本的静态图代码），可以通过添加`paddle.enable_static()`后来运行。
- API：飞桨框架2.0.0版本推荐用户使用位于paddle根目录下的API，同时在paddle.fluid目录下保留了所有的1.x版本的API，保留对之前版本API体系的支持。因此，1.x版本的静态图训练代码，添加`paddle.enable_static()`即可在2.0.0版本上正常运行；1.x版本训练保存的模型，可以使用2.0.0版本进行推理。
- 我们整理了1.8版本API到2.0版本API的[对应关系表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/09_others_information/api_mapping_cn.html)。
- 我们提供了迁移工具，来方便您将基于旧版本的代码迁移为2.0.0版本的代码，详情请见：[版本迁移工具](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/guides/01_paddle2.0_introduction/migration_cn.html)。

### 动态图模式

默认开启动态图模式进行模型开发和训练，通过动转静的方式进行模型部署和训练加速。详情，请参看：[动态图](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/tutorial/quick_start/dynamic_graph/dynamic_graph.html)，[动态图转静态图](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/guides/04_dygraph_to_static/index_cn.html)。

### API体系

- 基础API
  - API目录结构调整，1.x 版本的API主要位于paddle.fluid目录，本版本对API目录结构进行调整，使得分类更为合理，具体调整后的目录说明请参见[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/index_cn.html)。
  - 新增API共186个，修复和完善API共260个：详情请参考2.0.0 pre release版本的release notes，以及[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/index_cn.html)。
  - 新增分布式基础通信类API到paddle.distributed: broadcast, all_reduce, reduce, all_gather, scatter, barrier；动态图多卡训练启动API spawn, init_parallel_env，动静统一启动方式fleetrun
  -  组网类API实现动静统一，支持在动态图模式和静态图模式两种模式下运行。
- 高层API
  - 新增飞桨高层API，对模型开发过程中常见的组网、训练、评估、预测、存取等操作进行封装，实现低代码开发，请参见[飞桨高层API使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/tutorial/quick_start/high_level_api/high_level_api.html)。
  - 新增分布式高层API paddle.distributed.fleet，支持通过配置DistributedStrategy来支持多种优化策略组合和自动并行、分布式指标计算、InMemoryDataset

### 功能优化（含分布式）
#### 动态图基础功能

- 易用性优化：
  - Tensor功能增强：新增Tensor拷贝接口Tensor.clone()，及120余个Tensor计算操作接口（如Tensor.cos()等）；新增使用索引或切片原地修改Tensor的功能；新增Tensor与Scalar运算时自动类型提升的功能；动态图Tensor打印信息优化，展示形式与Numpy保持相同。
  - Layer功能增强：新增Layer深拷贝接口Layer.deepcopy()；新增Layer属性和函数查看接口Layer.dir()；自本版本起，Layer.eval()调用后，Trace功能仍会自动记录反向操作，如不需要记录反向，需要显式调用paddle.no_grad()。
  - Optimizer新增set_lr()接口，可在动态图模式下灵活调整学习率。
  - 新增set_global_initializer()接口，可定义全局的参数初始化方法。
  - 多卡运行代码简化，不需要再显式调用scale_loss和apply_collective_grads。
- 性能优化：
  - 多卡训练时Embedding等API支持使用稀疏参数梯度更新的功能。
  - 动态图训练和推理新增对Intel加速库oneDNN（原MKL-DNN）的支持，CPU训练场景Resnet50模型可提速6倍。
  - 新增动态图Inplace计算功能，可复用Tensor存储空间，减小显存占用，并新增View方法，可以在共享底层存储的情况下改变Tensor描述。
  - 【不兼容升级】新增动态图梯度累加功能，起到变相“扩大BatchSize”的作用，backward()接口默认不清空梯度，**需要显式调用optimizer.clear_grad()来清空梯度**。
- Bug修复：
  - 修复了多个模型在train/eval切换时会互相干扰的问题。

#### 动态图转静态图

- **动静转换新增语法支持**
  - 新增return语法支持，可以在if-elif-else或者循环条件中提前return，并能够return不同类型的tensor或None。
  - 新增对函数signature中含有**kwargs参数的支持。
  - 新增for、for enumerate遍历Tensor和TensorList的语法支持，遍历Tensor的操作更加灵活。
  - 新增更多python语法支持，如print，assert，cast，isinstance，tuple，dict.pop()等。
- **动静转换易用性优化**
  - 动转静的返回类型从callable函数改为Class，可以调用Class的code，main_program等接口更轻松获取转化后的静态图信息。
  - 动转静装饰器to_static新增支持直接装饰model实例，如to_static(model, input_spec) 。
  - 新增jit.not_to_static装饰器，可以在动转静过程中，不转化该函数。
  - 增加set_verbosity()和set_code_level()接口，可以设置不同级别来查看动转静过程的log或者中间状态的代码。
  - 新增InputSpec，可以指定动转静时输入Tensor变量的形状和数据类型。
  - 报错信息优化，可以定位到原动态图错误的代码行，并隐藏与用户无关的报错信息。
  - 支持用 pdb.set_trace() 进行断点调试。
- **优化部署模型存储载入接口**
  - 新增paddle.jit.save接口用于动转静模型的保存，该接口同时兼容存储未经paddle.jit.to_static转写的Layer对象以及paddle.DataParallel模型，删除旧接口ProgramTranslator.save_inference_model。
  - 新增 paddle.jit.load 接口用于载入静态图格式存储的预测模型，包括paddle.jit.save和paddle.io.save_inference_model保存的模型，模型载入后可在动态图下用于模型推理或者模型训练调优。
  - paddle.jit.TransLatedLayer新增program方法，用于获取paddle.jit.load载入模型的program，便于了解模型结构。
  - 【不兼容升级】paddle.jit.save, paddle.jit.load接口参数model_path含义变更，改为存储文件前缀而非目录。

#### 混合精度训练
- 混合精度策略升级：黑白名单策略（下简称“O1策略”）之外，新增“Almost FP16（下简称O2策略）”支持，即尽可能多使用FP16进行计算。
  - 新增FP16 Guard功能（`paddle.static.amp.fp16_guard`），支持用户自由控制模型中单个Op是否选用FP16计算类型。
  - 用户可自定义`custom_black_list`，以控制某一类Op保持FP32计算。
  - 使用O2策略，Resnet50和Bert base在V100单卡训练速度分别可达1400images/s和590sequences/s。
- 易用性优化：
  - 使用`paddle.static.amp`包统一管理与静态图混合精度训练相关的接口。
  - 为`AutoMixedPrecisionLists`提供简化名称`CustomOpLists`，即用户可使用`CustomOpLists`自定义AMP黑白名单Op列表。

#### 分布式训练优化

- 集合通信All Reduce
  - 支持千亿语言模型混合并行训练：支持基于executor接口的流水线并行训练，sharding-DP策略，GradientMerge+AMP策略，Recompute+Offload策略，megatron策略
  - 支持动态图：支持多流通信策略，自动rebuild group策略，高性能稀疏参数通信，多卡梯度顺序一致性策略
- 参数服务器PS
  - 大规模稀疏功能升级：升级大规模稀疏PS-API，抽象通信组件/参数表/优化器基类，方便用户通过子类派生方式进行二次开发；同时还支持千亿特征流式训练，含特征准入，退场，增量训练，分布式指标预测等；通信方式从GRPC切换成了BRPC
  - 开源异构参数服务器，既支持传统的纯CPU机器PS，也支持基于三级存储(SSD/内存/显存)的纯GPU机器PS，还支持CPU机器+GPU机器/昆仑机器混布PS，可以完成万亿参数点击率预估模型的分钟级训练
- 新训练机制支持：
  - 支持基于控制流的多任务分布式训练，性能较基于Intag的多任务提升50%以上
- 分布式启动方式优化
  - 支持使用`paddle.distibuted.spawn`接口启动`all_gather`等分布式低阶API；
  - `paddle.distributed.launch`接口升级，支持指定单节点的进程数，并可简化为`fleetrun`；
  - 优化`gen_nccl_id`，去除grpc依赖，添加一定的容错性，提升分布式任务启动的稳定性；
  - 支持Gloo方式启动集合通信多CPU

#### 模型保存与载入
- 规范Layer、Optimzier等API的set_dict方法名，统一改为set_state_dict。
- paddle.load兼容性增强：支持从fluid.io.save_inference_model和fluid.io.save_params/persistables等接口的存储结果中载入Layer的state_dict。
- 修改paddle.save/load接口行为，paddle.save不再为存储结果添加后缀，paddle.load每次载入仅返回一个结果，规范接口语义。
- 移除paddle.SaveLoadConfig，对于paddle.jit.save, paddle.jit.load, paddle.load等接口兼容载入的场景，使用**kwargs传入额外的配置，简化接口的使用。
- 原静态图API paddle.io.save, paddle.io.load, paddle.io.save_inference_model, paddle.io.load_inference_model移动到paddle.static模块下。
- 优化paddle.static.load_program_state接口使用体验，在不指定载入var_list的使用场景中，载入目录存在干扰文件时仅警告而不报错。

#### 复数计算

- 扩展动静态图执行引擎，支持复数神经网络训练与复数梯度累加。
- 新增mul, div, matmul, kron, abs等Op对复数计算支持。

#### ONNX功能升级

- 新增API: `paddle.onnx.export`支持paddle2.0动态图转换到ONNX协议
- 新增PPOCR，PPYOLO，FasterRCNN，ERNIE等模型转换
- 更丰富的Paddle op覆盖，支持88个Paddle OP算子，同时支持导出为ONNX 1~12不同版本的算子集

#### 性能优化（含分布式）

- 动态图性能优化：
  - 数据读取性能优化：简化动态图模式下DataLoader底层实现逻辑，降低读取线程开销，进一步提升数据读取效率，提升模型整体训练速度。MobileNetV1在V100单卡、BatchSize=128的场景下整体训练速度提升34%。
  - 动态图组网API升级和性能优化，大量动态图API将直接调用自动生成的Pybind接口，性能显著提升。
  - 提高了Resnet50 oneDNN动态图训练的性能。目前CPU场景Resnet50 oneDNN 动态图训练速度提升6.4 倍。
- OP性能优化：
  - argsort：优化输入Tensor的元素个数等于其`axis`维长度时的性能，前向速度提升34倍，反向速度提升10倍。
  - dropout：优化GPU性能，FP32性能提升约20%，FP16性能提升约50%。
  - cast：优化GPU性能，性能提升10%～20%。
  - softmax：优化axis=-1的情况下的GPU性能，针对不同shape有3倍~96倍的提升。
  - 其他OP性能优化：cumsum，reshape，Flatten，IndexSelect，Roll，elementwise_add，AdamW及RNN类（LSTM，GRU，SimpleRNN）等OP，均有明显性能提升。
- 策略优化：
  - 新增fused_bn_add_act融合策略，可以自动对batch_norm+elementwise_add+activation的组合模式进行自动融合加速。
  - 新增梯度聚合的inplace addto策略，支持原位梯度累加，在ResNet-50混合精度训练中性能提升6.3%。

- 优化FastThreadedSSAGraphExecutor调度，修复通信同步场景下，通信计算不重叠的情况，4机32卡resnet50提升约0.3%。

- 分布式性能优化：
  - 优化lars策略， ResNet50 分布式多卡训练 16k batch size 的 time2train 指标小于 10 分钟。
  - 优化paddle.fleet amp分布式性能，修复最后一个通信和计算不重叠的情况，fp16 4机32卡性能提升约0.5%。
  - 优化paddle.fleet.gradient_merge分布式性能，先聚合梯度再通信，多机性能可提升20%-40%，达到线性加速比。
  - 优化参数服务器通信组件Communicator性能。GEO-400batch通信一次的情况下，W2V模型吞吐率、Simnet-Bow模型性能均有显著提升。Async模式下，相较于飞桨框架1.8按本，W2V模型吞吐率提升11%，CTR-DNN模型性能提升14% 。

#### 调试分析

- 将框架内仅100处使用LOG(FATAL)抛出异常的写法统一改为使用PADDLE_THROW，优化由于框架不支持某种行为而导致的报错格式与内容。
- 完善框架内Signal Handler实现，优化执行遇到系统Signal错误时的报错格式与内容。
- 优化框架报错栈格式，将编译时python报错栈移至原生报错栈下方，提升报错信息阅读体验。
- 累计进一步完善约1500余条框架内检查报错的错误类型与提示文案，提升框架整体调试易用性。
- 动态图报错信息增强，动态图下Pybind层的报错信息进行系统性增强，提升用户体验。
- 优化Paddle Python端报错异常类型，与Python原生报错类型对齐。
- 默认隐藏C++报错栈，优化隐藏C++栈之后的报错格式，去掉分界标志`Error Message Summary`，与Python原生报错格式对齐。
- 优化部分static模块下API在非静态图模式下使用报错提示，包括static.append_backward, static.gradients, static.scope_guard, static.Print, static.nn.embedding, static.nn.data_norm, static.nn.multi_box_head, static.nn.nce, static.nn.py_func共9个API。
- 优化了动态图模型下传入Tensor为None时的报错信息。
- 优化了Layer的打印信息，支持打印Layer中的各个层次结构关系。

## 推理部署

#### 模型量化

- 动态图训练时量化功能增强，新增`ImperativeQuantAware`类统一管理动态图量化功能。目前支持对Conv2D、Linear等带权重层的量化，并支持对权重进行分channel求取量化参数，同时也支持无权重层如ReLU，Tanh的量化，以及skip指定Layer量化的功能。
- 新增动态图量化训练过程中对模型layer求取output scale参数功能，供Server端量化推理部署使用
- 动态图量化模型支持使用Paddle-Lite进行预测部署。
- 离线量化功能支持提前融合conv+bn，及产出LSTM量化模型的功能，移除保存采样数据到临时文件的功能。
- 静态图量化支持Conv2d_tranpose量化，支持Linear使用per-channel形式量化。

### Paddle Inference

预测库默认命名从fluid_inference改为paddle_inference。

#### API

- 全面升级推理C++ API，推荐使用新版API。原API暂时保留，但使用时会报 warning，计划未来会删除；新版API主要是从规范命名、简化使用方法角度做的升级，重要变化包括：
  - C++ 接口新增 `paddle_infer` 命名空间，包含推理相关接口；
  - `ZeroCopyTensor` 更名为 `Tensor`，作为推理接口默认输入输出表示方式；
  - 简化 `CreatePaddlePredictor` 为 `CreatePredictor`，只保留 对`AnalysisConfig` 的支持，不再支持其他多种Config；
  - 新增服务相关的工具类，比如 `PredictorPool`，便于创建多个predictor 时使用。

#### 功能升级
- 算子版本信息相关
  - Paddle 在 2.0 中新增或升级了部分算子。从本版本起，对前向算子版本进行定义与兼容约束。通过框架间算子版本的对齐，确保不同框架中同一算子版本的定义和行为一致，从而增强框架整体的健壮性。
  - 增加推理前向算子版本的注册机制，并将算子的不兼容升级行为纳入统计。
  - 增加预测模型的算子版本信息。预测库通过模型文件，将可以对此模型对应的算子定义进行识别，避免定义不同导致计算错误。
- 模型接口
  - `load_inference_model` 和 `save_inference_model` 两个API迁移到 `paddle.static` 下，兼容旧接口，提升易用性。
  - 新增 `serialize_program`, `deserialize_program`, `serialize_persistables`, `deserialize_persistables`, `save_to_file`,   `load_from_file` 六个API，用来满足用户执行序列化/反序列化 program，序列化/反序列化 params，以及将模型/参数保存到文件，或从文件中加载模型/参数的需求。

- NV GPU 推理相关
  - 新增对TRT 7.1版本的适配支持。
  - 新增对Jetson Nx硬件的适配支持。
  - Paddle-TensorRT增强对 PaddleSlim 量化模型的支持，涵盖CV上检测，分类，分割等多个任务。
  - Paddle-TRT支持clip op，支持分类模型GhostNet在Paddle-TRT下运行。
  - Paddle-TRT 支持含有channelwise量化的mul op的模型，支持PaddleOCR检测和识别量化模型在Paddle-TRT int8下运行。
  - Paddle-TRT 动态shape功能支持PaddleSlim量化Int8模型。
- X86 CPU 推理相关
  - 添加了对oneDNN BF16的支持：支持conv2d 和gru bf16 计算，目前支持resnet50，googlenet，mobilenetv1和mobilenetv2模型的BF16预测。
  - 在oneDNN INT8量化策略中增加对有偏移scales的量化和反量化的支持。
  - 添加了一些oneDNN 算子的版本兼容性支持。
  - CPU端增加了`elementwise_add` 和`elementwise_mul` INT8 oneDNN 内核支持。
  - 提升CPU端测试量化模型的易用性，支持同时对比测试原始模型和量化模型。

- 自定义OP
  - Python端推理新增对用户自定义OP支持。
-  内存 /显存相关
  - 新增TryShrinkMemory接口，通过释放临时tensor的方式减少应用显/内存占用，demo示例可参考[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/test/shrink_memory)。
- 动态图量化模型支持
  - X86 推理支持动态图量化模型。
  - NVIDIA GPU 推理支持动态图量化模型。
- 报错信息
  - 编译打开ON_INFER时，FLAGS_call_stack_level默认为打开，报错信息显示调用栈。

#### 性能优化
- 升级了量化模型的转换和优化
- NV GPU 相关
  - 优化了CUDA 的ArgMin, ArgMax OP，使得该OP的二进制大小从60M下降至1.3M。
  - ERNIE模型在T4上使用Paddle-TRT FP16推理性能提升15%。
  - ERNIE模型在开启TenorRT时增加变长输入的支持，带来性能提升147%。在软件版本cuda10.1、cudnn 7.6、tensorrt 6.0、[OSS 7.2.1](https://github.com/NVIDIA/TensorRT/tree/7.2.1)，模型ernie-base-2.0，数据集QNLI，输入BatchSize = 32时，Nvidia Telsa T4上的性能从905 sentences/s提升到2237 sentences/s。示例代码：[Paddle-Inference-Demo/c++](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++)。
- X86 CPU相关
  - 新增 conv + affine_op pass，在6248机器上，MASK-RCNN fp32单线程性能提高了26％。
  - 新增fc + gru pass和oneDNN（原MKL-DNN） GRU fp32内核，使得GRU fp32模型4线程推断速度在机器Intel Xeon 6248上提高 20％。
  - 通过支持oneDNN INT8 GRU，GRU INT8模型的速度与NativeConfig推理相比，提高了约1.65倍（线程= 1，batch_size = 50）。
  - 添加了oneDNN batchnorm + activation的fuse支持，pvanet_ocr模型性能因此提高了2.8％。
  - 增加了oneDNN FC + Gelu，FC + Sigmoid 和 FC + tanh 算子融合，将BERT推理模型提高了4.5％。
  - 增加了对部分Op的oneDNN inplace支持。
  - 优化的oneDNN LRN op，使得GoogleNet fp32模型提速1％。
  - 通过oneDNN升级到1.6，Ernie Large oneDNN在Skylake上(Intel Core 6148）推理的速度提高了约2.7倍（即单元测试 test_analyzer_ernie_large）。
  - 增加了插值interpolate oneDNN前向算子支持，目前ocr_det模型推理性能相比单纯CPU Native推理提高了2.04倍。

## Paddle Lite
端侧推理引擎Paddle Lite v2.8适配主框架v2.0

## 环境适配

### 编译安装

#### 训练框架Paddle
- 发布支持使用x86 CPU及飞腾CPU下使用昆仑芯片的安装包。
- 新增安装包对python3.8的支持。
- 新增安装包对cuda10.1、cuda10.2的支持。
- （experimental）发布支持cuda11的安装包。
- 将cuda10.1及以上的Paddle镜像以及CI系统镜像中的NCCL版本到2.7.8。
- 升级oneDNN（原MKL-DNN）从1.3至1.5版本。
- 镜像中新增预装openssl-dev依赖。
- 删除安装依赖包：nltk、opencv、scipy、rarfile、prettytable、pathlib、matplotlib、graphviz、objgraph。
- Paddle的avx与no_avx单独发版，whl包减小40%，默认安装avx版本，优化了安装报错信息，会检查用户的CPU类型与Paddle版本，自动给出对应的安装报错提示。
- Paddle develop版本pypi安装用户体验提升，缩减用户安装路径，用pip --pre方式即可进行安装。

#### 推理引擎Paddle Inference
- 预测库支持cuda10.2-cudnn8-trt7.1的版本。
- 发布支持jetpack的安装包，以及支持nv_jetson的C++预测库。
- 新增发布联编tensorrt的两个wheel包，cuda10.0-cudnn7.6-trt6.0.1.5-python36、cuda10.0-cudnn7.6-trt6.0.1.5-python36。
- 修复联编策略，单独发布包含tensorrt的gpu包，避免用户在安装其他GPU版本的包出现没有tensorrt的报错。
- 修复预测库打包有重复的问题。


### 新硬件训练支持
- 昆仑芯片：支持单卡训练，静态图多卡训练，并发布10+模型。
- 昇腾910芯片：支持单卡训练。

## 已知问题

- 由于cuDNN 8.0.x自身的问题，使用cuDNN 8.0.x编译推理库且没有使用TensorRT加速时，在很多模型上有性能退化现象，等待cuDNN后续版本解决。可以尝试使用TensorRT加速，或者使用cuDNN7.6版本。
- 由于cuDNN 8.0.x自身的问题，使用cuDNN 8.0.x版本进行推理时，在某些模型会发生内存泄露现象，当前发现可能发生的为使用cuDNN的convolutionBiasActivationForward时。可以尝试通过推理配置文件config.pass_builder()->DeletePass()禁用conv_elementwise_add_act_fuse_pass、conv_elementwise_add_act_fuse_pass。如果还有泄露现象，可以尝试cuDNN7.6，并将发现问题的模型通过issue方式发给我们分析。
