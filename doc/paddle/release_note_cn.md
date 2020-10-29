# Release Note

## 重要更新
相对2.0-beta版，本版本在如下方面进一步完善：

 - 默认模式：paddle2.0-rc后将默认开启动态图模式；如果需要使用静态图编程模式，可以通过paddle.enable_static()来切换到静态图模式。
 - 框架API：修改50个常用API名称，新增8个基础API实现，移除220个API（包含别名移除），8个API增加二阶导数计算，更多API增加了对昆仑芯片的支持，分布式FleetAPI正式化，高层API进行了功能增强。
 - 框架功能：优化动静转换用法，优化模型读取和载入，优化混合精度训练和量化策略，优化分布式训练策略。删除了nltk等6项编译依赖；安装包增加对Python 3.8、CUDA 10.1/10.2的支持。
 - 推理引擎：增强int8量化能力，增加算子版本信息，oneDNN相关的功能强化和性能优化。
 - 修复了beta版本中的存在的bug。

##  训练框架

### 基础API（含分布式）

#### 新增API
1. 新增 paddle.emtpy API，返回未初始化的内存
2. 新增 paddle.emtpy_like API，返回未初始化的内存
3. 新增 paddle.mv API，返回矩阵-向量乘的结果
4. 新增paddle.multinomial多项分布API
5. 新增paddle.nn.LocalResponseNorm和paddle.nn.functional.local_response_norm
6. 新增paddle.nn.Pad1D/Pad2D/Pad3D api，支持constant，reflect，replicate和circular模式
7. 新增paddle.add_n
8. 新增动态图混合精度训练API，paddle.amp.auto_cast和paddle.amp.GradScaler

#### 修复和完善API
1. paddle.reshape API支持bool类型输入
2. paddle.distribution.Categorical API添加sample和log_prob方法
3. BatchNorm1D, BatchNorm2D, BatchNorm3D 添加了 channel last 数据布局支持
4. paddle.optimzier.Adam和paddle.optimizer.AdamaW参数顺序修改
5. yolo_box支持输入特征图H，W不相等，用于完成长宽不相等的图像预测
6. paddle.nn.function.interpolate 支持 scale_factor 输入类型为 list
7. 添加了adaptive pool2d运算符的oneDNN支持
8. 添加了dilated conv和dilated conv_transpose的oneDNN支持
9. unique支持GPU设备计算
10. paddle.multiply 支持非variable 和 tensor 数据类型 输入
11. paddle.nn.AdaptiveMaxPool1D/2D/3D 和paddle.nn.functional.adaptivemaxpool1d/2d/3d，重构python端PoolAPI的实现
12. paddle.set_printoptions支持设置动态图Tensor的显示选项
13. paddle.assign API，支持数组/张量到张量的赋值
14. paddle.nn.functional.swish/paddle.nn.Swish，删除beta参数
15. paddle.nn.functional.thresholded_relu/paddle.nn.ThresholdedReLU，threshold参数默认值为1.0
16. paddle.norm，升级后支持fro、inf、-inf、0、1、2，和任何正实数p对应的p范数
17. RNN类（SimpleRNN、LSTM、GRU）优化参数顺序和基类RNNBase实现，集成cudnn lstm
18. 修复adaptive_pool op在特殊输出情况下GPU梯度异常的问题
19. 新增支持二阶求导功能：batch_norm、abs、log、expand、tile、squeeze、unsqueeze、matmul
20. 新增50余个算子对昆仑（XPU）训练的支持

#### API名称变化
1. 对2.0-beta的50个API名称进行了修改，详见 [链接](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0rc-renamed-API-List)

#### 移除API（包括别名）
1. 移除220个API（包括别名），详见 [链接](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-2.0rc-removed-API-List)

#### 多设备/分布式训练API
1. Fleet API正式化，统一到paddle.distributed.fleet作为Paddle通用分布式训练统一入口
2. paddle.distributed.fleet.DistributedStrategy作为Paddle统一并行策略定义入口暴露
3. 增加paddle.distributed.fleet.meta_optimizer.RecomputeOptimizer API，支持分布式下的重计算机制
4. 增加paddle.distributed.fleet.meta_optimizer.GradientMergeOptimizer API，支持分布式下的梯度累加机制
3. 增加paddle.distributed.fleet.meta_optimizer.PipelineOptimizer API，支持分布式下的流水线并行机制
4. paddle.distributed.fleet.DistributedStrategy新增amp优化策略，支持分布式下自动混合精度机制的开启
5. paddle.distributed.fleet.DistributedStrategy新增dgc优化策略，支持分布式下深度梯度压缩机制的开启
6. paddle.distributed.fleet.DistributedStrategy新增fp16_allreduce优化策略，支持分布式下fp16 allreduce通信机制的开启
7. paddle.distributed.fleet.DistributedStrategy新增lars优化策略，支持分布式下大batch size 训练使用 lars 优化器
8. paddle.distributed.fleet.DistributedStrategy新增lamb优化策略，支持分布式下大batch size 训练使用 lamb 优化器
9. paddle.distributed.fleet支持多优化策略组合，支持包括amp+recompute, dgc+recompute, amp+recompute+lars等十余种策略的组合
10. paddle.distributed.fleet.DistributedStrategy新增a_sync优化策略，支持分布式下使用参数服务器进行同步、异步、GeoSGD以及异构参数服务器优化训练
11. paddle.distributed.fleet.DistributedStrategy新增auto实验性优化策略，支持分布式下多策略最优化自动并行
12. 增加fleetrun启动分布式训练任务，支持Collective模式在单机单卡，单机多卡和多机多卡下启动，支持参数服务器模式在CPU集群、GPU集群、异构集群下启动，支持直接提交PaddleCloud集群
13. paddle.distributed.fleet支持动态图执行，支持GPU模式下动态图单机单机、单机多卡和多机多卡训练
14. paddle.distributed.fleet 新增通信集合功能，支持all_reduce，all_gather及 barrier功能
15. paddle.distributed.fleet 新增分布式指标计算功能，包括auc，rmse， mae，acc 等
16. paddle.distributed.fleet下废弃原fleet.main_program和fleet.startup_program，替换为paddle.static.default_main_program() 和 paddle.static.default_startup_program()
17. paddle.distributed.fleet支持异构参数服务器模式，可通过fleetAPI配合用户组网实现异构计算设备训练，跨设备协作进行分布式训练
18. 分布式集合通信API支持CPU设备
19. paddle.distributed.fleet.DistributedStrategy新增localsgd优化策略
20. paddle.distributed.fleet.DistributedStrategy新增adaptivelocalsgd优化策略，支持分布式下自动计算step步长的localsgd策略
21. 新增paddle.distributed添加InMemoryDataset和QueueDataset支持使用Dataset进行分布式训练

### 高层API
1. 新增IterableDataset基类支持流式数据集，DataLoader支持对IterableDataset进行多进程加速，并支持通过paddle.io.get_worker_info()获取子进程状态并进行进程间数据划分
2. paddle.io.DataLoader的places参数更新为可选，不指定places使用默认的places
3. 新增CIFAR10, CIFAR100, Conll05st等10个map-style数据集，支持数据集自动下载并以map-style方式获取数据
4. DIstributedBatchSampler接口新增num_replicas和rank参数用于指定卡数和当前卡逻辑序号
5. 新增paddle.io.TensorDataset支持tensor数据集读取
6. 新增paddle.io.Sampler基类，并新增SequenceSampler，RandomSampler用于在BatchSampler中顺序或乱序获取数据
7. paddle.io.BatchSampler支持Sampler作为输入，删除原输入参数indices
8. 下线paddle.reader下原有API
9. paddle.vision.transforms中的图像变换算子添加处理PIL的后端
10. paddle.summary支持多个输入与多个输出的Layer
11. model.save升级，在动态图保存预测模型时，用户不需要调用paddle.jit_to_static或者为layer函数增加装饰器（动转静的功能）。并且如果用户在Model初始化时如果传入了inputs，则可以保存正确的输入shape，否则模型的输入shape会按照运行模型时传入的输入shape保存

### 功能优化（含分布式）
#### 动态图基础功能

1. 新增Tensor的clone接口，会拷贝一个完全相同的Tensor，同时clone后的Tensor继续保留在计算图中，并支持梯度回传
2. 支持通过索引或切片原地(inplace) 修改 Tensor
3. 动态图Tensor打印和显示优化，高维tensor数据显示方式对齐numpy，支持缩略形式
4. 优化了initializer类的`__call__`方法，不再需要传入block，避免用户在动态图中感知到静态图block概念
5. 隐藏动态图多卡API DataParallel的scale_loss和apply_collective_grads方法，编写多卡模型代码时不再需要调用这两个方法，简化写法，提升易用性
6. 添加oneDNN 动态图支持，支持了 Resnet50模型训练和推理。@intel

#### 动态图转静态图

1. 动态图转静态图相关API接口迁移2.0，简化了import 路经
2. 动转静装饰器 to_static 新增支持直接装饰 model 实例，如 to_static(model, input_spec)
3. 新增InputSpec中name参数的默认值解析机制，若未指定name，则使用被装饰函数参数名作为name
4. StaticLayer重命名为StaticFunction
5. 优化了动转静Debug log
6. 修复了一些场景下动转静的bug

#### 混合精度训练
1. 重构静态图混合精度训练中的梯度有效性检查和动态loss scaling逻辑，去除一些condition block逻辑

#### 模型量化
1. 新增动态图分channel量化功能，支持对Conv2D和Linear等layer的权重进行分channel求取量化参数
2. 新增动态图量化训练过程中对模型layer求取output scale参数功能，供Server端量化推理部署使用

#### 分布式训练优化

1. 支持流水线并行训练
2. 支持参数服务器模式下异构分布式训练，支持PS+GPU，PS+昆仑， PS+CPU，PS+CPU+GPU(昆仑)等多种设备进行训练，单台GPU/昆仑机器+10台cpu机器上，完成千万数据千亿参数点击率模型分钟级训练
3. 大规模稀疏功能进行了升级，支持int64范围内的稀疏ID，支持稀疏表自增长、配置准入条件及增量模型保存功能
4. 分布式支持控制流多任务，性能较instag多任务提升50%以上

#### 模型保存与载入
1. 支持paddle.jit.save接口存储未经paddle.jit.to_static转写的Layer对象，扩大接口使用场景
2. 规范Layer、Optimzier等API的set_dict方法名，统一改为set_state_dict，规范接口名
3. 支持paddle.load从fluid.io.save_inference_model接口存储的结果中载入Layer的state_dict，打通接口体系，提升易用性
4. 支持paddle.load从fluid.io.save_params/persistables接口默认存储结果中载入Layer的state_dict，打通接口体系，提升易用性
5. 修改paddle.save/load接口行为，paddle.save不再为存储结果添加后缀，paddle.load每次载入仅返回一个结果，规范接口语义
6. 为paddle.jit.TransLatedLayer新增program方法，用于获取paddle.jit.load载入模型的program，便于了解模型结构
7. 移除paddle.SaveLoadConfig，对于paddle.jit.save, paddle.jit.load, paddle.load等接口兼容载入的场景，使用**kwargs传入额外的配置，简化接口的使用
8. 更新paddle.jit.save, paddle.jit.load接口参数model_path的含义，用户输入的字符串作为存储文件前缀而非目录
9. 原静态图API paddle.io.save, paddle.io.load, paddle.io.save_inference_model, paddle.io.load_inference_model移动到paddle.static模块下

#### 性能优化（含分布式)

1. 提升Argsort OP当输入Tensor的元素个数等于其`axis`维长度时的性能，前向速度提升34倍，反向速度提升10倍
2. 优化lars策略， ResNet50 分布式多卡训练 16k batch size 的 time2train 指标小于 10 分钟
3. 新增fused_bn_add_act OP，融合batch_norm、elementwise_add和activation OP
4. 新增梯度聚合的inplace addto策略，支持原位梯度累加，在ResNet-50混合精度训练中性能提升6.3%

#### 调试分析

1. 继续完善paddle中约1500条报错检查的提示文案，提升框架调试易用性

### 编译安装
1. 新增安装包对python3.8的支持
2. 删除对matplotlib的安装依赖
3. 删除对graphviz安装依赖
4. 删除对objgraph安装依赖
5. 删除对netifaces的安装依赖
6. 删除对nltk的安装依赖
7. 删除对opencv的安装依赖
8. 新增安装包对cuda10.1、cuda10.2的支持
9. 预测库支持cuda10.2-cudnn8-trt7.1的版本

### Bug修复

1. 修复梯度裁剪GradientClipByGlobalNorm在Paddle默认dtype是float64的网络下使用报错的bug
2. 修复Windows的CUDA10.1/10.2版本的无法加载CUDA相关dll的bug
3. 修复Tensor在CUDAPinnedPlace与其他Place之间相互拷贝的bug
4. 修复paddle.jit.load载入无参数Layer出错的bug
5. 修复paddle.diag对于大尺寸输入计算错误的bug，修复paddle.diag在Windows Python3.8环境下内存占用异常的bug
6. 修复paddle.topk在静态图组网时输出的shape不合理的问题
7. 修复paddle.io.DataLoader多进程模式经paddle.distributed.spawn启动时直接报错退出的bug
8. 修复paddle.set_device接口设置运行时设备在部分场景中失效的问题
9. 修复paddle.static.nn.while_loop反向计算中使用前向计算的变量而导致的梯度计算错误的bug
10. 修复fleet不支持paddle.optimizer的bug
11. 修复Adam优化器计算公式与论文有diff的bug
12. 修复logsumexp导致部分机器上编译太慢的问题
13. 修复ParamAttr缺失类型检查的问题
14. 修复AvgPool API ceil_mode=true情况下在CPU上平均池化核计算问题
15. 修复paddle.distributed.fleet.init_server()加载模型时维度不匹配的问题
16. 修复paddle.distributed.fleet参数服务器模式下训练节点不支持GPU的问题
17. 修paddle.allclose在float64数据类型下精度diff问题
18. 修复了反向传播支持分组的conv算子（conv2d grad op with groups）的错误 @intel
19. 修复了动转静to_static装饰模型，直接切换eval模式无法保存模型的bug
20. 修复matmul不支持fp16bug
21. 修复matmul反向计算性能差以及显存占比高的问题
22. 修复paddle.nn.Transformer参数bias_attr和weight_attr指定为bool，list/tuple出错问题
23. 修复dynamic_decode预测解码不能正确提前结束的问题
24. 修复paddle.unsqueeze在axis为Tensor的情况下结果错误的问题
25. 修复了paddle.to_tensor在某些场景下zero_copy带来的问题，暂时禁止了zero_copy行为

## 推理

###  Paddle Inference

1. 预测库默认命名从fluid_inference改为paddle_inference

#### 功能升级
1. Paddle-TRT 动态shape功能支持PaddleSlim量化Int8模型
2. Paddle Inference GPU Int8支持conv2d_transpose量化
3. 增加预测模型的算子版本信息
4. 在oneDNN INT8量化策略中增加了对有偏移的scales的量化和反量化的支持
5. 添加了对oneDNN BF16的支持：支持conv2d bf16运算符和gru bf16 op，启用了resnet50 bf16模型推断

#### 性能优化
1. ERNIE模型在T4上使用Paddle-TRT FP16推理性能提升15%。@NVIDIA
2. 通过支持oneDNN FP32 GRU和oneDNN INT8 GRU，GRU INT8模型的速度与NativeConfig推理相比，提高了约1.49倍（线程= 1，batch_size = 50
3. 通过oneDNN升级到1.6，Ernie Large oneDNN在Skylake上(Intel Core 6148）推理的速度提高了约2.7倍（即单元测试 test_analyzer_ernie_large)

#### Bug修复
1. 修复用户使用Paddle Inference ZeroCopyRun接口，开启MKLDNN时，在变长输入下内存泄露的bug
2. 修复ERNIE模型含有共享参数时预测出错的bug
3. 修复带Paddle-TensorRT功能的预测库在未安装TensorRT的环境下初始化报错的bug
4. 修复softmax op、layer_norm op使用Paddle-TRT预测时维度计算错误的bug
5. 解决了增加cpu_math_library_num_threads_数目，预测性能却无法提高的问题（PaddleOCR repository
6. 解决了oneDNN concat重载数据错误的问题
7. 解决了开启oneDNN推理NHWC模型会报错的问题
8. 解决了rec_r34_vd_tps_bilstm_attn模型oneDNN预测失败的问题
9. 解决了deeplabv3p_xception oneDNN预测失败的问题
