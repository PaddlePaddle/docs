# 2.4.2 Release Note

 版本修复了已知问题，并新增了少量功能。

## 训练框架（含分布式）

 * 修复 paddle.utils.dlpack.to_dlpack 在 for 循环里 API 多次创建 dlpack 对象的报错问题，修复引用对象计数错误导致 dlpack 实际指向内容被析构的问题。 [#50138](https://github.com/PaddlePaddle/Paddle/pull/50138)
 * 修复 paddle.multiplex API 在多维 Input Tensor 场景下访存越界的问题并添加 check 机制。 [#49368](https://github.com/PaddlePaddle/Paddle/pull/49368)
 * 引入 cutlass，实现 gemm+gather+scatter 的融合；优化 sparse conv 的训练和推理性能；优化 batch_norm 在 1D 输入数据下的推理性能。 [#50118](https://github.com/PaddlePaddle/Paddle/pull/50118)
 * 修复因使用 constexpr 导致 gcc54 环境下编译失败的问题。 [#50421](https://github.com/PaddlePaddle/Paddle/pull/50421)
 * 将 sum op 的 Kernel 迁移到 PHI 算子库，并且修复 infermeta 中 SelectedRows 无法获取正确 dim 的 bug。 [#49342](https://github.com/PaddlePaddle/Paddle/pull/49342)
 * 修复 eigen 头文件错误引用导致的偶发编译错误。 [#48157](https://github.com/PaddlePaddle/Paddle/pull/48157)
 * 修复 fold 算子在大 bs 输入下访存越界的问题。 [#49491](https://github.com/PaddlePaddle/Paddle/pull/49491)
 * 通过增加类型判别，解决发送张量时，维度不统一，造成流水线并行 hang 住的问题。 [#50337](https://github.com/PaddlePaddle/Paddle/pull/50337)
 * 修复了自定义算子输出梯度的参数顺序不连续时，反向算子的输出值可能为 None 的 bug。 [#48656](https://github.com/PaddlePaddle/Paddle/pull/48656)
 * 修复 paddle.queeze_ API 在 inplace 操作时 shape 重复修改导致结果错误 bug。 [#49903](https://github.com/PaddlePaddle/Paddle/pull/49903)
 * 修复动转静模式下无参数 Layer 无法调用 backward 的问题。 [#49812](https://github.com/PaddlePaddle/Paddle/pull/49812)
 * 修复 CUDA11.8 在 windows 的编译问题。 [#50205](https://github.com/PaddlePaddle/Paddle/pull/50205)
 * 修复 `FusedDropoutActBiasGrad` 在 H100 上不支持的错误。 [#47285](https://github.com/PaddlePaddle/Paddle/pull/47285)
 * 新增 `debug_graphviz_path` 选项至 `build_strategy`。 [#46531](https://github.com/PaddlePaddle/Paddle/pull/46531)
 * 修复未关闭的 `popen` 物件。 [#47053](https://github.com/PaddlePaddle/Paddle/pull/47053)

## 部署方向（Paddle Inference）

 * 完善混合精度推理功能，提高混合精度推理稳定性。重构二阶段式 convert_to_mixed_precision 接口底层实现， enable_use_gpu 新增 precision 参数支持一阶段式。 [#49077](https://github.com/PaddlePaddle/Paddle/pull/49077)、[#49239](https://github.com/PaddlePaddle/Paddle/pull/49239)、[#49477](https://github.com/PaddlePaddle/Paddle/pull/49477)
 * 支持 jetson ampere 架构下编译。 [#49364](https://github.com/PaddlePaddle/Paddle/pull/49364)
 * 修复 fc kernel 低精度模式下的精度问题。 [#49781](https://github.com/PaddlePaddle/Paddle/pull/49781)
 * 修复 CAPI 下， trt workspace 参数类型的错误。 [#48350](https://github.com/PaddlePaddle/Paddle/pull/48350)
 * 修复 Paddle 1.x 版本下 arg_max arg_min 没有 flatten dtype 参数，推理时会报错的问题。 [#49771](https://github.com/PaddlePaddle/Paddle/pull/49771)
 * 修复 split infermeta 重构后关于 lod 逻辑信息缺失问题。 [#49745](https://github.com/PaddlePaddle/Paddle/pull/49745)
 * 修复常量折叠 pass 不正确设置，导致 conv2d 权重经折叠后为非 persistable 而没有进入 TensorRT engine 问题。 [#50105](https://github.com/PaddlePaddle/Paddle/pull/50105)

# 2.4.1 Release Note


去除飞桨对 python.so 的依赖，修复在包括 conda 在内的特定的环境下，因无法找到 python.so 而造成运行失败的 Bug。



# 2.4.0 Release Note

## 1. 重要更新

- **新动态图架构正式生效**：新动态图框架调大幅提升了调度性能，超 90%API 的调度性能提升超过 50%，超 50%套件模型性能提升超过 5%，功能架构更加清晰，二次开发能力和体验显著增强。

- **全面提升了飞桨的动静统一能力：** 动转静功能提供了更加丰富的 Python 语法支持，飞桨的 Python 语法覆盖率达到 90%，对语法转写逻辑进行了重点地优化，完备地支持了控制流语法，提供了更加流畅的一键转静态图体验；借助全新升级的静态图执行器，让动转静训练具有更优的加速能力，重点模型测试显示接近静态图最佳水平；提升了动转静的可扩展性，新增支持多函数合并导出和推理，支持用户使用 PHI 算子库进行二次开发和灵活部署，有效支撑语音领域 U2++特色模型的自定义解码。

- **新增稀疏计算类 API：** 新增 55 个稀疏 API `paddle.sparse.*`，支持稀疏计算主流场景，已应用于 3D 点云目标检测、Sparse Transformers 等任务的稀疏训练和推理部署，高稀疏度场景下相比使用 DenseTensor 提速 105.75%，相比同类产品稀疏计算提速 4.01%~58.55%；支持多种稀疏 Tensor(SparseCoo 和 SparseCsr 等)的计算，极致节省显存；同时保持了一致的使用体验，和稠密 Tensor 的 API 使用方式一致。

- **大规模图神经网络 GPU 训练引擎：** 通过 SSD、内存、显存的异构层次化存储技术，突破显存瓶颈,支持超大规模图的全 GPU 存储和训练；实现了游走、采样、训练的全 GPU 一体化解决方案，相比传统的分布式 CPU 解决方案，相同成本的情况下训练速度提升 10+倍。

- **环境适配：** 新增了适配 CUDA11.7 版本的预编译安装包，新增了支持在 Ubuntu 22.04 及以上版本中运行。

### 前瞻性预告

- 飞桨框架将在 2.5 版本废弃对 python 3.6 的支持。
- 飞桨框架将会逐步废弃 python 端的`paddle.fluild`命名空间下的 API，在 2.5 版本时，部分该命名空间下的 API 将会被直接删除。

## 2. 不兼容升级

- 取消了适配 CUDA10.1 版本的预编译安装包。
- Tensor.clear_gradient(bool set_to_zero)接口不再接收 kwargs 传入的值，只能通过 args 传入 set_to_zero 的 bool 变量。
- 为了提高显存利用效率，动态图默认仅保留前向叶子结点变量的梯度如训练中网络参数的梯度，而不再支持默认保留非叶子结点的梯度。如果需要保留特定 Tensor 的梯度，可以在反向执行前调用 Tensor.retain_grads()接口。
- paddle.autograd.PyLayer 将不再支持输入是 tuple 的情况，如果输入希望是一组 Tensor 的情况请传入 list of Tensor。

## 3. 训练框架（含分布式）

### （1）新增 API 和增强 API 功能
- **新增稀疏计算类 API**：paddle.sparse
  - 新增 55 个稀疏 API，支持稀疏计算主流场景，已应用于 3D 点云目标检测、Sparse Transformers 等任务的稀疏训练和推理部署，高稀疏度场景下相比使用 DenseTensor 提速 105.75%，相比同类产品稀疏计算提速 4.01%~58.55%；支持多种稀疏 Tensor(SparseCoo 和 SparseCsr 等)的计算，极致节省显存；同时保持了一致的使用体验，和稠密 Tensor 的 API 使用方式一致。[#45849](https://github.com/PaddlePaddle/Paddle/pull/45849), [#46694](https://github.com/PaddlePaddle/Paddle/pull/46694), [#45086](https://github.com/PaddlePaddle/Paddle/pull/45086), [#41857](https://github.com/PaddlePaddle/Paddle/pull/41857), [#42935](https://github.com/PaddlePaddle/Paddle/pull/42935), [#43475](https://github.com/PaddlePaddle/Paddle/pull/43475), [#43668](https://github.com/PaddlePaddle/Paddle/pull/43668), [#43966](https://github.com/PaddlePaddle/Paddle/pull/43966), [#44022](https://github.com/PaddlePaddle/Paddle/pull/44022), [#44346](https://github.com/PaddlePaddle/Paddle/pull/44346), [#44432](https://github.com/PaddlePaddle/Paddle/pull/44432), [#44451](https://github.com/PaddlePaddle/Paddle/pull/44451), [#44743](https://github.com/PaddlePaddle/Paddle/pull/44743), [#42013](https://github.com/PaddlePaddle/Paddle/pull/42013), [#43520](https://github.com/PaddlePaddle/Paddle/pull/43520), [#41434](https://github.com/PaddlePaddle/Paddle/pull/41434), [#42130](https://github.com/PaddlePaddle/Paddle/pull/42130), [#41276](https://github.com/PaddlePaddle/Paddle/pull/41276), [#41857](https://github.com/PaddlePaddle/Paddle/pull/41857), [#41356](https://github.com/PaddlePaddle/Paddle/pull/41356)
- **新增语音领域 API：** paddle.audio
  - 新增 MFCC、Spectrogram、LogMelSpectrogram 等特征提取 API，支持 GPU 计算，相比 CPU 实现处理性能提升 15x 倍以上，可大幅提升语音模型训练 GPU 利用率。[#45424](https://github.com/PaddlePaddle/Paddle/pull/45424)
  - 新增窗函数、离散余弦变换等特征提取基础 API，方便用户自定义语音特征提取。[#45424](https://github.com/PaddlePaddle/Paddle/pull/45424)
  - 新增语音 IO 模块，提供 2 种 音频 I/O backend，支持 6 种编解码，便捷地实现语音数据的加载。 [#45939](https://github.com/PaddlePaddle/Paddle/pull/45939)
  - 新增 TESS，ESC50 语音分类数据集，方便用户完成经典语音分类模型。[#45939](https://github.com/PaddlePaddle/Paddle/pull/45939)
- **新增图学习领域 API：** paddle.geometric
  - 图学习逐渐成为机器学习领域的关键技术，飞桨新增 paddle.geometric 模块提供更好的图学习建模和训练开发体验。
    - 消息传递：图学习消息传递机制是图建模的基础，因此新增 7 个图学习消息传递 API，更方便完成进行图学习建模。其中，新增的 3 个消息传递融合算子可大幅减少图模型训练显存占用，稠密图场景下 GCN 系列模型可节省 50%+显存，训练速度可提升 20%+。[#44848](https://github.com/PaddlePaddle/Paddle/pull/44848), [#44580](https://github.com/PaddlePaddle/Paddle/pull/44580), [#43174](https://github.com/PaddlePaddle/Paddle/pull/43174), [#44970](https://github.com/PaddlePaddle/Paddle/pull/44970)
    - 图采样：图采样是图模型训练的性能瓶颈，此次新增了高性能图采样算子，支持高并发图采样，GraphSage 的采样速度可提升 32 倍以上，模型训练速度可提升 12 倍以上。[#44970](https://github.com/PaddlePaddle/Paddle/pull/44970)
- **新增视觉领域 API**
  - paddle.vision 新增目标检测领域算子 paddle.vision.distribute_fpn_proposals([#43736](https://github.com/PaddlePaddle/Paddle/pull/43736)), paddle.vision.generate_proposals([#43611](https://github.com/PaddlePaddle/Paddle/pull/43611)), paddle.vision.matrix_nms([#44357](https://github.com/PaddlePaddle/Paddle/pull/44357)), paddle.vision.prior_box 和 paddle.vision.box_coder([#47282](https://github.com/PaddlePaddle/Paddle/pull/47282))。

- - **新增其他 API**
  - 新增 iinfo([#45321](https://github.com/PaddlePaddle/Paddle/pull/45321)), count_nonzero([#44169](https://github.com/PaddlePaddle/Paddle/pull/44169)), nanmedian([#42385](https://github.com/PaddlePaddle/Paddle/pull/42385)), remainder\_ ([#45266](https://github.com/PaddlePaddle/Paddle/pull/45266)), take([#44741](https://github.com/PaddlePaddle/Paddle/pull/44741)), triu_indices([#45168](https://github.com/PaddlePaddle/Paddle/pull/45168)), sgn([#44568](https://github.com/PaddlePaddle/Paddle/pull/44568)), bucketize([#44195](https://github.com/PaddlePaddle/Paddle/pull/44195)), nanquantile([#41343](https://github.com/PaddlePaddle/Paddle/pull/41343)), frac([#41226](https://github.com/PaddlePaddle/Paddle/pull/41226)), logcumsumexp([#42267](https://github.com/PaddlePaddle/Paddle/pull/42267)), pairwise_distance([#44161](https://github.com/PaddlePaddle/Paddle/pull/44161)), heaviside([#41872](https://github.com/PaddlePaddle/Paddle/pull/41872)), logspace([#41261](https://github.com/PaddlePaddle/Paddle/pull/41261)), corrcoef([#40690](https://github.com/PaddlePaddle/Paddle/pull/40690))
  - 新增 RReLU([#41823](https://github.com/PaddlePaddle/Paddle/pull/41823)), CyclicLR([#40698](https://github.com/PaddlePaddle/Paddle/pull/40698)), OneCycleLR([#41825](https://github.com/PaddlePaddle/Paddle/pull/41825)), Softmax2D([#40910](https://github.com/PaddlePaddle/Paddle/pull/40910)), SoftMarginLoss([#42364](https://github.com/PaddlePaddle/Paddle/pull/42364)), MultiLabelSoftMarginLoss([#41183](https://github.com/PaddlePaddle/Paddle/pull/41183)), TripletMarginLoss([#40487](https://github.com/PaddlePaddle/Paddle/pull/40487)), TripletMarginWithDistanceLoss([#40545](https://github.com/PaddlePaddle/Paddle/pull/40545)), CosineEmbeddingLoss 和 cosine_embedding_loss([#41680](https://github.com/PaddlePaddle/Paddle/pull/41680)), PixelUnshuffle([#40728](https://github.com/PaddlePaddle/Paddle/pull/40728)), ChannelShuffle([#40743](https://github.com/PaddlePaddle/Paddle/pull/40743))
- **增强 API 功能**
  - 增加 BatchNorm1D 的大 batch_size 计算功能 [#43072](https://github.com/PaddlePaddle/Paddle/pull/43072)
- **完善集合通信分布式训练 API**
  - 完善`fleet.init`函数，增加`log_level`参数，方便用户查看运行过程中的日志 [#45909](https://github.com/PaddlePaddle/Paddle/pull/45909)
  - 新增`paddle.distributed.fleet.recompute_sequential paddle.distributed.fleet.recompute_hybrid`接口，方便用户使用 recompute 功能[#45348](https://github.com/PaddlePaddle/Paddle/pull/45348)
  - 新增`paddle.distributed.fleet.layers.mpu` package，方便用户使用张量并行功能 [#45803](https://github.com/PaddlePaddle/Paddle/pull/45803)
  - 新增通信 API `paddle.distributed.destroy_process_group paddle.distributed.isend paddle.distributed.irecv paddle.distributed.all_to_all_single`，提升了通信的功能完备性和易用性 [#43918](https://github.com/PaddlePaddle/Paddle/pull/43918)
  - 新增`paddle.distributed.stream` 通信 package，性能比基础版本提升 5%到 10% [#46023](https://github.com/PaddlePaddle/Paddle/pull/46023) [#45282](https://github.com/PaddlePaddle/Paddle/pull/45282)
  - 通信 API 新增多种数据类型`Char/Byte/Bool`等的支持，提升了通信的功能完备性和易用性 [#45574](https://github.com/PaddlePaddle/Paddle/pull/45574) [#45440](https://github.com/PaddlePaddle/Paddle/pull/45440)
  - 通信 API 异步参数从`use_calc_stream`变成`sync_op`，增强了接口的语义可读性 [#46493](https://github.com/PaddlePaddle/Paddle/pull/46493)
- **增强高层 API**
  - 高层 API 中视觉模型 ResNeXt 实现复用 ResNet 代码进行重构。 [#40588](https://github.com/PaddlePaddle/Paddle/pull/40588)
  - 高层 API 中视觉模型 Inceptionv3、MobileNetv1、MobileNetv2、ShuffleNetv2 实现改进。[#40431](https://github.com/PaddlePaddle/Paddle/pull/40431)

### （2）新功能及重要功能升级

- **新动态图架构正式上线**：新动态图框架调度性能大幅提升，相比原有架构大幅提升了调度性能，超 90%API 的调度性能提升超过 50%，超 50%套件模型性能提升超过 5%; 新动态图架构清晰，耦合度低，基于新架构实现 Hook、PyLayer 等扩展模块的学习与开发成本显著降低。[#37550](https://github.com/PaddlePaddle/Paddle/pull/37550)，[#37574](https://github.com/PaddlePaddle/Paddle/pull/37574)，[#37813](https://github.com/PaddlePaddle/Paddle/pull/37813)，[#37926](https://github.com/PaddlePaddle/Paddle/pull/37926)，[#39192](https://github.com/PaddlePaddle/Paddle/pull/39192)，[#37599](https://github.com/PaddlePaddle/Paddle/pull/37599)，[#37406](https://github.com/PaddlePaddle/Paddle/pull/37406)，[#37466](https://github.com/PaddlePaddle/Paddle/pull/37466)，[#37599](https://github.com/PaddlePaddle/Paddle/pull/37599)，[#40945](https://github.com/PaddlePaddle/Paddle/pull/40945)，[#39989](https://github.com/PaddlePaddle/Paddle/pull/39989)

- **高阶自动微分机制**：为了更好支持科学计算等场景，飞桨框架针对高阶自动微分能力进一步完善优化。目前，已在`paddle.incubate.autograd` 目录下提供了支持前反向高阶自动微分相关试用功能及 API（当前处于孵化状态，相关功能及 API 签名可能会发生变化）。如果想自行实现相关模型、探索自动微分机制，请仔细阅读[高阶自动微分使用方法及限制](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/incubate/autograd/Overview_cn.html)。具体的升级包括：
  1. 静态图高阶微分机制升级，通过基础算子体系和程序变换，支持高阶前向及反向微分，并打通编译器、分布式功能。[#41919](https://github.com/PaddlePaddle/Paddle/pull/41919), [#41201](https://github.com/PaddlePaddle/Paddle/pull/41201)
  2. 新增前向和反向高阶自动微分 API， `paddle.incubate.autograd.forward_grad`, `paddle.incubate.autograd.grad`。[#43354](https://github.com/PaddlePaddle/Paddle/pull/43354)
  3. 新增 18 个高阶自动微分算子`sin`, `cos`, `exp`, `erf`, `abs`, `log`, `cast`, `where`, `equal`, `not_equal`, `greater_than`, `greater_equal`, `elementwise_pow` `square`, `elementwise_max`, `gelu`, `reduce_mean`, `size`。[#46184](https://github.com/PaddlePaddle/Paddle/pull/46184), [#46024](https://github.com/PaddlePaddle/Paddle/pull/46024), [#45888](https://github.com/PaddlePaddle/Paddle/pull/45888), [#45338](https://github.com/PaddlePaddle/Paddle/pull/45338), [#44345](https://github.com/PaddlePaddle/Paddle/pull/44345)
  4. 修复现有`elementwise_div`, `reduce_sum`, `p_norm`等算子缺陷。[#46514](https://github.com/PaddlePaddle/Paddle/pull/46514), [#46184](https://github.com/PaddlePaddle/Paddle/pull/46184)

- **通用异构参数服务器架构**：
  - 参数服务器 GPUGraph 基础架构升级，满足大规模应用落地：针对传统 CPU 存储和训练大规模图神经网络的成本高，稳定性低，性能不足的问题打造了纯 GPU 图训练引擎（PGLBox），通过 SSD、内存、显存的异构层次化存储技术，支持超大规模图模型训练，同等成本下训练性能相对 CPU 图训练引擎提升 10+倍，任务失败率下降到极低。[#44594](https://github.com/PaddlePaddle/Paddle/pull/44594)
  - 大规模联邦参数服务器架构：针对大规模个性化推荐场景，基于异构 PS 基础架构，开发了大规模联邦参数服务器训练，支持千亿参数下的横向纵向联邦，它包括两个特性：用户私有参数本地更新，公共参数在远端更新，用户可灵活配置私有参数和公共参数的切分策略；新增中心调度节点 Coordinator，用户可从基类进行二次开发，自定义 Client 选择策略。[#42682](https://github.com/PaddlePaddle/Paddle/pull/42682)，[#44864](https://github.com/PaddlePaddle/Paddle/pull/44864)，[#44327](https://github.com/PaddlePaddle/Paddle/pull/44327)
- **自适应并行**
  - 设计并推出了完善的自动并行接口体系，支持自动动转静分布式训练、自动分布式数据加载、自动分布式保存与加载、自动参数转换、自定义切分标记和自定义执行过程等。用户只需要基于单机组网就可以非常容易获得自动分布式训练能力，支持数据并行、模型并行、流水线并行和混合并行。[#45776](https://github.com/PaddlePaddle/Paddle/pull/45776) ，[#46552](https://github.com/PaddlePaddle/Paddle/pull/46552)，[#44202](https://github.com/PaddlePaddle/Paddle/pull/44202)，[#45840](https://github.com/PaddlePaddle/Paddle/pull/45840)，[#45518](https://github.com/PaddlePaddle/Paddle/pull/45518)，[#40528](https://github.com/PaddlePaddle/Paddle/pull/40528)，[#42838](https://github.com/PaddlePaddle/Paddle/pull/42838)，[#43093](https://github.com/PaddlePaddle/Paddle/pull/43093)，[#43312](https://github.com/PaddlePaddle/Paddle/pull/43312)，[#45053](https://github.com/PaddlePaddle/Paddle/pull/45053)。
  - 完善了自适应并行底层机制，包括升级分布式 cost model 设计和实现，为切分策略提供更好评价；为 Program IR 添加了原生分布式属性，丰富了 Cluster 功能。[#40457](https://github.com/PaddlePaddle/Paddle/pull/40457)，[#42601](https://github.com/PaddlePaddle/Paddle/pull/42601)，[#42727](https://github.com/PaddlePaddle/Paddle/pull/42727)，[#42874](https://github.com/PaddlePaddle/Paddle/pull/42784)，[#43114](https://github.com/PaddlePaddle/Paddle/pull/43114)，[#44095](https://github.com/PaddlePaddle/Paddle/pull/44095)，[#44146](https://github.com/PaddlePaddle/Paddle/pull/44146)，[#44701](https://github.com/PaddlePaddle/Paddle/pull/44701)，[#44973](https://github.com/PaddlePaddle/Paddle/pull/44973)，[#45002](https://github.com/PaddlePaddle/Paddle/pull/45002)，[#45118](https://github.com/PaddlePaddle/Paddle/pull/45118)，[#45237](https://github.com/PaddlePaddle/Paddle/pull/45237)，[#42576](https://github.com/PaddlePaddle/Paddle/pull/42576)，[#41722](https://github.com/PaddlePaddle/Paddle/pull/41722)，[#44150](https://github.com/PaddlePaddle/Paddle/pull/44150)， [#44989](https://github.com/PaddlePaddle/Paddle/pull/44989)， [#44951](https://github.com/PaddlePaddle/Paddle/pull/44951)， [#44963](https://github.com/PaddlePaddle/Paddle/pull/44963)。
  - 新增数据并行下 Sharding stage1/2/3 自动调优功能，在保证满足显存约束情况下，自动选择吞吐最高的 Sharding stage 策略。[#43782](https://github.com/PaddlePaddle/Paddle/pull/43782)。

- **训练硬件接入-插件式方案**：新增了自定义 Runtime/Kernel/CCL/Graph/Pass 等方案，硬件厂商可以根据硬件特性按需选择实现哪些模块。

- **ONNX 格式导出**
  - 支持量化模型导出，导出后的 ONNX 模型使用 TensorRT 或 ONNXRuntime 加载推理，可获得 1.5~4 倍的推理加速 [#856](https://github.com/PaddlePaddle/Paddle2ONNX/pull/856)，[#782](https://github.com/PaddlePaddle/Paddle2ONNX/pull/782)
  - 新增大于 2GB 的大模型导出 [#942](https://github.com/PaddlePaddle/Paddle2ONNX/pull/942)

### （3）功能优化
- **动转静分析转换 & 扩展能力全面提升**
  - 为了提升模型动转静转换成功率和使用体验，重构了控制流语法的转写逻辑，升级核心语法为 JIT （just-in-time）范式，实现与 Python 代码的等价转写，并完善了 break、return、continue 等语法功能。[#43666](https://github.com/PaddlePaddle/Paddle/pull/43666)，[#43846](https://github.com/PaddlePaddle/Paddle/pull/43846)，[#43848](https://github.com/PaddlePaddle/Paddle/pull/43848)，[#43880](https://github.com/PaddlePaddle/Paddle/pull/43880)，[#43957](https://github.com/PaddlePaddle/Paddle/pull/43957)，[#43328](https://github.com/PaddlePaddle/Paddle/pull/43328)，[#43348](https://github.com/PaddlePaddle/Paddle/pull/43348)，[#43998](https://github.com/PaddlePaddle/Paddle/pull/43998)，[#44465](https://github.com/PaddlePaddle/Paddle/pull/44465)，[#44504](https://github.com/PaddlePaddle/Paddle/pull/44504)，[#43713](https://github.com/PaddlePaddle/Paddle/pull/43713)，[#43864](https://github.com/PaddlePaddle/Paddle/pull/43864)，[#43967](https://github.com/PaddlePaddle/Paddle/pull/43967)，[#44155](https://github.com/PaddlePaddle/Paddle/pull/44155)，[#44487](https://github.com/PaddlePaddle/Paddle/pull/44487)，[#44527](https://github.com/PaddlePaddle/Paddle/pull/44527)，[#45105](https://github.com/PaddlePaddle/Paddle/pull/45105)，[#45900](https://github.com/PaddlePaddle/Paddle/pull/45900)
  - 为了支撑语音等场景自定义解码灵活部署场景，扩展了 jit.save/load 接口功能，支持用户多函数合并导出，并新增了 JITLayer 组件，支持类函数式调用，同时配合 PHI 算子库 C++ API 实现了自定义推理部署功能。[#44283](https://github.com/PaddlePaddle/Paddle/pull/44283)，[#41783](https://github.com/PaddlePaddle/Paddle/pull/41783)，[#43607](https://github.com/PaddlePaddle/Paddle/pull/43607)，[#43754](https://github.com/PaddlePaddle/Paddle/pull/43754)，[#43758](https://github.com/PaddlePaddle/Paddle/pull/43758)，[#43798](https://github.com/PaddlePaddle/Paddle/pull/43798)，[#44010](https://github.com/PaddlePaddle/Paddle/pull/44010)，[#44351](https://github.com/PaddlePaddle/Paddle/pull/44351)，[#44465](https://github.com/PaddlePaddle/Paddle/pull/44465)，[#44504](https://github.com/PaddlePaddle/Paddle/pull/44504)，[#44597](https://github.com/PaddlePaddle/Paddle/pull/44597)，[#44738](https://github.com/PaddlePaddle/Paddle/pull/44738)，[#44984](https://github.com/PaddlePaddle/Paddle/pull/44984)，[#46249](https://github.com/PaddlePaddle/Paddle/pull/46249)
  - 为了统一 API 动静行为，升级了 20 个算子，支持在静态图中 Op 的 attribute 信息可变，保证动静行为一致，提升模型的动转静转换成功率。包括`pad2d`、`depthwise_conv2d_transpose`、`conv2d_transpose`、`adaptive_avg_pool2d`、`reverse`、`bincount`、`multinomial`、`reduce_sum`、`reduce_mean`、`reduce_prod`、`reduce_min`、`reduce_max`、`uniform`、`squeeze`、`max_unpool2d`、`dropout`、`cumsum`、`eye`、`argmin`、`argmax`，[#44737](https://github.com/PaddlePaddle/Paddle/pull/44737)，[#45084](https://github.com/PaddlePaddle/Paddle/pull/45084)，[#45189](https://github.com/PaddlePaddle/Paddle/pull/45189)，[#45391](https://github.com/PaddlePaddle/Paddle/pull/45391)，[#45417](https://github.com/PaddlePaddle/Paddle/pull/45417)，[#45427](https://github.com/PaddlePaddle/Paddle/pull/45427)、[#45514](https://github.com/PaddlePaddle/Paddle/pull/45514)、[#45525](https://github.com/PaddlePaddle/Paddle/pull/45525)、[#45543](https://github.com/PaddlePaddle/Paddle/pull/45543)、[#45660](https://github.com/PaddlePaddle/Paddle/pull/45660)、[#46352](https://github.com/PaddlePaddle/Paddle/pull/46352/)、[#46433](https://github.com/PaddlePaddle/Paddle/pull/46433)、[#45078](https://github.com/PaddlePaddle/Paddle/pull/45078)，[#45342](https://github.com/PaddlePaddle/Paddle/pull/45342)，[#45372](https://github.com/PaddlePaddle/Paddle/pull/45372)，[#45453](https://github.com/PaddlePaddle/Paddle/pull/45453)，[#45522](https://github.com/PaddlePaddle/Paddle/pull/45522)，[#45620](https://github.com/PaddlePaddle/Paddle/pull/45620)
  - 为了解决用户动转静报错栈偶尔丢失问题，优化了报错模块的逻辑，提升了报错栈的可读性以及用户调试的使用体验。[#44054](https://github.com/PaddlePaddle/Paddle/pull/44054)，[#44083](https://github.com/PaddlePaddle/Paddle/pull/44083)，[#44781](https://github.com/PaddlePaddle/Paddle/pull/44781)，[#44996](https://github.com/PaddlePaddle/Paddle/pull/44996)
  - 为了全面支持 Python 类型 Type Hint 语法，新增了 TypeHint 语法识别和转写模块。[#47121](https://github.com/PaddlePaddle/Paddle/pull/47121)

- **PHI 算子库覆盖全量运算类算子**：继续建设高可复用算子库 PHI，将剩余的飞桨 2.x 运算类 PythonAPI 关联的算子以及相关内核均迁移到 PHI 算子库，并改写为函数式，新增了约 180 个前反向算子的 CPU&GPU 内核，以及 170 个 Kunlun 专用算子内核，进一步提升了新增算子时可复用的内核函数集。同时，新增了 100 余个 C++运算类 API，可支持在自定义算子中使用，进一步提升了基于飞桨进行外部扩展开发的易用性。[#44577](https://github.com/PaddlePaddle/Paddle/pull/44577)，[#44631](https://github.com/PaddlePaddle/Paddle/pull/44631)，[#44434](https://github.com/PaddlePaddle/Paddle/pull/44434)，[#44605](https://github.com/PaddlePaddle/Paddle/pull/44605)，[#44676](https://github.com/PaddlePaddle/Paddle/pull/44676)，[#44742](https://github.com/PaddlePaddle/Paddle/pull/44742)，[#44436](https://github.com/PaddlePaddle/Paddle/pull/44436)，[#45887](https://github.com/PaddlePaddle/Paddle/pull/45887)，[#45851](https://github.com/PaddlePaddle/Paddle/pull/45851)，[#45623](https://github.com/PaddlePaddle/Paddle/pull/45623)，[#45397](https://github.com/PaddlePaddle/Paddle/pull/45397)，[#45863](https://github.com/PaddlePaddle/Paddle/pull/45863)

- **规范化算子定义，大幅提升模型简洁度**：针对飞桨 1.x 历史算子定义存在诸多冗余参数，理解适配成本高的问题，对约 150 个高频算子的冗余参数进行了集中清理，基本上将数学无关的参数清理完毕。这些冗余参数清理后，飞桨存储的推理模型中信息量明显减少，普遍裁减掉了约 40%的属性变量，显著提升了飞桨算子定义的清晰程度，提升了模型分析调试的体验；同时，也显著减小了飞桨存储推理模型的体积，普遍减小超过 70%，显著提升了飞桨模型的轻量化程度。[#44310](https://github.com/PaddlePaddle/Paddle/pull/44310) , [#45613](https://github.com/PaddlePaddle/Paddle/pull/45613) , [#45684](https://github.com/PaddlePaddle/Paddle/pull/45684) , [#45708](https://github.com/PaddlePaddle/Paddle/pull/45708) , [#45758](https://github.com/PaddlePaddle/Paddle/pull/45758) , [#45786](https://github.com/PaddlePaddle/Paddle/pull/45786) , [#45772](https://github.com/PaddlePaddle/Paddle/pull/45772) , [#45845](https://github.com/PaddlePaddle/Paddle/pull/45845) , [#45984](https://github.com/PaddlePaddle/Paddle/pull/45984) , [#46218](https://github.com/PaddlePaddle/Paddle/pull/46218) , [#46553](https://github.com/PaddlePaddle/Paddle/pull/46553)

### （4）性能优化

- AMP 性能及精度优化
  - 更多算子增加 FP16 数据类型支持，包括 elementwise 系列算子, compare 系列算子, strided_slice, set_value, uniform_ramdom 等。（[#45504](https://github.com/PaddlePaddle/Paddle/pull/45504) [#44405](https://github.com/PaddlePaddle/Paddle/pull/44405) [#45496](https://github.com/PaddlePaddle/Paddle/pull/45496) [#46641](https://github.com/PaddlePaddle/Paddle/pull/46641) [#46906](https://github.com/PaddlePaddle/Paddle/pull/46906)）
  - 优化 hard_swish 算子 FP16 Kernel 实现方案，保证精度无损。（ [35386](https://github.com/PaddlePaddle/Paddle/pull/35386) ）
  - 更多算子增加 BF16 数据类型支持，包括 fused_linear、empty、selu、pow、adam、clip、embedding、gelu、pad3d、pixel_shuffle、tile、where 等。[#46364](https://github.com/PaddlePaddle/Paddle/pull/46364)，[#47177](https://github.com/PaddlePaddle/Paddle/pull/47177)
- 单机训练性能自动调优
  - Transpose OP 支持自动 Kernel 选择机制，可以针对不同模型配置自动搜索到性能最优的 Kernel 实现，提升模型性能。[#43310](https://github.com/PaddlePaddle/Paddle/pull/43310) (Transpose Op 接入自动调优功能)
  - AMP Layout 自动切换支持新动态图模式，ResNet50、TSM、DeepLabV3 等模型在新动态图下通过 Layout 自动调整获得性能提升 9%~21%。([#45409](https://github.com/PaddlePaddle/Paddle/pull/45409), [#45751](https://github.com/PaddlePaddle/Paddle/pull/45751), [#45826](https://github.com/PaddlePaddle/Paddle/pull/45826), [#46880](https://github.com/PaddlePaddle/Paddle/pull/46880))
- GPU 单机训练通用性能优化
  - 优化 Conv 类算子 cuDNN 算法的 Cache 方案，并 Cache 所有算法获取方式下的结果，大幅减少算子的 CPU 开销。（[#41891](https://github.com/PaddlePaddle/Paddle/pull/41891) [#47197](https://github.com/PaddlePaddle/Paddle/pull/47197)）
  - 进一步优化多个算子的 GPU Kernel 和 Python 端性能，包括 dist, poisson, depthwise_conv2d、transpose, eigh, broadcast 类计算，reduce 类计算，layer_norm，cross_entropy 等，在更多配置场景下达到更优性能。（[#44946](https://github.com/PaddlePaddle/Paddle/pull/44946), [#45057](https://github.com/PaddlePaddle/Paddle/pull/45057), [#45160](https://github.com/PaddlePaddle/Paddle/pull/45160), [#42491](https://github.com/PaddlePaddle/Paddle/pull/42491), [#42704](https://github.com/PaddlePaddle/Paddle/pull/42704), [#42853](https://github.com/PaddlePaddle/Paddle/pull/42853), [#46287](https://github.com/PaddlePaddle/Paddle/pull/46287), [#46362](https://github.com/PaddlePaddle/Paddle/pull/46362), [#46490](https://github.com/PaddlePaddle/Paddle/pull/46490), [#46412](https://github.com/PaddlePaddle/Paddle/pull/46412), [#46623](https://github.com/PaddlePaddle/Paddle/pull/46623), [#40051](https://github.com/PaddlePaddle/Paddle/pull/40051)）
- 集合通信分布式训练性能优化
  - 为提高流水线并行调度效率，支持动态图 Interleaving 1F1B 调度策略，在 GPT-3 模型上性能提升 3%~4%。[#45797](https://github.com/PaddlePaddle/Paddle/pull/45797)，[#45869](https://github.com/PaddlePaddle/Paddle/pull/45869)，[#45922](https://github.com/PaddlePaddle/Paddle/pull/45922)，[#46209](https://github.com/PaddlePaddle/Paddle/pull/46209)，[#45402](https://github.com/PaddlePaddle/Paddle/pull/45402)，[#45444](https://github.com/PaddlePaddle/Paddle/pull/45444)，[#45497](https://github.com/PaddlePaddle/Paddle/pull/45497)，[#45797](https://github.com/PaddlePaddle/Paddle/pull/45797)，[#45869](https://github.com/PaddlePaddle/Paddle/pull/45869)，[#45922](https://github.com/PaddlePaddle/Paddle/pull/45922)，[#46209](https://github.com/PaddlePaddle/Paddle/pull/46209)，[#46399](https://github.com/PaddlePaddle/Paddle/pull/46399)，[#46483](https://github.com/PaddlePaddle/Paddle/pull/46483)，[#46876](https://github.com/PaddlePaddle/Paddle/pull/46876)，[#47242](https://github.com/PaddlePaddle/Paddle/pull/47242)，[#47249](https://github.com/PaddlePaddle/Paddle/pull/47249)，[#47497](https://github.com/PaddlePaddle/Paddle/pull/47497)，[#47517](https://github.com/PaddlePaddle/Paddle/pull/47517)
  - 为提升 MLPerf BERT 模型的分布式训练性能，DistributedFusedLamb 分布式优化器支持分层 AllReduce，在 DCU 1024 卡上 MLPerf BERT 性能提升 17%。[#44821](https://github.com/PaddlePaddle/Paddle/pull/44821)，[#44843](https://github.com/PaddlePaddle/Paddle/pull/44843)
  - 为优化使用数据并行 Data Parallel 时的显存占用，支持 Tensor Fusion 时的 Buffer Lazy 初始化策略，可降低等于模型参数量的显存占用量。[#45631](https://github.com/PaddlePaddle/Paddle/pull/45631)。
  - 分布式并行策略 Data Parallel 和 Sharding 支持 BF16 训练。[#46846](https://github.com/PaddlePaddle/Paddle/pull/46846)，[#47246](https://github.com/PaddlePaddle/Paddle/pull/47246)
  - 为支持 Sequence Parallel 等策略，分布式流水线并行策略支持 enable_partial_send_recv 策略，支持传输 sequence parallel 切分后的 tensor。[#46992](https://github.com/PaddlePaddle/Paddle/pull/46992)，[#47083](https://github.com/PaddlePaddle/Paddle/pull/47083)
  - 为提升 sharding stage 2 策略的性能，实现了 sharding stage 2 optimizer broadcast 参数与下一个 step forward 的 overlap，并使用多 CUDA Stream 进行通信，GPT 6.7B 模型 16 卡训练性能提升 11%。[#46495](https://github.com/PaddlePaddle/Paddle/pull/46495)，[#46656](https://github.com/PaddlePaddle/Paddle/pull/46656)，[#47061](https://github.com/PaddlePaddle/Paddle/pull/47061)

### （5）问题修复

- 动转静
  - 修复了模型在多卡训练时 Parameter 无梯度场景下，动转静会报错的问题。[#44485](https://github.com/PaddlePaddle/Paddle/pull/44485)
  - 修复了动转静时终端会有多余的框架日志误输出的问题。[#45754](https://github.com/PaddlePaddle/Paddle/pull/45754)，[#46800](https://github.com/PaddlePaddle/Paddle/pull/46800)
  - 修复了模型中控制流中包含无需梯度的 Tensor 时，在动转静训练时会报错的问题。[#43034](https://github.com/PaddlePaddle/Paddle/pull/43034)
  - 修复了动转静训练在梯度聚合时计算值错误的问题。[#44893](https://github.com/PaddlePaddle/Paddle/pull/44893)
  - 修复了函数被@staticmethod 装饰时动转静会报错的问题。[#44983](https://github.com/PaddlePaddle/Paddle/pull/44983)，[#45268](https://github.com/PaddlePaddle/Paddle/pull/45268)，[#45277](https://github.com/PaddlePaddle/Paddle/pull/45277)
  - 修复了部分场景下模型包含控制动转静训练时，显存占用过多的问题。[#45380](https://github.com/PaddlePaddle/Paddle/pull/45380)
  - 修复了模型中包含复杂控制流时，动转静在组网阶段 shape 推导报错的问题。[#45916](https://github.com/PaddlePaddle/Paddle/pull/45916)，[#46020](https://github.com/PaddlePaddle/Paddle/pull/46020)
- 报错机制修复
  - 使用 np.testing.assert_allclose 替换 self.assertTrue(np.allclose(...))，获得更充分的报错信息 ([#44947)(https://github.com/PaddlePaddle/Paddle/pull/44947)， [#44988](https://github.com/PaddlePaddle/Paddle/pull/44988)，[#45213](https://github.com/PaddlePaddle/Paddle/pull/45213))
- 集合通信分布式训练
  - 修复了通信库初始化、通信过程中的若干 bug，增强了系统运行稳定性 [#44964](https://github.com/PaddlePaddle/Paddle/pull/44964) [#45100](https://github.com/PaddlePaddle/Paddle/pull/45100) [#44758](https://github.com/PaddlePaddle/Paddle/pull/44758)
  - 修复流水线并行容易 hang 的问题，增强策略的易用性 [#47201](https://github.com/PaddlePaddle/Paddle/pull/47201)；增强流水线功能支持不均衡的输入 [#47199](https://github.com/PaddlePaddle/Paddle/pull/47199)
  - 修复新动态图 MP/PP 策略下性能低于老动态图的问题 [#47071](https://github.com/PaddlePaddle/Paddle/pull/47071)
  - 修复 sharding stage2 策略错误维护参数 trainable 属性的 bug [#47240](https://github.com/PaddlePaddle/Paddle/pull/47240)
  - 修复一系列 OP 在 tensor numel 大于 INT32_MAX 时的 bug。[#45711](https://github.com/PaddlePaddle/Paddle/pull/45711)，[#45741](https://github.com/PaddlePaddle/Paddle/pull/45741)，[#45897](https://github.com/PaddlePaddle/Paddle/pull/45897)，[#46158](https://github.com/PaddlePaddle/Paddle/pull/46158)，[#46767](https://github.com/PaddlePaddle/Paddle/pull/46767)，[#47191](https://github.com/PaddlePaddle/Paddle/pull/47191)，[#46045](https://github.com/PaddlePaddle/Paddle/pull/46045)，[#46160](https://github.com/PaddlePaddle/Paddle/pull/46160)
  - 修复 FusedAttention 和 FusedFeedForward OP 显存占用过大的 bug。[#47236](https://github.com/PaddlePaddle/Paddle/pull/47236)，[#47235](https://github.com/PaddlePaddle/Paddle/pull/47235)
  - 修复 multi_tensor_adam 和 multi_tensor_momentum OP 在传入的 parameters 是 list of dict 时参数更新错误的 bug。[#47352](https://github.com/PaddlePaddle/Paddle/pull/47352)，[#47372](https://github.com/PaddlePaddle/Paddle/pull/47372)

## 4. 部署方向（Paddle Inference）

### （1）新增特性

- 后端图引擎集成方案优化
  - 为了减少 Paddle-TensorRT 插件代码开发，以及减少 Paddle-TensorRT 子图数量从而降低资源占用率，开发了通用插件机制，可以自动对框架内丰富的 Phi 算子提供统一的 TensorRT 插件接口，在多数场景下可以有效减少显存占用。 [#46970](https://github.com/PaddlePaddle/Paddle/pull/46070)，[#46179](https://github.com/PaddlePaddle/Paddle/pull/46179)，[#46580](https://github.com/PaddlePaddle/Paddle/pull/46580)
  - 为了方便用户在框架定制算子且能使得 Paddle-TensorRT 高效推理，进行功能升级支持升级框架自定义 Paddle-TensorRT 插件。[#46970](https://github.com/PaddlePaddle/Paddle/pull/46070)
- Inference 推理库构建系统优化，体积可按需裁剪
  - 预编译的安装包默认支持 TensorRT：训练用的预编译安装包与部署用的预编译安装包（Paddle Inference）统一为一个预编译安装包，且优化了构建系统，使得预编译的安装包默认支持 TensorRT，减少用户使用 PaddleTensorRT 时的切换成本。[#46008](https://github.com/PaddlePaddle/Paddle/pull/46008)，[#45824](https://github.com/PaddlePaddle/Paddle/pull/45824)，[#46058](https://github.com/PaddlePaddle/Paddle/pull/46058)
  - 体积可按需裁剪：可依据模型算子进行裁剪。[#47033](https://github.com/PaddlePaddle/Paddle/pull/47033) , [#47049](https://github.com/PaddlePaddle/Paddle/pull/47049) , [#47047](https://github.com/PaddlePaddle/Paddle/pull/47047)
- Inference 支持原生 AMP
  - 为了充分利用 GPU Tensor Core 计算能力，提升模型的推理性能，开发了模型精度转换工具，Inference GPU 原生支持了混合精度模型的推理。使用方式可参考[文档](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/release/v2.4/docs-official/guides/nv_gpu_infer/gpu_mixed_precision.md)。[#43814](https://github.com/PaddlePaddle/Paddle/pull/43814)，[#43881](https://github.com/PaddlePaddle/Paddle/pull/43881)，[#44057](https://github.com/PaddlePaddle/Paddle/pull/44057)，[#44307](https://github.com/PaddlePaddle/Paddle/pull/44307)，[#44457](https://github.com/PaddlePaddle/Paddle/pull/44457)，[#44866](https://github.com/PaddlePaddle/Paddle/pull/44866)，[#45050](https://github.com/PaddlePaddle/Paddle/pull/45050)，[#45346](https://github.com/PaddlePaddle/Paddle/pull/45346)，[#45379](https://github.com/PaddlePaddle/Paddle/pull/45379)，[#45406](https://github.com/PaddlePaddle/Paddle/pull/45406)，[#45882](https://github.com/PaddlePaddle/Paddle/pull/45882)
  - 为了提升混合精度下模型的推理性能，补充了未支持 FP16 计算的高频算子的 FP16 kernel，减少了由于输入精度不匹配插入 cast 算子的可能性，提升推理性能。[#44642](https://github.com/PaddlePaddle/Paddle/pull/44642)，[#45061](https://github.com/PaddlePaddle/Paddle/pull/45061)，[#44653](https://github.com/PaddlePaddle/Paddle/pull/44653)，[#45504](https://github.com/PaddlePaddle/Paddle/pull/45504)，[#45061](https://github.com/PaddlePaddle/Paddle/pull/45061)，[#44969](https://github.com/PaddlePaddle/Paddle/pull/44969)，[#44558](https://github.com/PaddlePaddle/Paddle/pull/44558)，[#44710](https://github.com/PaddlePaddle/Paddle/pull/44710)，[#43871](https://github.com/PaddlePaddle/Paddle/pull/43871)，[#44792](https://github.com/PaddlePaddle/Paddle/pull/44792)
- 压缩与推理引擎打通升级
  - 升级量化模型存储格式，新格式支持 Paddle Inference、PaddleLite 和 Paddle2ONNX 3 种部署方式，支持芯片类型包括 X86 CPU、NVIDIA GPU、Arm CPU。（[#46305](https://github.com/PaddlePaddle/Paddle/pull/46305) [#462832](https://github.com/PaddlePaddle/Paddle/pull/46283) [#46022](https://github.com/PaddlePaddle/Paddle/pull/46022)）
  - 新增兼容 SoC/NPU 芯片的 INT8 全量化功能，可保证产出的 INT8 量化模型在 SoC/NPU 芯片上有最佳推理加速和精度。
- 推理引擎与飞桨编译器（CINN）打通升级
    - 升级飞桨框架与编译器的接口模块，支持推理模型通过 Paddle Inference 接入编译器进行优化（[#44499](https://github.com/PaddlePaddle/Paddle/pull/44499) [#44708](https://github.com/PaddlePaddle/Paddle/pull/44708) ）

### （2）底层优化

- **GPU 性能优化**
  - 新增 matmul_v2、LSTM、reshape、fill_constant、swish、mulitclass_nms3、bilinear_interp_v2、split、silu、shuffle_channel 算子的 TensorRT 映射及完善动态 shape 的支持。多类重点模型性能提升 7%～90% 。([#46177](https://github.com/PaddlePaddle/Paddle/pull/46177)，[#44678](https://github.com/PaddlePaddle/Paddle/pull/44678)，[#44314](https://github.com/PaddlePaddle/Paddle/pull/44314)，[#44561](https://github.com/PaddlePaddle/Paddle/pull/44561)，[#45166](https://github.com/PaddlePaddle/Paddle/pull/45166), [#44411](https://github.com/PaddlePaddle/Paddle/pull/44411)，[#43424](https://github.com/PaddlePaddle/Paddle/pull/43424), [#44516](https://github.com/PaddlePaddle/Paddle/pull/44516))
  - 增加常量折叠 PASS 进行推理性能优化，提升 SwinTransformer、HifiGAN、FastSpeech2 等模型的性能。（[#45494](https://github.com/PaddlePaddle/Paddle/pull/45494))
  - 增加 conv_fusion workspacesize 的 cache，提升 conv_fusion 计算性能。([#45902](https://github.com/PaddlePaddle/Paddle/pull/45902))
- **视觉 ViT 模型优化**
  - 新增 ViT 模型 Attention 结构融合 PASS，并支持 OSS Plugin 和自动 padding，ViT 推理速度提升 30%-40%  [#45019](https://github.com/PaddlePaddle/Paddle/pull/45019) [#45506](https://github.com/PaddlePaddle/Paddle/pull/45506)
- **大模型推理性能优化**
  - 为提高超大生成模型推理速度以及显存节省，对多层 Transformer 融合算子(fused_multi_transformer_op)增加 INT8 实现（fused_multi_transformer_int8_op），支持生成模型的量化推理。结合矩阵乘算法选择、量化反量化 kernel 融合进行性能优化。 [#46169](https://github.com/PaddlePaddle/Paddle/pull/46169)
  - 为了提升大模型推理使用 fused_multi_transformer 融合的易用性，增加 Pass 进行自动匹配融合。
- **CPU 性能优化**
  - 优化语音 U2++ 模型，FP32 模型推理速度提升 35%，INT8 模型推理速度提升 69% ([#47592](https://github.com/PaddlePaddle/Paddle/pull/47592) [#47127](https://github.com/PaddlePaddle/Paddle/pull/47127) [#47391](https://github.com/PaddlePaddle/Paddle/pull/47391) [#47234](https://github.com/PaddlePaddle/Paddle/pull/47234) [#47009](https://github.com/PaddlePaddle/Paddle/pull/47009) [#47080](https://github.com/PaddlePaddle/Paddle/pull/47080))


### （3）问题修复

- TensorRT workspace size 大小设置支持 int64。（[#44469](https://github.com/PaddlePaddle/Paddle/pull/44469)）
- Paddle-TRT 中，全面支持 Op 的输入为权重。（[#45545](https://github.com/PaddlePaddle/Paddle/pull/45545)）
- Paddle-TRT 中，支持 conv2d_transpose/conv3d_transpose 含 output_padding 属性。（[#45004](https://github.com/PaddlePaddle/Paddle/pull/45004)）
- Paddle-TRT 中，增强 strided_slice 对动态 shape 的支持。（[#46819](https://github.com/PaddlePaddle/Paddle/pull/46819)）
- Paddle-TRT 中，优化了在多线程场景下运行时 context 的显存占用。（[#45468](https://github.com/PaddlePaddle/Paddle/pull/45468)）
- Paddle-TRT 中，修复了多个模型在同一进程中运行时，当初始化顺序变动时，反复生成序列化文件的问题。（[#43942](https://github.com/PaddlePaddle/Paddle/pull/43942)）
- 修复了同一进程中，多次初始化 Predictor 并运行时，偶发崩溃的问题。（[#45203](https://github.com/PaddlePaddle/Paddle/pull/45203)）
- 修复 MobileNetV3_large、ERNIE 3.0-Medium 和 bert 等量化模型推理精度异常问题 ([#45416](https://github.com/PaddlePaddle/Paddle/pull/45416) [#46283](https://github.com/PaddlePaddle/Paddle/pull/46283) [#45920](https://github.com/PaddlePaddle/Paddle/pull/45920) [#47573](https://github.com/PaddlePaddle/Paddle/pull/47574))

## 5. 环境适配

- 训练用的预编译安装包与部署用的预编译安装包（Paddle Inference）统一为一个预编译安装包，且优化了构建系统，使得预编译的安装包默认支持 TensorRT。
- 取消了适配 CUDA10.1 版本的预编译安装包。
- 新增了适配 CUDA11.7 版本的预编译安装包。
- 源码编译时间缩短：减少模块间依赖，提升并行度，优化部分模块的编译速度，共同使的全量编译时间减少了约 20 分钟。
- 支持在 windows 11、Centos 8、Ubuntu 22.04、Jetson 5.02 系统环境上运行飞桨，支持使用 WSL 2 工具在 windows 系统中运行飞桨 linux 安装包。
- 修复飞桨在 glibc2.34+环境中运行错误的问题。
- 优化了整个代码仓库中的 C++、Python、CMake 的代码风格，并引入或升级了以下的代码风格检查工具。
  - pre-commit 由 1.10.4 升级到 2.17.0： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - pylint 由默认版本改为指定 2.12.0 版本： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - remove-crlf 由 1.0.1 升级到 1.1.14： [#43103](https://github.com/PaddlePaddle/Paddle/pull/43103)
  - cpplint 由默认版本改为指定 1.6.0 版本： [#43175](https://github.com/PaddlePaddle/Paddle/pull/43175)，[#43978](https://github.com/PaddlePaddle/Paddle/pull/43978)，[#43673](https://github.com/PaddlePaddle/Paddle/pull/43673)，[#43679](https://github.com/PaddlePaddle/Paddle/pull/43679)，[#43695](https://github.com/PaddlePaddle/Paddle/pull/43695)，[#43733](https://github.com/PaddlePaddle/Paddle/pull/43733)，[#43740](https://github.com/PaddlePaddle/Paddle/pull/43740)
  - clang-format 由 3.8 升级到 13.0： [#42840](https://github.com/PaddlePaddle/Paddle/pull/42840)，[#43248](https://github.com/PaddlePaddle/Paddle/pull/43248)，[#43329](https://github.com/PaddlePaddle/Paddle/pull/43329)，[#43333](https://github.com/PaddlePaddle/Paddle/pull/43333)，[#43633](https://github.com/PaddlePaddle/Paddle/pull/43633)，[#43678](https://github.com/PaddlePaddle/Paddle/pull/43678)
  - 引入 black 工具进行 python 代码的风格检查：[#46014](https://github.com/PaddlePaddle/Paddle/pull/46014)
  - 引入 cmakelint 工具用于 cmake 文件代码检查，版本为 1.4.2： [#43222](https://github.com/PaddlePaddle/Paddle/pull/43222)，[#43406](https://github.com/PaddlePaddle/Paddle/pull/43406)，[#43414](https://github.com/PaddlePaddle/Paddle/pull/43414)，[#43428](https://github.com/PaddlePaddle/Paddle/pull/43428)
  - 引入 cmake-format 用于 cmake 文件的自动格式化，版本为 0.6.13： [#43057](https://github.com/PaddlePaddle/Paddle/pull/43057)

## 6. 硬件适配
### 海光 DCU
- 增加在 DCU 上的 Profiler 功能，可以在 DCU 上对模型运行过程的性能数据进行收集、统计和展示，支持 kernel 层面的 DCU 占用率显示。
### 昆仑芯
- 增加在昆仑芯 2 代芯片上的 Profiler 功能，可以在昆仑芯 2 代芯片上对模型运行过程的性能数据进行收集、统计和展示，支持 kernel 层面的昆仑芯 2 代芯片占用率显示。
- 昆仑芯 2 代芯片（昆仑芯 AI 加速卡 R200、R300、R200-8F、R200-8FS、RG800）训练/推理支持，已验证 PPYOLOE、PP-OCR、ERNIE 3.0、PP-TSM、PP-TTS、DLRM、PPO 等总计 51 个模型，支持静态图+动态图训练，支持混合精度训练，支持单机单卡、单机多卡训练，覆盖了智能视觉、自然语言处理、智能语音、智能推荐、强化学习 5 个领域。
### 寒武纪
-  寒武纪 MLU 芯片（MLU370 系列板卡）训练/推理支持，已验证 ResNet50、BERT、YoloV3、OCR-DB、Deeplabv3 等多个模型，支持静态图+动态图训练，支持混合精度训练，支持单机单卡、单机多卡训练。
### Graphcore
- Graphcore IPU 芯片（包括 IPU Mk2 GC200 和 Bow IPU）训练/推理支持，支持 ResNet50、BERT 等模型，支持静态图和动转静模式训练，支持单芯片、单机、多机分布式训练。
- 增加更多算子支持
- 升级到 Poplar SDK v3.0.0 版本 [#46892](https://github.com/PaddlePaddle/Paddle/pull/46892)
* 支持使用动转静模式训练模型, 添加了一个新的 paddle.incubate.identity_loss op 用来辅助构图 [#43770](https://github.com/PaddlePaddle/Paddle/pull/43770)
* 支持 Paddle 原生的分布式训练 API paddle.distributed.launch [#43311](https://github.com/PaddlePaddle/Paddle/pull/43311)
* 支持使用混合精度训练模型 [#41733](https://github.com/PaddlePaddle/Paddle/pull/41733)
* Paddle Inference 支持使用 PopART 自定义算子 [#45235](https://github.com/PaddlePaddle/Paddle/pull/45235)

### Intel
- 迁移 oneDNN 算子 transpose2_grad([#46139](https://github.com/PaddlePaddle/Paddle/pull/46139)), relu6_grad([#46501](https://github.com/PaddlePaddle/Paddle/pull/46501)), gaussian_random([#46747](https://github.com/PaddlePaddle/Paddle/pull/46747), [#45481](https://github.com/PaddlePaddle/Paddle/pull/45481)), sgd and stack([#46374](https://github.com/PaddlePaddle/Paddle/pull/46374)), concat+grad, expand+grad,fill_constant([#45863](https://github.com/PaddlePaddle/Paddle/pull/45863)), slice, slice_grad, split,pad and pad3d([#46101](https://github.com/PaddlePaddle/Paddle/pull/46101)), softmax_grad([#46257](https://github.com/PaddlePaddle/Paddle/pull/46257)), Shape([#46051](https://github.com/PaddlePaddle/Paddle/pull/46051)), Sum([#46239](https://github.com/PaddlePaddle/Paddle/pull/46239)), Transpose2_grad([#46139](https://github.com/PaddlePaddle/Paddle/pull/46139)), Cast, clip+grad andpool+grad([#45775](https://github.com/PaddlePaddle/Paddle/pull/45775)), Reduce sum+grad,mean+grad, min and max([#45536](https://github.com/PaddlePaddle/Paddle/pull/45536)), Relu and abs([#45397](https://github.com/PaddlePaddle/Paddle/pull/45397)), Gelu([#45596](https://github.com/PaddlePaddle/Paddle/pull/45596)), Scale([#45537](https://github.com/PaddlePaddle/Paddle/pull/45537))
- 优化 fill_constant, fc, conv 等若干算子内核
- 增加若干 Pass 融合优化
- 优化 Adam-W CPU FP32 优化器 ([#42522](https://github.com/PaddlePaddle/Paddle/pull/42522))
- 优化 pad3d fp32 onednn 算子内核实现 ([#43990](https://github.com/PaddlePaddle/Paddle/pull/43990))
- 改进 matmul, FC andlookup_v2 内核的并发执行 ([#44023](https://github.com/PaddlePaddle/Paddle/pull/44023), [#44078](https://github.com/PaddlePaddle/Paddle/pull/444078), [#44640](https://github.com/PaddlePaddle/Paddle/pull/44640), [#44744](https://github.com/PaddlePaddle/Paddle/pull/44744), [#45249](https://github.com/PaddlePaddle/Paddle/pull/45249))
- FC onednn 算子内核支持 bf16 ( [#42758](https://github.com/PaddlePaddle/Paddle/pull/42758), [#43154](https://github.com/PaddlePaddle/Paddle/pull/43154), [#43109](https://github.com/PaddlePaddle/Paddle/pull/43109))
- 增加矩阵乘法和激活函数的融合([#43519](https://github.com/PaddlePaddle/Paddle/pull/43519), [#43198](https://github.com/PaddlePaddle/Paddle/pull/43198))
- 支持卷积算子 int8 参数生产 IR passes ( [#44680](https://github.com/PaddlePaddle/Paddle/pull/44680), [#42625](https://github.com/PaddlePaddle/Paddle/pull/42625))
- 增加 pool/avg 量化和 scales 修正 ([#44186](https://github.com/PaddlePaddle/Paddle/pull/44186))
- 增加 matmul 和 elementwise onednn 算子内核融合([#45077](https://github.com/PaddlePaddle/Paddle/pull/45077))
- 修复 QAT 精度问题 ([#43693](https://github.com/PaddlePaddle/Paddle/pull/43693), [#45936](https://github.com/PaddlePaddle/Paddle/pull/45936), [#46378](https://github.com/PaddlePaddle/Paddle/pull/46378))
- 迁移 42 个 oneDNN 算子内核到 PHI 算子库 ([#46374](https://github.com/PaddlePaddle/Paddle/pull/46374), [#46101](https://github.com/PaddlePaddle/Paddle/pull/46101), [#45989](https://github.com/PaddlePaddle/Paddle/pull/45989), [#45863](https://github.com/PaddlePaddle/Paddle/pull/45863), [#45775](https://github.com/PaddlePaddle/Paddle/pull/45775), [#45626](https://github.com/PaddlePaddle/Paddle/pull/45626), [#45536](https://github.com/PaddlePaddle/Paddle/pull/45536), [#46501](https://github.com/PaddlePaddle/Paddle/pull/46501), [#46257](https://github.com/PaddlePaddle/Paddle/pull/46257), [#45596](https://github.com/PaddlePaddle/Paddle/pull/45596), [#45537](https://github.com/PaddlePaddle/Paddle/pull/45537), [#45481](https://github.com/PaddlePaddle/Paddle/pull/45481), [#45397](https://github.com/PaddlePaddle/Paddle/pull/45397), [#46239](https://github.com/PaddlePaddle/Paddle/pull/46239), [#46139](https://github.com/PaddlePaddle/Paddle/pull/46139), [#46051](https://github.com/PaddlePaddle/Paddle/pull/46051))
- 量化 elementwise_sub 和 shape 算子内核 ([#42854](https://github.com/PaddlePaddle/Paddle/pull/42854), [#44124](https://github.com/PaddlePaddle/Paddle/pull/44124))

## Thanks to our Contributors

This release contains contributions from:

0x45f, Aganlengzi, Ainavo, Allen Guo, Asthestarsfalll, Aurelius84, Baibaifan, baoachun, BiynXu, Bo Zhang, BrilliantYuKaimin, cambriconhsq, caozhou, carryyu, ccrrong, ceci3, chalsliu, Chang Xu, Charles-hit, Chen Long, Chen Weihang, chenjian, chentianyu03, Chenxiao Niu, cifar10, crystal, csy0225, danleifeng, David Nicolas, dc-cheny, denglin-github, dongfangshenzhu, duanboqiang, duanyanhui, engineer, enzodechine, Fan Zhang, feifei-111, Feiyu Chan, Feng Ni, feng_shuai, FlyingQianMM, freeliuzc, furnace, fuyou765, fwenguang, Ghost Screaming, gongweibao, Guanghua Yu, guguguzi, Guoxia Wang, Haipeng Wang, handiz, Haohongxiang, haosicheng, helen88, heliqi, hong, HongyuJia, houj04, huangxu96, Hui Zhang, Huihuang Zheng, huzhiqiang, Jacek Czaja, Jack Zhou, jack603047588, Jackwaterveg, jakpiase, james, Jiabin Yang, jiangcheng, Jiaqi Liu, JingZhuangzhuang, joanna.wozna.intel, JYChen, JZ-LIANG, Kaipeng Deng, kangguangli, kuizhiqing, Leo Chen, Leo Guo, levi131, Li Min, Li-fAngyU, lidanqing, LielinJiang, Ligoml, Lijunhui, lilong12, limingshu, Lin Manhui, Linjie Chen, liqitong-a, littletomatodonkey, liu zhengxi, Liu-xiandong, liutiexing, Liyulingyue, LiYuRio, Lux et Veritas, lyq, Matsumoto Ruko, MayYouBeProsperous, mengqingchun02, Ming-Xu Huang, ming1753, minghaoBD, moyan, mrcangye, Netpunk, niuliling123, Nyakku Shigure, OccupyMars2025, onecatcn, pangyoki, parap1uie-s, peachlcy, piotrekobi, Qi Li, QingshuChen, qipengh, Rayman, Regan Yue, RichardWooSJTU, risemeup1, Roc, ronnywang, Rui Li, Ruibiao Chen, seemingwang, Shang Zhizhou, shangliang Xu, ShenLiang, shentanyue, Shijie, ShiningZhang, shixingbo, shiyutang, Shuangchi He, Siming Dai, Sing_chan, Skr Bang, SmirnovKol, sneaxiy, sprouteer, Sylwester Fraczek, Sławomir Siwek, taixiurong, Tao CHANG, TeFeng Chen, Thomas Young, thunder95, Thunderbrook, tiancaishaonvjituizi, tianshuo78520a, Tomasz Socha, TTerror, USTCKAY, Vigi Zhang, Walter, Wang Bojun, wangguanqun, wangguanzhong, wanghuancoder, wangna11BD, WangXi, wangxinxin08, Wangzheee, WangZhen, wangzhen38, wawltor, wbn, Wei Shengyu, Weilong Wu, weishengying, Wen Sun, wenbin, whs, Wilber, WJJ1995, wuhuachaocoding, wuhuanzhou, wuyefeilin, XiaoguangHu, xiaoguoguo626807, xiaohemaikoo, xiaoting, xiaoxiaohehe001, Xiaoxu Chen, xiayanming, Xingyuan Zhang, xiongkun, yang131313, yangguohao, YangZhou, Yanxing Shi, Yao Zihang, yaoxuefeng, yaozhixin, yeliang2258, Yilingyelu, Yiqun Liu, ykkk2333, Yuang Liu, Yuanle Liu, YuanRisheng, yuguo, Yulong Ao, Yulv-git, YUNSHEN XIE, Zhang Jun, Zhang Ting, Zhang Zheng, zhangbo9674, zhangbopd, zhangchunle, Zhangjingyu06, zhangkaihuo, zhangxiaoci, zhangyikun02, zhangzhenguo, Zhanlue Yang, zhaocaibei123, zhaoying9105, zhaoyingli, Zhen Wang, Zhengyang Song, zhiboniu, Zhong Hui, Zhou Wei, zhoutianzi666, zhupengyang, ziyoujiyi, zlsh80826, zmxdream, zn, Zuza Gawrysiak, zyfncg, 傅剑寒, 六个骨头, 津, 熊峻峰, 王明冬, 石晓伟


# 2.3.1 Release Note

## 1. 重要更新

- 2.3.1 版本是在 2.3 版本的基础上修复了已知问题，并且发布了支持 CUDA 11.6 的安装包。

## 2. 训练框架（含分布式）

### （1）功能优化

#### API

- 修改 `paddle.nn.initializer.KaimingUniform` 和 `paddle.nn.initializer.KaimingNormal` 两种初始化方式，使其支持多种类型的激活函数。([#43721](https://github.com/PaddlePaddle/Paddle/pull/43721), [#43827](https://github.com/PaddlePaddle/Paddle/pull/43827))
- 优化 `paddle.io.DataLoader` 的数据预读取功能，使其支持设置了 `prefetch_factor` 设定的预读取数据的缓存数量，避免在读取大块数据时出现 IO 阻塞。([#43674](https://github.com/PaddlePaddle/Paddle/pull/43674))

#### 新动态图执行机制

- 修改新动态图 API 逻辑中 optional 类型 Tensor 的初始化方法，防止被提前析构导致数据异常。([#42561](https://github.com/PaddlePaddle/Paddle/pull/42561))

#### 全新静态图执行器

- 延迟初始化执行器中的线程池，避免只执行一轮的 `program`（如 `save、load、startup_program` 等）创建线程池。([#43768](https://github.com/PaddlePaddle/Paddle/pull/43768))

#### 混合精度训练

- 设置 `paddle.nn.Layer` 中 `set_state_dict` 中禁用 `state_dict` hook。([#43407](https://github.com/PaddlePaddle/Paddle/pull/43407))

#### 分布式训练

- 使 `paddle.incubate.nn.functional.fused_attention` 和 `paddle.incubate.nn.functional.fused_feedforward` 支持张量模型并行。([#43505](https://github.com/PaddlePaddle/Paddle/pull/43505))

#### 其他

- 调整框架算子内核打印字符串的格式，便于进行自动化拆分解析。([#42931](https://github.com/PaddlePaddle/Paddle/pull/42931))
- 更新模型量化 API，支持 `rounding to nearest ties to even` 的四舍五入方式，支持量化取值范围 [-128, 127]。([#43829](https://github.com/PaddlePaddle/Paddle/pull/43829))
- 量化感知训练适配支持 AMP 混合精度训练。([#43689](https://github.com/PaddlePaddle/Paddle/pull/43689))
- 量化感知训练在启动时新增 `progress bar`，便于查看量化初始化进度，统计 out_threshold 时跳过 scale op，加速初始化过程。([#43454](https://github.com/PaddlePaddle/Paddle/pull/43454))
- 动态图量化训练支持 `conv` 和 `bn` 融合，静态图离线量化支持设置 `skip_tensor_list` 来跳过某些层不做量化。([#43301](https://github.com/PaddlePaddle/Paddle/pull/43301))

### （2）性能优化

- 优化 `paddle.incubate.nn.functional.fused_attention` 和`paddle.incubate.nn.functional.fused_feedforward` 算子，增加 `add_residual` 属性，用以控制最后一步是否进行加 `residual` 操作，CAE 模型性能提升 7.7%。([#43719](https://github.com/PaddlePaddle/Paddle/pull/43719))
- 优化 `linspace` 算子，将 `start`、`stop`、`num` 三个输入 Tensor 初始化在 CPU 上，避免在算子中进行 GPU -> CPU 拷贝，SOLOv2 模型性能提升 6%。([#43746](https://github.com/PaddlePaddle/Paddle/pull/43746))

### （3）问题修复

#### API

- 修复 `paddle.io.DataLoader` 在 `return_list=True` 时因多线程冲突小概率报错问题。([#43691](https://github.com/PaddlePaddle/Paddle/pull/43691))
- 修复 `paddle.nn.Layer` 的参数存在 `None` 类型参数时 `to` 方法报 NoneType 不存在 device 属性的错误。([#43597](https://github.com/PaddlePaddle/Paddle/pull/43597))
- 修复 cumsum op 在某些 `shape`下计算结果出错的问题。([#42500](https://github.com/PaddlePaddle/Paddle/pull/42500), [#43777](https://github.com/PaddlePaddle/Paddle/pull/43777))
- 修复静态图下 `Tensor.__getitem__`在使用 `bool`索引时组网阶段输出结果维度为 0 的问题。([#43246](https://github.com/PaddlePaddle/Paddle/pull/43246))
- 修复 `paddle.slice` 和 `paddle.strided_slice` 处理参数为负数时出现异常的问题。([#43432](https://github.com/PaddlePaddle/Paddle/pull/43432))
- 修复 set_value op 在处理切片 `step`为负数时赋值结果异常的问题。([#43694](https://github.com/PaddlePaddle/Paddle/pull/43694))
- 修复 C++ 端 `copy` 接口不能在多卡设备间拷贝的问题。([#43728](https://github.com/PaddlePaddle/Paddle/pull/43728))
- 修改 `paddle.incubate.nn.functional.fused_attention` 和 `paddle.incubate.nn.functional.fused_feedforward` 中属性命名引发的推理时的问题。([#43505](https://github.com/PaddlePaddle/Paddle/pull/43505))
- 修复 ConditionalBlockGrad op 处理不需要 `grad`的 Tensor 时异常的问题。([#43034](https://github.com/PaddlePaddle/Paddle/pull/43034))
- 解决 C++ 的 einsum op 反向速度优化引起的显存增加问题，并将反向优化默认打开。([#43397](https://github.com/PaddlePaddle/Paddle/pull/43397))
- 修复单卡下 `paddle.io.DataLoader`多进程数据读取在固定随机种子时数据无法固定的问题。([#43702](https://github.com/PaddlePaddle/Paddle/pull/43702))
- 修复 softmax op 在 Tensor 元素超过 2G 时，触发 CUDNN_STATUS_NOT_SUPPORT 的错误。([#43719](https://github.com/PaddlePaddle/Paddle/pull/43719))
- 修复 trace op `Event` 字符串在不同算子无区分，导致性能分析不便利的问题。([#42789](https://github.com/PaddlePaddle/Paddle/pull/42789))

#### 其他

- 修复动转静多次 deepcopy 并保存导致的显存溢出问题。([#43141](https://github.com/PaddlePaddle/Paddle/pull/43141))
- 修复自定义算子中使用的 PlaceType 类型升级引入的 device id 在多卡场景中出错的问题。([#43830](https://github.com/PaddlePaddle/Paddle/pull/43830))
- 优化 `paddle.profiler.Profiler` timeline 可视化逻辑，将在 python 脚本中自定义的事件从 C++ 折叠层显示移动至 python 折叠层显示。([#42790](https://github.com/PaddlePaddle/Paddle/pull/42790))

## 3. 部署方向（Paddle Inference）

### （1）新增特性

#### 新增功能

- CPU 上 ONNX Runtime 后端新增 PaddleSlim 量化模型支持。([#43774](https://github.com/PaddlePaddle/Paddle/pull/43774), [#43796](https://github.com/PaddlePaddle/Paddle/pull/43796))

### （2）底层优化

#### CPU 性能优化

- EnableMkldnn 配置中移除 `gpu_cpu_reshape2_matmul_fuse_pass`，修复 ResNet50 性能下降的问题。([#43750](https://github.com/PaddlePaddle/Paddle/pull/43750))

#### GPU 性能优化

- 添加 `bilinear_interp_v2` TensorRT convert 支持。([#43618](https://github.com/PaddlePaddle/Paddle/pull/43618))
- 添加 `matmul_scale_fuse_pass`、`multihead_matmul_fuse_pass_v3`到 GPU pass，并添加单测。([#43765](https://github.com/PaddlePaddle/Paddle/pull/43765))
- 添加 GPU handle 延迟初始化支持。([#43661](https://github.com/PaddlePaddle/Paddle/pull/43661))

### （3）问题修复

#### 框架及 API 修复

- 修复联编 Paddle-Lite XPU 时的编译报错问题。([#43178](https://github.com/PaddlePaddle/Paddle/pull/43178))
- 修复 ERNIE 3.0 pass 误触发的问题。([#43948](https://github.com/PaddlePaddle/Paddle/pull/43948))
- 修复 multihead op 中 int8 量化属性读不到的问题。([#43020](https://github.com/PaddlePaddle/Paddle/pull/43020))

#### 后端能力修复

- 修复 MKLDNN 中 elementwise_mul 和 matmul 两个 op 在运行量化推理过程中崩溃的问题。([#43725](https://github.com/PaddlePaddle/Paddle/pull/43725))
- 修复同一模型在推理时 TensorRT 子图序列化文件反复生成的问题。([#42945](https://github.com/PaddlePaddle/Paddle/pull/43945), [#42633](https://github.com/PaddlePaddle/Paddle/pull/42633))
- 修复 ONNX Runtime 后端与外部使用的 protobuf 冲突问题。([#43159](https://github.com/PaddlePaddle/Paddle/pull/43159), [#43742](https://github.com/PaddlePaddle/Paddle/pull/43742))
- 修复 python 预测库 ONNX Runtime 后端在多输入情况下推理报错问题。([#43621](https://github.com/PaddlePaddle/Paddle/pull/43621))

## 4. 环境适配

### 编译安装

- 完成对 CUDA 11.6 的验证和适配，并在官网发布 CUDA 11.6 的安装包。([#43935](https://github.com/PaddlePaddle/Paddle/pull/43935), [#44005](https://github.com/PaddlePaddle/Paddle/pull/44005))
- 修复在 Windows 上使用 CUDA 11.6 编译时的 cub 报错问题。([#43935](https://github.com/PaddlePaddle/Paddle/pull/43935), [#44005](https://github.com/PaddlePaddle/Paddle/pull/44005))
- 修复 elementwise、reduce op 编译时间较长的问题。([#43202](https://github.com/PaddlePaddle/Paddle/pull/43202), [#42779](https://github.com/PaddlePaddle/Paddle/pull/42779), [#43205](https://github.com/PaddlePaddle/Paddle/pull/43205))

### 新硬件适配

- 寒武纪 MLU 支持飞桨 Profiler。([#42115](https://github.com/PaddlePaddle/Paddle/pull/42115))
- GraphCore IPU 支持显示编译进度。([#42078](https://github.com/PaddlePaddle/Paddle/pull/42078))

# 2.3.0 Release Note

## 1. 重要更新

我们很高兴地发布飞桨框架 2.3.0 版本，本版本包含如下重要更新。

### API

- 新增 100 多个 API，覆盖自动微分、线性代数、概率分布、稀疏张量、框架性能分析、硬件设备管理、视觉领域等方面。

- 新增 4 个自动微分 API，11 个线性代数 API，21 个概率分布类 API，更好地支持科学计算、强化学习等场景。

- 新增 11 个 稀疏张量计算 API，支持创建 COO、CSR 格式的 Sparse Tensor 以及与 Tensor 互相转换等基础功能。

- 新增 9 个框架性能分析 API，以 `paddle.profiler.Profiler` 为核心，提供对训练、推理过程中性能数据的收集、导出和统计的功能。

- 新增 7 个硬件设备管理 API，更好支持硬件相关信息获取。

- 新增多个视觉、文本领域 API，方便复用 MobileNetV3, ResNeXt 等骨干网络，实现快速组网。

### 飞桨高可复用算子库 PHI

- 发布飞桨高可复用算子库 PHI (Paddle HIgh reusability operator library)，支持组合式算子功能复用、Primitive 算子内核复用、插件式硬件加速库复用。针对飞桨框架原算子库存在的算子接口不清晰、算子复用成本较高、调用性能不够快的问题，我们重构了飞桨框架的算子库，设计了灵活、高效的函数式算子库 Phi，可以通过对函数式算子接口组合调用的方式实现新算子。新算子库提供了 200 余个跟 python 开发接口保持一致的 C++ 运算类 API，以及近 500 个可供组合调用的前、反向函数式算子内核 Kernel，可大幅降低框架原生算子和自定义算子的开发成本。新算子库支持 Primitive API 方式开发算子内核，可支持不同硬件（比如 GPU 和 XPU）的算子内核复用。新算子库支持以插件方式接入硬件（比如 NPU）的加速库，实现低成本复用硬件加速库。

### 分布式训练

- 全面升级自适应分布式训练架构，含弹性扩缩容、异步流水执行器、异构通信、自动并行等多个模块，支持了多种异构硬件下自动感知的分布式训练及分布式推理。

- 动态图混合并行下新增 MoE 并行策略、GroupSharded 并行策略、Pure FP16 等，进一步支持了动态图下大模型的高效并行训练。

- 全面升级优化了通用异构参数服务器架构，进行各模块的抽象简化，如通信、存储等，提升了参数服务器的二次开发体验；GPU 参数服务器在千亿参数百亿数据分钟级流式训练下性能提升 2.38 倍。

### 编译安装

- 从 2.3.0 版本开始，飞桨对框架支持的 GPU 架构种类进行了调整和升级。

### 推理部署

- 新增 Java API 和 ONNX Runtime CPU 后端。

- 支持 TensorRT 8.0 / 8.2 和结构化稀疏，针对 ERNIE 类结构模型性能深度优化。

### 硬件适配

- 新增自定义新硬件接入：提供一种插件式扩展 PaddlePaddle 硬件后端的方式。

- 新增对华为昇腾 910 / GraphCore IPU / 寒武纪 MLU / 昆仑芯 2 代多种异构芯片的训练/推理支持。

### 框架架构

- 这个版本中，我们在框架的执行器也做了大量工作，详情请见：[新动态图执行机制](#%E6%96%B0%E5%8A%A8%E6%80%81%E5%9B%BE%E6%89%A7%E8%A1%8C%E6%9C%BA%E5%88%B6) 与 [全新静态图执行器](#%E5%85%A8%E6%96%B0%E9%9D%99%E6%80%81%E5%9B%BE%E6%89%A7%E8%A1%8C%E5%99%A8)。

## 2. 不兼容升级

- 预编译安装包中移除 CUDA sm35 ARCH： 受到包体积大小的影响，在预编译的安装包中移除了 CUDA sm35 架构。([#41754](https://github.com/PaddlePaddle/Paddle/pull/41754))

- `paddle.to_tensor` 将一个 python int scalar 转换为 Tensor 时，在 Windows 上的默认数据类型由 int32 变为 int64，从而与 Linux/Mac 保持对齐。([#39662](https://github.com/PaddlePaddle/Paddle/pull/39662))

- 为了与 python3 下的除法行为保持一致，除法符号 `/` 从 rounding divide 变成 true divide，计算输出结果的数据类型从 int 切换成 float。([#40890](https://github.com/PaddlePaddle/Paddle/pull/40890))

<table>
<tr>
<th>
2.2
</th>
<th>
2.3.0
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> a = paddle.to_tensor([327])
>>> b = paddle.to_tensor([80])
>>> a / b
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
      [4])
```
</pre>
</td>
<td>
<pre>

```python
>>> import paddle
>>> a = paddle.to_tensor([327])
>>> b = paddle.to_tensor([80])
>>> a / b
Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
      [4.08750010])
```
</pre>
</td>
</tr>
</table>

- 修正 ELU 的公式，alpha < 0 时的计算方式与原论文对齐，从而修复小部分情况下的计算结果错误。同时，由于在 alpha < 0 无法在数学上仅从输出计算反向梯度，因此 elu_ 在 alpha < 0 时将报错。([#37316](https://github.com/PaddlePaddle/Paddle/pull/37316))

<table>
<tr>
<th>
2.2
</th>
<th>
2.3.0
</th>
</tr>

<tr>
<td>
<pre>

```python
# elu(x) = max(0, x) + min(0, α ∗ (e^x − 1))
>>> import paddle
>>> x = paddle.to_tensor([-1., 6.])
>>> m = paddle.nn.ELU(-0.2)
>>> out = m(x)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [ 0.         , -74.48576355])
>>> out = paddle.nn.functional.elu_(x, alpha=-0.2, name=None)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [ 0.         , -74.48576355])
```
</pre>
</td>
<td>
<pre>

```python
# elu(x) = x, if x > 0
# elu(x) = α ∗ (e^x − 1), if x <= 0
>>> import paddle
>>> x = paddle.to_tensor([-1., 6.])
>>> m = paddle.nn.ELU(-0.2)
>>> out = m(x)
>>> out
Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [0.12642412,  6.        ])
>>> out = paddle.nn.functional.elu_(x, alpha=-0.2, name=None)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.7/dist-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/usr/local/lib/python3.7/dist-packages/paddle/fluid/wrapped_decorator.py", line 25, in __impl__
    return wrapped_func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/paddle/fluid/dygraph/inplace_utils.py", line 34, in __impl__
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/paddle/nn/functional/activation.py", line 89, in elu_
    assert alpha >= 0., "elu_ only support alpha >= 0, please use elu instead."
AssertionError: elu_ only support alpha >= 0, please use elu instead.
```
</pre>
</td>
</tr>
</table>

## 3. 训练框架（含分布式）

### （1）新功能

#### API

- 新增 4 个自动微分类 API，支持科学计算需求，具体列表如下：([#40692](https://github.com/PaddlePaddle/Paddle/pull/40692))

  - `paddle.incubate.autograd.vjp`，计算向量-雅可比矩阵乘积。

  - `paddle.incubate.autograd.jvp`，计算雅可比矩阵-向量乘积。

  - `paddle.incubate.autograd.Jacobian`，计算雅可比矩阵。

  - `paddle.incubate.autograd.Hessian`，计算海森矩阵。

- 新增线性代数类 API

  - 新增 `paddle.linalg.triangular_solve`，计算具有唯一解的三角系数线性方程组。([#36714](https://github.com/PaddlePaddle/Paddle/pull/36714))

  - 新增 `paddle.linalg.eig`，计算一般方阵的特征分解。([#35764](https://github.com/PaddlePaddle/Paddle/pull/35764))

  - 新增 `paddle.linalg.sovle`，计算线性方程组的解。([#35715](https://github.com/PaddlePaddle/Paddle/pull/35715))

  - 新增 `paddle.linalg.lstsq`，计算线性方程组的最小二乘解。([#38585](https://github.com/PaddlePaddle/Paddle/pull/38585), [#38621](https://github.com/PaddlePaddle/Paddle/pull/38621))

  - 新增 `paddle.linalg.qr`，计算矩阵的 QR 分解。([#35742](https://github.com/PaddlePaddle/Paddle/pull/35742), [#38824](https://github.com/PaddlePaddle/Paddle/pull/38824))

  - 新增 `paddle.inner`，计算矩阵内积。([#37706](https://github.com/PaddlePaddle/Paddle/pull/37706))

  - 新增 `paddle.outer`，计算矩阵外积。([#37706](https://github.com/PaddlePaddle/Paddle/pull/37706))

  - 新增 `paddle.linalg.cov`，计算向量间协方差。([#38392](https://github.com/PaddlePaddle/Paddle/pull/38392))

  - 新增 `paddle.linalg.cholesky_sovle`，计算方程 cholesky 解。([#38167](https://github.com/PaddlePaddle/Paddle/pull/38167))

  - 新增 `paddle.linalg.lu`、 `paddle.linalg.lu_unpack`，计算矩阵 lu 分解、解压缩 lu 矩阵。([#38617](https://github.com/PaddlePaddle/Paddle/pull/38617), [#38559](https://github.com/PaddlePaddle/Paddle/pull/38559), [#38616](https://github.com/PaddlePaddle/Paddle/pull/38616))

- 新增 21 个概率分布类 API，包括 6 个随机变量分布，13 个随机变量变换，2 个 KL 散度计算，用于强化学习、变分推断、科学计算等场景，具体列表如下：([#40536](https://github.com/PaddlePaddle/Paddle/pull/40536), [#38820](https://github.com/PaddlePaddle/Paddle/pull/38820), [#38558](https://github.com/PaddlePaddle/Paddle/pull/38558/files), [#38445](https://github.com/PaddlePaddle/Paddle/pull/38445), [#38244](https://github.com/PaddlePaddle/Paddle/pull/38244), [#38047](https://github.com/PaddlePaddle/Paddle/pull/38047))

  - `paddle.distribution.ExponentialFamily`，指数分布族基类。

  - `paddle.distribution.Beta`，`Beta` 分布。

  - `paddle.distribution.Dirichlet`，`Dirichlet` 分布。

  - `paddle.distribution.Independent`，独立分布，用于创建高阶分布。

  - `paddle.distribution.TransformedDistribution`，变换分布，用于通过基础分布及一系列变换生成高阶分布。

  - `paddle.distribution.Multionmial`，多项分布。

  - `paddle.distribution.Transform`，随机变量变换的基类。

  - `paddle.distribution.AbsTransform`，取绝对值变换。

  - `paddle.distribution.AffineTransform`，仿射变换。

  - `paddle.distribution.ChainTransform`，变换的链式组合。

  - `paddle.distribution.ExpTransform`，指数变换。

  - `paddle.distribution.IndependentTransform`，独立变换，用于扩展变换定义域的 `event_dim`。

  - `paddle.distribution.PowerTransform`，幂变换。

  - `paddle.distribution.ReshapeTransform`，`reshape` 变换。

  - `paddle.distribution.SigmoidTransform`，`sigmoid` 变换。

  - `paddle.distribution.SoftmaxTransform`，`softmax` 变换。

  - `paddle.distribution.StackTransform`，`stack` 变换，用于以 `stack` 方式组合多个变换。

  - `paddle.distribution.StickBreakingTransform`, `stickbreaking` 变换。

  - `paddle.distribution.TanhTransform`，`tanh` 变换。

  - `paddle.distribution.kl_divergence`，计算 KL 散度。

  - `paddle.distribution.register_kl`，注册用户自定义 KL 散度计算函数。

- 新增高层 API

  - 新增 `paddle.vision.models.AlexNet`、`paddle.vision.models.alexnet`，支持直接使用 AlexNet 模型。([#36058](https://github.com/PaddlePaddle/Paddle/pull/36058))

  - 新增 `paddle.vision.models.DenseNet`、 `paddle.vision.models.densenet121`、 `paddle.vision.models.densenet161`、 `paddle.vision.models.densenet169`、 `paddle.vision.models.densenet201`、 `paddle.vision.models.densenet264`，支持直接使用 DenseNet 模型。([#36069](https://github.com/PaddlePaddle/Paddle/pull/36069))

  - 新增 `paddle.vision.models.GoogLeNet`、`paddle.vision.models.googlenet`，支持直接使用 GoogLeNet 模型。([#36034](https://github.com/PaddlePaddle/Paddle/pull/36034))

  - 新增 `paddle.vision.models.InceptionV3`、`paddle.vision.models.inception_v3`，支持直接使用 InceptionV3 模型。([#36064](https://github.com/PaddlePaddle/Paddle/pull/36064))

  - 新增 `paddle.vision.models.MobileNetV3Small`、 `paddle.vision.models.MobileNetV3Large`、`paddle.vision.models.mobilenet_v3_small`、`paddle.vision.models.mobilenet_v3_large`，支持直接使用 MobileNetV3 模型。([#38653](https://github.com/PaddlePaddle/Paddle/pull/38653))

  - 新增 `paddle.vision.models.resnext50_32x4d`、 `paddle.vision.models.resnext50_64x4d`、`paddle.vision.models.resnext101_32x4d`、`paddle.vision.models.resnext101_64x4d`、`paddle.vision.models.resnext152_32x4d`、`paddle.vision.models.resnext152_64x4d`，支持直接使用 ResNeXt 模型。([#36070](https://github.com/PaddlePaddle/Paddle/pull/36070))

  - 新增 `paddle.vision.models.ShuffleNetV2`、 `paddle.vision.models.shufflenet_v2_x0_25`、`paddle.vision.models.shufflenet_v2_x0_33`、`paddle.vision.models.shufflenet_v2_x0_5`、`paddle.vision.models.shufflenet_v2_x1_0`、`paddle.vision.models.shufflenet_v2_x1_5`、`paddle.vision.models.shufflenet_v2_x2_0`、`paddle.vision.models.shufflenet_v2_swish`，支持直接使用 ShuffleNetV2 模型。([#36067](https://github.com/PaddlePaddle/Paddle/pull/36067))

  - 新增 `paddle.vision.models.SqueezeNet`、 `paddle.vision.models.squeezenet1_0`、`paddle.vision.models.squeezenet1_1`，支持直接使用 SqueezeNet 模型。([#36066](https://github.com/PaddlePaddle/Paddle/pull/36066))

  - 新增 `paddle.vision.models.wide_resnet50_2`、`paddle.vision.models.wide_resnet101_2`，支持直接使用 WideResNet 模型。([#36952](https://github.com/PaddlePaddle/Paddle/pull/36952))

  - 新增`paddle.vision.ops.nms` API，支持单类别和多类别非极大抑制(non-maximum supression, nms)算法，用于目标检测预测任务加速。([#40962](https://github.com/PaddlePaddle/Paddle/pull/40962))

  - 新增`paddle.vision.ops.roi_pool` 和 `paddle.vision.ops.RoIPool`，支持检测任务中 RoI 区域池化操作。([#36154](https://github.com/PaddlePaddle/Paddle/pull/36154))

  - 新增`paddle.vision.ops.roi_align` 和 `paddle.vision.ops.RoIAlign`，支持检测任务中 RoI Align 操作。([#35102](https://github.com/PaddlePaddle/Paddle/pull/36154))

  - 新增 `paddle.text.ViterbiDecoder`、`paddle.text.viterbi_decode` Viterbi 解码 API，主要用于序列标注模型的预测。([#35778](https://github.com/PaddlePaddle/Paddle/pull/35778))

- 新增 11 个 Sparse 类 API，支持创建 COO、CSR 格式的 Sparse Tensor，与 Tensor 互相转换等基础功能：

  - `paddle.sparse.sparse_coo_tensor`，创建 COO 格式的 Sparse Tensor。([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `paddle.sparse.sparse_csr_tensor`，创建 CSR 格式的 Sparse Tensor。([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `paddle.sparse.ReLU`，支持 SparseCooTensor 的 ReLU 激活层。([#40959](https://github.com/PaddlePaddle/Paddle/pull/40959))

  - `paddle.sparse.functional.relu`，支持 SparseCooTensor 的 ReLU 函数。([#40959](https://github.com/PaddlePaddle/Paddle/pull/40959))

  - `Tensor.values()`，获取 SparseCooTensor 或者 SparseCsrTensor 的非零元素方法。([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.indices()`，获取 SparseCooTensor 的坐标信息的方法。([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.crows()`，获取 SparseCsrTensor 的压缩行信息的方法。([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.cols()`，获取 SparseCsrTensor 的列信息的方法。([#40608](https://github.com/PaddlePaddle/Paddle/pull/40608))

  - `Tensor.to_sparse_coo()`，将 DenseTensor 或者 SparseCsrTensor 转换为 SparseCooTensor。([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `Tensor.to_sparse_csr()`，将 DenseTensor 或者 SparseCooTensor 转换为 SparseCsrTensor。([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

  - `Tensor.to_dense()`，将 SparseCooTensor 或者 SparseCsrTensor 转换为 DenseTensor。([#40780](https://github.com/PaddlePaddle/Paddle/pull/40780))

- 新增硬件相关 API

  - 新增 `paddle.device.cuda.max_memory_allocated`、`paddle.device.cuda.max_memory_reserved`、 `paddle.device.cuda.memory_allocated` 和 `paddle.device.cuda.memory_reserved` 四个 GPU 显存监测相关 API，方便实时查看和分析模型显存占用指标。([#38657](https://github.com/PaddlePaddle/Paddle/pull/38657))

  - 新增 `paddle.device.cuda.get_device_properties`，支持返回 CUDA 设备属性信息。([#35661](https://github.com/PaddlePaddle/Paddle/pull/35661))

  - 新增 `paddle.device.cuda.get_device_name` 和 `paddle.device.cuda.get_device_capability`，支持返回 GPU 设备名称信息和计算能力的主要和次要修订号。([#35672](https://github.com/PaddlePaddle/Paddle/pull/35672))

- 新增 Tensor 操作 API

  - 新增 `paddle.nansum`，沿 `axis` 对输入 Tensor 求和，且忽略掉 `NaNs` 值。([#38137](https://github.com/PaddlePaddle/Paddle/pull/38137))

  - 新增 `paddle.nanmean`，沿 `axis`对输入 Tensor 求平均，且忽略掉 `NaNs` 值。([#40472](https://github.com/PaddlePaddle/Paddle/pull/40472))

  - 新增 `paddle.clone`，返回输入 Tensor 的拷贝，并且提供梯度计算。([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - 新增 `paddle.Tensor.element_size`，返回 Tensor 中的单个元素在计算机中所分配的 bytes 数量。([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - 新增 `paddle.Tensor.to_uva_tensor`，支持将 numpy 对象转换为实际存储在 CPU，但可作为 CUDA 对象进行虚拟地址访问的功能。([#39146](https://github.com/PaddlePaddle/Paddle/pull/39146), [#38950](https://github.com/PaddlePaddle/Paddle/pull/38950))

  - 新增`paddle.rot90`，沿 `axes` 指定的平面将 n 维 Tensor 旋转 90 度。([#37634](https://github.com/PaddlePaddle/Paddle/pull/37634))

  - 新增`paddle.logit` 和 `paddle.Tensor.logit`，计算输入 Tensor 的 logit 函数值。([#37844](https://github.com/PaddlePaddle/Paddle/pull/37844))

  - 新增 `paddle.repeat_interleave`，沿着指定轴对输入进行复制，创建并返回到一个新的 Tensor。([#37981](https://github.com/PaddlePaddle/Paddle/pull/37981))

  - 新增 `paddle.renorm`，把 Tensor 在指定的 `axis` 切分成多块后分别进行 p norm 操作。([#38130](https://github.com/PaddlePaddle/Paddle/pull/38130), [#38459](https://github.com/PaddlePaddle/Paddle/pull/38459))

  - 新增 `paddle.mode` 和 `paddle.Tensor.mode`，沿指定轴查找输入 Tensor 的众数及对应的索引。([#38446](https://github.com/PaddlePaddle/Paddle/pull/38446))

  - 新增 `paddle.quantile` 和 `paddle.Tensor.quantile`，沿指定轴计算 Tensor 的 q 分位数。([#38567](https://github.com/PaddlePaddle/Paddle/pull/38567))

  - 新增 `paddle.kthvalue` 和 `paddle.Tensor.kthvalue`，查找 Tensor 中指定轴上第 k 小的数及对应的索引。([#38386](https://github.com/PaddlePaddle/Paddle/pull/38386))

  - 新增 `paddle.is_floating_point` 和 `paddle.Tensor.is_floating_point`，判断输入 Tensor 是否为浮点类型。([#37885](https://github.com/PaddlePaddle/Paddle/pull/37885))

  - 新增 `paddle.erfinv` 和 `paddle.Tensor.erfinv`，计算输入 Tensor 的逆误差函数。([#38295](https://github.com/PaddlePaddle/Paddle/pull/38295))

  - 新增 `paddle.lerp` 和 `paddle.Tensor.lerp`，根据给定权重计算输入 Tensor 间的线性插值。([#37253](https://github.com/PaddlePaddle/Paddle/pull/37253))

  - 新增 `paddle.angle`，用于计算复数 Tensor 的相位角。([#37689](https://github.com/PaddlePaddle/Paddle/pull/37689))

  - 新增`paddle.rad2deg`和`paddle.Tensor.rad2deg`，将元素从弧度的角度转换为度。([#37598](https://github.com/PaddlePaddle/Paddle/pull/37598))

  - 新增`paddle.deg2rad`和`paddle.Tensor.deg2rad`，将元素从度的角度转换为弧度。([#37598](https://github.com/PaddlePaddle/Paddle/pull/37598))

  - 新增`paddle.gcd`和`paddle.Tensor.gcd`，计算两个输入的按元素绝对值的最大公约数。([#37819](https://github.com/PaddlePaddle/Paddle/pull/37819))

  - 新增`paddle.lcm`和`paddle.Tensor.lcm`，计算两个输入的按元素绝对值的最小公倍数。([#37819](https://github.com/PaddlePaddle/Paddle/pull/37819))

  - 新增`paddle.amax`和`paddle.Tensor.amax`，对指定维度上的 Tensor 元素求最大值，正向结果和 max 一样，有多个相等的最大值时，反向的梯度平均分到这多个值的位置上。([#38417](https://github.com/PaddlePaddle/Paddle/pull/38417))

  - 新增`paddle.amin`和`paddle.Tensor.amin`，对指定维度上的 Tensor 元素求最小值，正向结果和 min 一样，有多个相等的最小值时，反向的梯度平均分到这多个值的位置上。([#38417](https://github.com/PaddlePaddle/Paddle/pull/38417))

  - 新增`paddle.isclose`，用于判断两个 Tensor 的每个元素是否接近。([#37135](https://github.com/PaddlePaddle/Paddle/pull/37135))

  - 新增`paddle.put_along_axis` 和`paddle.take_along_axis`，用于提取或放置指定索引下标的元素。([#38608](https://github.com/PaddlePaddle/Paddle/pull/38608))

  - 新增 `paddle.bincount` 和 `paddle.Tensor.bincount`，用于统计 Tensor 中每个元素出现的次数。([#36317](https://github.com/PaddlePaddle/Paddle/pull/36317))

  - 新增 `paddle.fmax`、 `paddle.fmin`，扩展了 max/min 的功能，支持比较的两个 Tensor 中有 NaN 值的情况，即如果对应位置上有 1 个 NaN 值，则返回那个非 NaN 值；如果对应位置上有 2 个 NaN 值，则返回 NaN 值。([#37826](https://github.com/PaddlePaddle/Paddle/pull/37826))

  - 新增 `paddle.diff`，用于计算沿给定维度的第 n 个前向差值，目前支持 n=1。([#37441](https://github.com/PaddlePaddle/Paddle/pull/37441))

  - 新增 `paddle.asinh`、`paddle.acosh`、`paddle.atanh` 反双曲函数类 API。([#37076](https://github.com/PaddlePaddle/Paddle/pull/37076))

  - 新增 `paddle.as_real`，`paddle.as_complex` 用于实数 Tensor 和复数 Tensor 之间的转换。([#37784](https://github.com/PaddlePaddle/Paddle/pull/37784))

  - 新增 `paddle.complex` 用于给定实部和虚部构造复数 Tensor。([#37918](https://github.com/PaddlePaddle/Paddle/pull/37918), [#38272](https://github.com/PaddlePaddle/Paddle/pull/38272))

  - 新增 `paddle.det` 与 `paddle.slogdet`，用于计算矩阵的行列式和行列式的自然对数。([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992))

  - 新增`paddle.nn.utils.parameters_to_vector`，可以将输入的多个 parameter 展平并连接为 1 个 1-D Tensor。([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

  - 新增`paddle.nn.utils.vector_to_parameters`，将 1 个 1-D Tensor 按顺序切分给输入的多个 parameter。([#38020](https://github.com/PaddlePaddle/Paddle/pull/38020))

- 新增组网类 API

  - 新增 `paddle.nn.Fold`、`paddle.nn.functional.fold`，支持将提取出的滑动局部区域块还原成 batch 的 Tensor。([#38613](https://github.com/PaddlePaddle/Paddle/pull/38613))

  - 新增 `paddle.nn.CELU`、`paddle.nn.functional.celu`，支持 CELU 激活层。([#36088](https://github.com/PaddlePaddle/Paddle/pull/36088))

  - 新增 `paddle.nn.HingeEmbeddingLoss`，增加计算 hinge embedding 损失的方式，通常用于学习 nonlinear embedding 或半监督学习。([#37540](https://github.com/PaddlePaddle/Paddle/pull/37540))

  - 新增 `paddle.nn.ZeroPad2D` API，按照 padding 属性对输入进行零填充。([#37151](https://github.com/PaddlePaddle/Paddle/pull/37151))

  - 新增 `paddle.nn.MaxUnPool3D` 和 `paddle.nn.MaxUnPool1D`，用于计算 3D 最大反池化和 1D 最大反池化。([#38716](https://github.com/PaddlePaddle/Paddle/pull/38716))

  - 新增 `paddle.incubate.graph_khop_sampler`、`paddle.incubate.graph_sample_neighbors`、 `paddle.incubate.graph_reindex` API，支持图多阶邻居采样和图编号重索引操作，主要用于图神经网络模型训练。([#39146](https://github.com/PaddlePaddle/Paddle/pull/39146), [#40809](https://github.com/PaddlePaddle/Paddle/pull/40809))

- 新增随机数类 API

  - 新增 `paddle.poisson`，以输入 Tensor 为泊松分布的 lambda 参数，生成一个泊松分布的随机数 Tensor。([#38117](https://github.com/PaddlePaddle/Paddle/pull/38117))

  - 新增 `paddle.randint_like` API，支持新建服从均匀分布的、范围在[low, high) 的随机 Tensor，输出的形状与输入的形状一致。([#36169](https://github.com/PaddlePaddle/Paddle/pull/36169))

  - 新增 `paddle.Tensor.exponential_`，为 inplace 式 API，通过指数分布随机数来填充输入 Tensor。([#38256](https://github.com/PaddlePaddle/Paddle/pull/38256))

- 新增参数初始化类 API

  - 新增`paddle.nn.initializer.Dirac`，通过迪拉克 delta 函数来初始化 3D/4D/5D 参数，其常用于卷积层 Conv1D/Conv2D/Conv3D 的参数初始化。([#37389](https://github.com/PaddlePaddle/Paddle/pull/37389))

  - 新增`paddle.nn.initializer.Orthogonal`，正交矩阵初始化，被初始化后的参数是（半）正交向量。([#37163](https://github.com/PaddlePaddle/Paddle/pull/37163))

  - 新增`paddle.nn.initializer.calculate_gain`，获取激活函数的推荐增益值，增益值可用于设置某些初始化 API，以调整初始化范围。([#37163](https://github.com/PaddlePaddle/Paddle/pull/37163))

- 新增学习率类 API

  - 新增 `paddle.optimizer.lr.MultiplicativeDecay`，提供 `lambda` 函数设置学习率的策略。([#38250](https://github.com/PaddlePaddle/Paddle/pull/38250))

- 新增分布式相关 API

  - 新增 `paddle.incubate.optimizer.DistributedFusedLamb`，使得 Lamb 优化器可分布式更新参数。([#40011](https://github.com/PaddlePaddle/Paddle/pull/40011), [#39972](https://github.com/PaddlePaddle/Paddle/pull/39972), [#39900](https://github.com/PaddlePaddle/Paddle/pull/39900), [#39747](https://github.com/PaddlePaddle/Paddle/pull/39747), [#39148](https://github.com/PaddlePaddle/Paddle/pull/39148), [#39416](https://github.com/PaddlePaddle/Paddle/pull/39416))

- 新增优化器相关 API([#40710](https://github.com/PaddlePaddle/Paddle/pull/40710))

  - `paddle.incubate.optimizer.functional.minimize_bfgs`，增加二阶优化器 BFGS。

  - `paddle.incubate.optimizer.functional.minimize_lbfgs`，增加二阶优化器 L-BFGS。

- 新增 `paddle.incubate.multiprocessing`模块，支持 Tensor（CPU/GPU）在 python 进程间传输。([#37302](https://github.com/PaddlePaddle/Paddle/pull/37302), [#41339](https://github.com/PaddlePaddle/Paddle/pull/41339))

- 新增 `paddle.incubate.autotune.set_config` API，支持多版本 Kernel 自动选择、混合精度数据布局自动转换、DataLoader 的 num_workers 自动选择，以自动提升模型性能。([#42301](https://github.com/PaddlePaddle/Paddle/pull/42301))

- 新增 `paddle.incubate.nn.FusedMultiTransformer` 和 `paddle.incubate.nn.functional.fused_multi_transformer` API，可将多层 transformer 融合到一个 op 中，提升模型推理性能，注意：仅支持前向推理。([#42311](https://github.com/PaddlePaddle/Paddle/pull/42311))

- 新增动静统一的 einsum_v2 op，兼容原有 python 端 `paddle.einsum` 实现的同时支持动转静导出和更加完备的 Infershape 推导。([#42495](https://github.com/PaddlePaddle/Paddle/pull/42495), [#42327](https://github.com/PaddlePaddle/Paddle/pull/42327), [#42397](https://github.com/PaddlePaddle/Paddle/pull/42397), [#42105](https://github.com/PaddlePaddle/Paddle/pull/42105))

#### IR(Intermediate Representation)

- 动态图转静态图

  - 变量类型 StaticAnalysis 模块新增支持类似 `a, b = paddle.shape(x)` 的类型标记。([#39245](https://github.com/PaddlePaddle/Paddle/pull/39245))

  - 新增支持 `InputSpec.name` 作为 Program 缓存 hash key 的计算字段。([#38273](https://github.com/PaddlePaddle/Paddle/pull/38273))

  - 新增支持 `dict['key'] = x.shape` 语法。([#40611](https://github.com/PaddlePaddle/Paddle/pull/40611))

  - 新增支持 Pure FP16 训练。([#36944](https://github.com/PaddlePaddle/Paddle/pull/36944))

  - 新增支持 `for i in [x,y,z]` 语法。([#37259](https://github.com/PaddlePaddle/Paddle/pull/37259))

  - 新增支持 python3 的 type hint 语法。([#36544](https://github.com/PaddlePaddle/Paddle/pull/36544))

- Pass 开发

  - 新增基于 NVIDIA cuBlasLt Epilogue 的 FC + [relu|gelu] 的前向与反向融合。([#39437](https://github.com/PaddlePaddle/Paddle/pull/39437))

- Kernel Primitive API

  - 新增 GPU 平台 KP 算子，包括 cast、scale、clip、bce_loss、abs_grad、reduce_sum_grad、reduce_mean_grad、clip、bce_loss、full、full_like、distribution、 random、masked_select_kernel、where_index、masked_select_grad、dropout、sigmoid、where、abs_grad。([#36203](https://github.com/PaddlePaddle/Paddle/pull/36203), [#36423](https://github.com/PaddlePaddle/Paddle/pull/36423), [#39390](https://github.com/PaddlePaddle/Paddle/pull/39390), [#39734](https://github.com/PaddlePaddle/Paddle/pull/39734), [#38500](https://github.com/PaddlePaddle/Paddle/pull/38500), [#38959](https://github.com/PaddlePaddle/Paddle/pull/38959), [#39197](https://github.com/PaddlePaddle/Paddle/pull/39197/), [#39563](https://github.com/PaddlePaddle/Paddle/pull/39563), [#39666](https://github.com/PaddlePaddle/Paddle/pull/39666), [#40517](https://github.com/PaddlePaddle/Paddle/pull/40517), [#40617](https://github.com/PaddlePaddle/Paddle/pull/40617), [#40766](https://github.com/PaddlePaddle/Paddle/pull/40766), [#39898](https://github.com/PaddlePaddle/Paddle/pull/39898), [#39609](https://github.com/PaddlePaddle/Paddle/pull/39609))

  - 新增支持 XPU2 源码编译模式。([#37254](https://github.com/PaddlePaddle/Paddle/pull/37254), [#40397](https://github.com/PaddlePaddle/Paddle/pull/40397), [#38455](https://github.com/PaddlePaddle/Paddle/pull/38455))

  - 新增支持 KP 算子在 XPU2 和 GPU 中复用，包括 reduce、broadcast、elementwise_add、`exp、log、relu、sigmoid、leaky_relu、softplus、hard_swish、reciprocal`。([#36904](https://github.com/PaddlePaddle/Paddle/pull/36904), [#37226](https://github.com/PaddlePaddle/Paddle/pull/37226), [#38918](https://github.com/PaddlePaddle/Paddle/pull/38918), [#40560](https://github.com/PaddlePaddle/Paddle/pull/40560/), [#39787](https://github.com/PaddlePaddle/Paddle/pull/39787), [#39917](https://github.com/PaddlePaddle/Paddle/pull/39917), [#40002](https://github.com/PaddlePaddle/Paddle/pull/40002), [#40364](https://github.com/PaddlePaddle/Paddle/pull/40364))

  - 新增 XPU2 平台 KP 算子单测，包括 `brelu、ceil、celu、elu、floor、hard_shrink、hard_sigmoid、log1p、logsigmoid、relu6、silu、soft_relu、softsign、sqrt、square、swish、thresholded_relu、softshrink`。([#40448](https://github.com/PaddlePaddle/Paddle/pull/40448), [#40524](https://github.com/PaddlePaddle/Paddle/pull/40524))

  - 新增 XPU2 KP 模型支持，包括 resnet50、deepfm、wide_deep、yolov3-darknet53、det_mv3_db、bert、transformer、mobilenet_v3、GPT2。

#### 混合精度训练

- 从混合精度训练 `paddle.amp.GradScaler` 的 `minimize` 中拆分出 `paddle.amp.Gradscaler.unscale_` 方法，提供恢复 loss 的独立接口。([#35825](https://github.com/PaddlePaddle/Paddle/pull/35825))

- 为 `paddle.nn.ClipByGlobalNorm` 动态图模式添加 FP16 支持，为 clip op 添加 FP16 Kernel，使`clip`相关操作支持 FP16。([#36198](https://github.com/PaddlePaddle/Paddle/pull/36198), [#36577](https://github.com/PaddlePaddle/Paddle/pull/36577))

- 支持 `paddle.amp.decorate` 传入的`optimizer`参数为 None。([#37541](https://github.com/PaddlePaddle/Paddle/pull/37541))

- 为 merged_momentum op 添加支持输入多学习率、支持 use_nesterov 策略的计算、支持 regularization 计算。([#37527](https://github.com/PaddlePaddle/Paddle/pull/37527))

- 为`paddle.optimizer.Momentum`优化器添加 multi_tensor 策略、为`Optimzizer`类的`clear_grad`添加`set_to_zero`分支。([#37564](https://github.com/PaddlePaddle/Paddle/pull/37564))

- 为`paddle.optimizer.Adam`优化器添加 multi_tensor 策略。([#38010](https://github.com/PaddlePaddle/Paddle/pull/38010))

- 为`paddle.optimizer.SGD`优化器添加 multi_precision 策略。([#38231](https://github.com/PaddlePaddle/Paddle/pull/38231))

- 为优化器 `state_dict` 方法添加存储 `master weight` 参数。([#39121](https://github.com/PaddlePaddle/Paddle/pull/39121))

- 添加支持 op CUDA bfloat16 混合精度训练，支持 O1、O2 模式，通过 `paddle.amp.auto_cast` 可开启上述训练模式。([#39029](https://github.com/PaddlePaddle/Paddle/pull/39029), [#39815](https://github.com/PaddlePaddle/Paddle/pull/39815))

- 为如下 ops 添加 bfloat16 CUDA Kernel：matmul、concat、split、dropout、reshape、slice、squeeze、stack、transpose、unbind、elementwize_max、elementwize_add、elementwize_mul、elementwize_sub、scale、sum、layer_norm、p_norm、reduce_sum、softmax、log_softmax、sigmoid、sqrt、softplus、square、gaussian_random、fill_constant、fill_any_like。([#39485](https://github.com/PaddlePaddle/Paddle/pull/39485), [#39380](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39395](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39402](https://github.com/PaddlePaddle/Paddle/pull/39402), [#39457](https://github.com/PaddlePaddle/Paddle/pull/39457), [#39461](https://github.com/PaddlePaddle/Paddle/pull/39461), [#39602](https://github.com/PaddlePaddle/Paddle/pull/39602), [#39716](https://github.com/PaddlePaddle/Paddle/pull/39716), [#39683](https://github.com/PaddlePaddle/Paddle/pull/39683), [#39843](https://github.com/PaddlePaddle/Paddle/pull/39843), [#39999](https://github.com/PaddlePaddle/Paddle/pull/39999), [#40004](https://github.com/PaddlePaddle/Paddle/pull/40004), [#40027](https://github.com/PaddlePaddle/Paddle/pull/40027))

- 为如下 ops 添加 bfloat16 CPU Kernel：dropout、reshape、slice、squeeze、unsqueeze、stack、transpose、unbind、elementwize_max、elementwise_mul、elementwise_sub、gather。([#39380](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39395](https://github.com/PaddlePaddle/Paddle/pull/39380), [#39402](https://github.com/PaddlePaddle/Paddle/pull/39402), [#39457](https://github.com/PaddlePaddle/Paddle/pull/39457), [#39461](https://github.com/PaddlePaddle/Paddle/pull/39461), [#39602](https://github.com/PaddlePaddle/Paddle/pull/39602), [#39716](https://github.com/PaddlePaddle/Paddle/pull/39716), [#39683](https://github.com/PaddlePaddle/Paddle/pull/39683))

- 支持打印 bfloat16 类型的 Tensor。([#39375](https://github.com/PaddlePaddle/Paddle/pull/39375), [#39370](https://github.com/PaddlePaddle/Paddle/pull/39370))

- 为`p_norm`、`elementwise_max` 、`fill_constant_batch_size_like``scatter`增加 FP16 计算支持。([#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [#39907](https://github.com/PaddlePaddle/Paddle/pull/39907), [#38136](https://github.com/PaddlePaddle/Paddle/pull/38136), [#38499](https://github.com/PaddlePaddle/Paddle/pull/38499))

- 为如下 ops 增加 int16_t 支持：cumsum、less_than、less_equal、greater_than、greater_equal、equal、not_equal、fill_any_like、grather_nd、reduce_sum、where_index、reshape、unsqueeze。([#39636](https://github.com/PaddlePaddle/Paddle/pull/39636))

- 为 cross_entropy op 增加 int16_t label 类型的支持。([#39409](https://github.com/PaddlePaddle/Paddle/pull/39409))

- 为 embedding op 增加 int16_t id 类型的支持。([#39381](https://github.com/PaddlePaddle/Paddle/pull/39381))

- 为 reduce_mean op 增加 FP16 类型的支持。([#38289](https://github.com/PaddlePaddle/Paddle/pull/38289))

- 为 elementwise_min op 增加 FP16 类型的支持。([#38123](https://github.com/PaddlePaddle/Paddle/pull/38123))

- 更新 bfloat16 AMP oneDNN 默认支持列表。([#39304](https://github.com/PaddlePaddle/Paddle/pull/39304))

#### 飞桨高可复用算子库 PHI

针对飞桨框架原算子库存在的算子接口不清晰、算子复用成本较高、调用性能不够快的问题，我们重构了飞桨框架的算子库，设计了灵活、高效的函数式算子库 PHI，可以通过对函数式算子接口组合调用的方式实现新算子。新算子库提供了 200 余个跟 python 开发接口保持一致的 C++ 运算类 API，以及近 500 个可供组合调用的前、反向函数式算子内核 Kernel，可大幅降低框架原生算子和自定义算子的开发成本。新算子库支持 Primitive API 方式开发算子内核，可支持不同硬件（比如 GPU 和 XPU）的算子内核复用。新算子库支持以插件方式接入硬件（比如 NPU）的加速库，实现低成本复用硬件加速库。主要可分为以下几部分工作：

- **算子库基础架构、核心组件与机制实现**：合理规划新算子库的目录结构，设计实现了新算子库的公共基础数据结构、新的函数式 InferMeta 和 Kernel 开发范式以及相应的注册和管理组件，并且支持 Kernel 文件的自动化编译对象生成及编译依赖关系生成，使开发者仅需关注函数式 Kernel 的实现，开发范式简洁清晰。([#34425](https://github.com/PaddlePaddle/Paddle/pull/34425), [#37107](https://github.com/PaddlePaddle/Paddle/pull/37107), [#36946](https://github.com/PaddlePaddle/Paddle/pull/36946), [#36948](https://github.com/PaddlePaddle/Paddle/pull/36948), [#37876](https://github.com/PaddlePaddle/Paddle/pull/37876), [#37916](https://github.com/PaddlePaddle/Paddle/pull/37916), [#37977](https://github.com/PaddlePaddle/Paddle/pull/37977), [#38078](https://github.com/PaddlePaddle/Paddle/pull/38078), [#38861](https://github.com/PaddlePaddle/Paddle/pull/38861), [#39123](https://github.com/PaddlePaddle/Paddle/pull/39123), [#39131](https://github.com/PaddlePaddle/Paddle/pull/39131), [#39748](https://github.com/PaddlePaddle/Paddle/pull/39748), [#39790](https://github.com/PaddlePaddle/Paddle/pull/39790), [#39941](https://github.com/PaddlePaddle/Paddle/pull/39941), [#40239](https://github.com/PaddlePaddle/Paddle/pull/40239), [#40635](https://github.com/PaddlePaddle/Paddle/pull/40635), [#41091](https://github.com/PaddlePaddle/Paddle/pull/41091), [#37409](https://github.com/PaddlePaddle/Paddle/pull/37409), [#37942](https://github.com/PaddlePaddle/Paddle/pull/37942), [#39002](https://github.com/PaddlePaddle/Paddle/pull/39002), [#38109](https://github.com/PaddlePaddle/Paddle/pull/38109), [#37881](https://github.com/PaddlePaddle/Paddle/pull/37881), [#37517](https://github.com/PaddlePaddle/Paddle/pull/37517), [#39870](https://github.com/PaddlePaddle/Paddle/pull/39870), [#40975](https://github.com/PaddlePaddle/Paddle/pull/40975), [#39475](https://github.com/PaddlePaddle/Paddle/pull/39475), [#37304](https://github.com/PaddlePaddle/Paddle/pull/37304), #36910, #37120, #37146, #37215, #37255, #37369, #38258, #38257, #38355, #38853, #38937, #38977, #38946, #39085, #39153, #39228, #38301, #38275, #38506, #38607, #38473, #38632, #38811, #38880, #38996, #38914, #39101)

- **算子库 C++ API 体系建设**：设计实现了基于 yaml 配置文件的算子定义范式、自动生成了 200 余个 C++运算类 API，供内外部开发者复用，降低了基础运算的重复开发成本。([#37668](https://github.com/PaddlePaddle/Paddle/pull/37668), [#36938](https://github.com/PaddlePaddle/Paddle/pull/36938), [#38172](https://github.com/PaddlePaddle/Paddle/pull/38172), [#38182](https://github.com/PaddlePaddle/Paddle/pull/38182), [#38311](https://github.com/PaddlePaddle/Paddle/pull/38311), [#38438](https://github.com/PaddlePaddle/Paddle/pull/38438), [#39057](https://github.com/PaddlePaddle/Paddle/pull/39057), [#39229](https://github.com/PaddlePaddle/Paddle/pull/39229), [#39281](https://github.com/PaddlePaddle/Paddle/pull/39281), [#39263](https://github.com/PaddlePaddle/Paddle/pull/39263), [#39408](https://github.com/PaddlePaddle/Paddle/pull/39408), [#39436](https://github.com/PaddlePaddle/Paddle/pull/39436), [#39482](https://github.com/PaddlePaddle/Paddle/pull/39482), [#39497](https://github.com/PaddlePaddle/Paddle/pull/39497), [#39651](https://github.com/PaddlePaddle/Paddle/pull/39651), [#39521](https://github.com/PaddlePaddle/Paddle/pull/39521), [#39760](https://github.com/PaddlePaddle/Paddle/pull/39760), [#40060](https://github.com/PaddlePaddle/Paddle/pull/40060), [#40196](https://github.com/PaddlePaddle/Paddle/pull/40196), [#40218](https://github.com/PaddlePaddle/Paddle/pull/40218), [#40640](https://github.com/PaddlePaddle/Paddle/pull/40640), [#40732](https://github.com/PaddlePaddle/Paddle/pull/40732), [#40729](https://github.com/PaddlePaddle/Paddle/pull/40729), [#40840](https://github.com/PaddlePaddle/Paddle/pull/40840), [#40867](https://github.com/PaddlePaddle/Paddle/pull/40867), [#41025](https://github.com/PaddlePaddle/Paddle/pull/41025), [#41368](https://github.com/PaddlePaddle/Paddle/pull/41368))

- **算子库兼容各执行体系**：实现新的 InferMeta 及 Kernel 接入原动静态图执行体系、支持原 OpKernel 注册安全移除并迁移为新的 Kernel 形式。([#34425](https://github.com/PaddlePaddle/Paddle/pull/34425), [#38825](https://github.com/PaddlePaddle/Paddle/pull/38825), [#38837](https://github.com/PaddlePaddle/Paddle/pull/38837), [#38842](https://github.com/PaddlePaddle/Paddle/pull/38842), [#38976](https://github.com/PaddlePaddle/Paddle/pull/38976), [#39134](https://github.com/PaddlePaddle/Paddle/pull/39134), [#39140](https://github.com/PaddlePaddle/Paddle/pull/39140), [#39135](https://github.com/PaddlePaddle/Paddle/pull/39135), [#39252](https://github.com/PaddlePaddle/Paddle/pull/39252), [#39222](https://github.com/PaddlePaddle/Paddle/pull/39222), [#39351](https://github.com/PaddlePaddle/Paddle/pull/39351))

- **算子库底层数据结构及工具函数与框架解耦**：解除 Phi 在核心数据结构上对 框架的依赖，为后续 Phi 独立编译奠定基础，支持 infrt、自定义 Kernel 等一系列基于 Phi 的建设工作。([#38583](https://github.com/PaddlePaddle/Paddle/pull/38583), [#39188](https://github.com/PaddlePaddle/Paddle/pull/39188), [#39560](https://github.com/PaddlePaddle/Paddle/pull/39560), [#39931](https://github.com/PaddlePaddle/Paddle/pull/39931), [#39169](https://github.com/PaddlePaddle/Paddle/pull/39169), [#38951](https://github.com/PaddlePaddle/Paddle/pull/38951), [#38898](https://github.com/PaddlePaddle/Paddle/pull/38898), [#38873](https://github.com/PaddlePaddle/Paddle/pull/38873), [#38696](https://github.com/PaddlePaddle/Paddle/pull/38696), [#38651](https://github.com/PaddlePaddle/Paddle/pull/38651), [#39359](https://github.com/PaddlePaddle/Paddle/pull/39359), [#39305](https://github.com/PaddlePaddle/Paddle/pull/39305), [#39234](https://github.com/PaddlePaddle/Paddle/pull/39234), [#39098](https://github.com/PaddlePaddle/Paddle/pull/39098), [#39120](https://github.com/PaddlePaddle/Paddle/pull/39120), [#38979](https://github.com/PaddlePaddle/Paddle/pull/38979), [#38899](https://github.com/PaddlePaddle/Paddle/pull/38899), [#38844](https://github.com/PaddlePaddle/Paddle/pull/38844), [#39714](https://github.com/PaddlePaddle/Paddle/pull/39714), [#39729](https://github.com/PaddlePaddle/Paddle/pull/39729), [#39889](https://github.com/PaddlePaddle/Paddle/pull/39889), [#39587](https://github.com/PaddlePaddle/Paddle/pull/39587), [#39558](https://github.com/PaddlePaddle/Paddle/pull/39558), [#39514](https://github.com/PaddlePaddle/Paddle/pull/39514), [#39502](https://github.com/PaddlePaddle/Paddle/pull/39502), [#39300](https://github.com/PaddlePaddle/Paddle/pull/39300), [#39246](https://github.com/PaddlePaddle/Paddle/pull/39246), [#39124](https://github.com/PaddlePaddle/Paddle/pull/39124))

- **自定义算子机制与 Phi 整合并完善**：支持在自定义算子编写时调用 Phi 自动生成的 200 余个 C++运算类 API，降低自定义算子开发成本，并进行一系列问题修复。([#37122](https://github.com/PaddlePaddle/Paddle/pull/37122), [#37276](https://github.com/PaddlePaddle/Paddle/pull/37276), [#37281](https://github.com/PaddlePaddle/Paddle/pull/37281), [#37262](https://github.com/PaddlePaddle/Paddle/pull/37281), [#37415](https://github.com/PaddlePaddle/Paddle/pull/37415), [#37423](https://github.com/PaddlePaddle/Paddle/pull/37423), [#37583](https://github.com/PaddlePaddle/Paddle/pull/37683), [#38776](https://github.com/PaddlePaddle/Paddle/pull/38776), [#39353](https://github.com/PaddlePaddle/Paddle/pull/39353), [#41072](https://github.com/PaddlePaddle/Paddle/pull/41072))

- **算子规模化迁移改写**：迁移了约 250 个高频算子的前、反向算子内核 Kernel 至新算子库，改写为函数式，支持在 C++端通过调用多个基础 Kernel 函数封装，快速组合实现高性能算子；同时，添加相应的 yaml 算子定义，并接入新动态图执行体系，提升 python API 调度性能。迁移改写的算子包括：

  - sqrt ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - square ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - sin ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - sinh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - elementwise_fmax ([#40140](https://github.com/PaddlePaddle/Paddle/pull/40140))

  - elementwise_fmin ([#40140](https://github.com/PaddlePaddle/Paddle/pull/40140))

  - pool2d ([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - max_pool2d_with_index ([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - pool3d ([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - max_pool3d_with_index ([#40208](https://github.com/PaddlePaddle/Paddle/pull/40208), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - fill_constant ([#36930](https://github.com/PaddlePaddle/Paddle/pull/36930), [#39465](https://github.com/PaddlePaddle/Paddle/pull/39465))

  - p_norm ([#40819](https://github.com/PaddlePaddle/Paddle/pull/40819))

  - fill_constant_batch_size_like ([#40784](https://github.com/PaddlePaddle/Paddle/pull/40784))

  - conv2d ([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - conv2d_transpose ([#40675](https://github.com/PaddlePaddle/Paddle/pull/40675), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - conv3d ([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - conv3d_transpose ([#40675](https://github.com/PaddlePaddle/Paddle/pull/40675), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - mish ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - gather_nd ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - gather ([#40500](https://github.com/PaddlePaddle/Paddle/pull/40500))

  - scatter ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - scatter_nd_add ([#40090](https://github.com/PaddlePaddle/Paddle/pull/40090), [#40043](https://github.com/PaddlePaddle/Paddle/pull/40043))

  - sgd ([40045](https://github.com/PaddlePaddle/Paddle/pull/40045))

  - momentum ([#41319](https://github.com/PaddlePaddle/Paddle/pull/41319))

  - rmsprop ([#40994](https://github.com/PaddlePaddle/Paddle/pull/40994))

  - index_sample ([#38130](https://github.com/PaddlePaddle/Paddle/pull/38130), [#38459](https://github.com/PaddlePaddle/Paddle/pull/38459),[#39905](https://github.com/PaddlePaddle/Paddle/pull/39905))

  - adam ([#40351](https://github.com/PaddlePaddle/Paddle/pull/40351))

  - layer_norm ([#40193](https://github.com/PaddlePaddle/Paddle/pull/40193))

  - adagrad ([#40994](https://github.com/PaddlePaddle/Paddle/pull/40994/))

  - adamax ([#40173](https://github.com/PaddlePaddle/Paddle/pull/40173))

  - adadelta ([#40173](https://github.com/PaddlePaddle/Paddle/pull/40173))

  - clip ([#40602](https://github.com/PaddlePaddle/Paddle/pull/40602), [#41661](https://github.com/PaddlePaddle/Paddle/pull/41661), [#41675](https://github.com/PaddlePaddle/Paddle/pull/41675))

  - ceil ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - cos ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - atan ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - cosh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - erf ([#40388](https://github.com/PaddlePaddle/Paddle/pull/40388))

  - asin ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - acos ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - scale ([#39278](https://github.com/PaddlePaddle/Paddle/pull/39278))

  - elementwise_pow ([#40993](https://github.com/PaddlePaddle/Paddle/pull/40993))

  - elementwise_sub ([#39225](https://github.com/PaddlePaddle/Paddle/pull/39225), [#37260](https://github.com/PaddlePaddle/Paddle/pull/37260))

  - round ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - floor ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - pow ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - elementwise_floordiv ([#40993](https://github.com/PaddlePaddle/Paddle/pull/40993))

  - reciprocal ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - log1p ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - allclose ([#40469](https://github.com/PaddlePaddle/Paddle/pull/40469))

  - mul ([#40833](https://github.com/PaddlePaddle/Paddle/pull/40833))

  - elementwise_max ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_min ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_mod ([#40590](https://github.com/PaddlePaddle/Paddle/pull/40590))

  - elementwise_add ([#39048](https://github.com/PaddlePaddle/Paddle/pull/39048), [#37043](https://github.com/PaddlePaddle/Paddle/pull/37043))

  - matmul_v2 ([#36844](https://github.com/PaddlePaddle/Paddle/pull/36844), [#38713](https://github.com/PaddlePaddle/Paddle/pull/38713))

  - elementwise_mul ([#41042](https://github.com/PaddlePaddle/Paddle/pull/41042), [#40252](https://github.com/PaddlePaddle/Paddle/pull/40252), [#37471](https://github.com/PaddlePaddle/Paddle/pull/37471))

  - elementwise_div ([#40172](https://github.com/PaddlePaddle/Paddle/pull/40172), [#40039](https://github.com/PaddlePaddle/Paddle/pull/40039), [#37418](https://github.com/PaddlePaddle/Paddle/pull/37418))

  - SelectedRows ([#39037](https://github.com/PaddlePaddle/Paddle/pull/39037), [#39087](https://github.com/PaddlePaddle/Paddle/pull/39087), [#39128](https://github.com/PaddlePaddle/Paddle/pull/39128), [#39162](https://github.com/PaddlePaddle/Paddle/pull/39162), [#39236](https://github.com/PaddlePaddle/Paddle/pull/39236))

  - fill_any_like ([#39807](https://github.com/PaddlePaddle/Paddle/pull/39807))

  - dot ([#38359](https://github.com/PaddlePaddle/Paddle/pull/38359))

  - sum ([#40873](https://github.com/PaddlePaddle/Paddle/pull/40873))

  - cumsum ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - diag_v2 ([#39914](https://github.com/PaddlePaddle/Paddle/pull/39914))

  - auc ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - log_loss ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - one_hot_v2 ([39876](https://github.com/PaddlePaddle/Paddle/pull/39876))

  - sigmoid_cross_entropy_with_logits ([#39976](https://github.com/PaddlePaddle/Paddle/pull/39976), [#40200](https://github.com/PaddlePaddle/Paddle/pull/40200))

  - bce_loss ([#39868](https://github.com/PaddlePaddle/Paddle/pull/39868))

  - argsort ([#40151](https://github.com/PaddlePaddle/Paddle/pull/40151))

  - arg_max ([#40222](https://github.com/PaddlePaddle/Paddle/pull/40222))

  - arg_min ([#40222](https://github.com/PaddlePaddle/Paddle/pull/40222))

  - segment_pool ([#40099](https://github.com/PaddlePaddle/Paddle/pull/40099))

  - frobenius_norm ([#40707](https://github.com/PaddlePaddle/Paddle/pull/40707), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - dist ([#40178](https://github.com/PaddlePaddle/Paddle/pull/40178))

  - isnan_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - logical_and ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - logical_not ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - isfinite_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - logical_or ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - isinf_v2 ([#40076](https://github.com/PaddlePaddle/Paddle/pull/40076))

  - is_empty ([#39919](https://github.com/PaddlePaddle/Paddle/pull/39919))

  - logical_xor ([#39942](https://github.com/PaddlePaddle/Paddle/pull/39942))

  - less_than ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - not_equal ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - equal ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - less_equal ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - equal_all ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - uniform_random ([#39937](https://github.com/PaddlePaddle/Paddle/pull/39937))

  - randint ([#39876](https://github.com/PaddlePaddle/Paddle/pull/39876), [#41375](https://github.com/PaddlePaddle/Paddle/pull/41375))

  - randperm ([#41265](https://github.com/PaddlePaddle/Paddle/pull/41265))

  - unbind ([#39789](https://github.com/PaddlePaddle/Paddle/pull/39789))

  - bernoulli ([#39590](https://github.com/PaddlePaddle/Paddle/pull/39590))

  - increment ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - multinomial ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - addmm ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - cholesky ([#39858](https://github.com/PaddlePaddle/Paddle/pull/39858), [#39913](https://github.com/PaddlePaddle/Paddle/pull/39913))

  - where ([#39811](https://github.com/PaddlePaddle/Paddle/pull/39811))

  - log10 ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - log2 ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - expm1 ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - atan2 ([#39806](https://github.com/PaddlePaddle/Paddle/pull/39806))

  - gaussian_random ([#39932](https://github.com/PaddlePaddle/Paddle/pull/39932), [#40122](https://github.com/PaddlePaddle/Paddle/pull/40122), [#40191](https://github.com/PaddlePaddle/Paddle/pull/40191))

  - empty ([#38334](https://github.com/PaddlePaddle/Paddle/pull/38334))

  - truncated_gaussian_random ([#39971](https://github.com/PaddlePaddle/Paddle/pull/39971), [#40191](https://github.com/PaddlePaddle/Paddle/pull/40191))

  - mv ([#39861](https://github.com/PaddlePaddle/Paddle/pull/39861), [#39954](https://github.com/PaddlePaddle/Paddle/pull/39954))

  - tan ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - set_value ([#40195](https://github.com/PaddlePaddle/Paddle/pull/40195), [#40478](https://github.com/PaddlePaddle/Paddle/pull/40478), [#40636](https://github.com/PaddlePaddle/Paddle/pull/40636))

  - bitwise_and ([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - bitwise_not ([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - bitwise_or ([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - poisson ([#39814](https://github.com/PaddlePaddle/Paddle/pull/39814))

  - cholesky_solve ([#40387](https://github.com/PaddlePaddle/Paddle/pull/40387))

  - bitwise_xor ([#40031](https://github.com/PaddlePaddle/Paddle/pull/40031))

  - triangular_solve ([#40417](https://github.com/PaddlePaddle/Paddle/pull/40417))

  - sigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - atanh ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - softsign ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - thresholded_relu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - tanh_shrink ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - stanh ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - reduce_mean ([#37559](https://github.com/PaddlePaddle/Paddle/pull/37559))

  - reduce_max ([#40225](https://github.com/PaddlePaddle/Paddle/pull/40225))

  - reduce_min ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - mean ([#40872](https://github.com/PaddlePaddle/Paddle/pull/40872), [#41319](https://github.com/PaddlePaddle/Paddle/pull/41319))

  - reduce_all ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - reduce_any ([#40374](https://github.com/PaddlePaddle/Paddle/pull/40374))

  - logsumexp ([#40790](https://github.com/PaddlePaddle/Paddle/pull/40790))

  - softshrink ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - range ([#41265](https://github.com/PaddlePaddle/Paddle/pull/41265), [#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - stack ([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - tile ([#40371](https://github.com/PaddlePaddle/Paddle/pull/40371))

  - unique ([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - unstack ([#40581](https://github.com/PaddlePaddle/Paddle/pull/40851))

  - slice ([#40736](https://github.com/PaddlePaddle/Paddle/pull/40736))

  - transpose2 ([#39327](https://github.com/PaddlePaddle/Paddle/pull/39327))

  - unsqueeze2 ([#40596](https://github.com/PaddlePaddle/Paddle/pull/40596))

  - squeeze2 ([#40596](https://github.com/PaddlePaddle/Paddle/pull/40596))

  - strided_slice ([#40708](https://github.com/PaddlePaddle/Paddle/pull/40708))

  - softmax ([#39547](https://github.com/PaddlePaddle/Paddle/pull/39547))

  - leaky_relu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - gelu ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - prelu ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - log_softmax ([#40393](https://github.com/PaddlePaddle/Paddle/pull/40393))

  - elu ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - logsigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - psroi_pool ([#40353](https://github.com/PaddlePaddle/Paddle/pull/40353), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - kthvalue（[#40575](https://github.com/PaddlePaddle/Paddle/pull/40575))

  - mode ([#40571](https://github.com/PaddlePaddle/Paddle/pull/40571))

  - yolo_box ([#40112](https://github.com/PaddlePaddle/Paddle/pull/40112))

  - yolov3_loss ([#40944](https://github.com/PaddlePaddle/Paddle/pull/40944))

  - temporal_shift ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - depthwise_conv2d ([#39354](https://github.com/PaddlePaddle/Paddle/pull/39354))

  - pad3d ([#40701](https://github.com/PaddlePaddle/Paddle/pull/40701))

  - pad ([#40012](https://github.com/PaddlePaddle/Paddle/pull/40012))

  - greater_equal ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - kldiv_loss ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - isclose ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - silu ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - unfold ([#39778](https://github.com/PaddlePaddle/Paddle/pull/39778))

  - batch_norm ([39347](https://github.com/PaddlePaddle/Paddle/pull/39347))

  - norm ([#39324](https://github.com/PaddlePaddle/Paddle/pull/39324))

  - roi_pool ([#40574](https://github.com/PaddlePaddle/Paddle/pull/40574), [#40682](https://github.com/PaddlePaddle/Paddle/pull/40682), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - roi_align ([#40382](https://github.com/PaddlePaddle/Paddle/pull/40382), [#40556](https://github.com/PaddlePaddle/Paddle/pull/40556), [#41402](https://github.com/PaddlePaddle/Paddle/pull/41402))

  - deformable_conv ([#40700](https://github.com/PaddlePaddle/Paddle/pull/40700), [#40794](https://github.com/PaddlePaddle/Paddle/pull/40794), [#41644](https://github.com/PaddlePaddle/Paddle/pull/41644))

  - deformable_conv_v1 ([#40794](https://github.com/PaddlePaddle/Paddle/pull/40794), [#41644](https://github.com/PaddlePaddle/Paddle/pull/41644))

  - label_smooth ([#39796](https://github.com/PaddlePaddle/Paddle/pull/39796))

  - grid_sampler ([#40585](https://github.com/PaddlePaddle/Paddle/pull/40585))

  - greater_than ([#39970](https://github.com/PaddlePaddle/Paddle/pull/39970))

  - pixel_shuffle ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - nearest_interp_v2 ([#40855](https://github.com/PaddlePaddle/Paddle/pull/40855))

  - bilinear_interp_v2 ([#40855](https://github.com/PaddlePaddle/Paddle/pull/40855))

  - softmax_with_cross_entropy ([#40832](https://github.com/PaddlePaddle/Paddle/pull/40832))

  - rnn ([#41007](https://github.com/PaddlePaddle/Paddle/pull/41007))

  - reverse ([#40791](https://github.com/PaddlePaddle/Paddle/pull/40791))

  - trace ([#39510](https://github.com/PaddlePaddle/Paddle/pull/39510))

  - kron ([#40427](https://github.com/PaddlePaddle/Paddle/pull/40427))

  - accuracy ([#39982](https://github.com/PaddlePaddle/Paddle/pull/39982))

  - gather_tree ([#40082](https://github.com/PaddlePaddle/Paddle/pull/40082), [#39844](https://github.com/PaddlePaddle/Paddle/pull/39844))

  - dropout ([#40148](https://github.com/PaddlePaddle/Paddle/pull/40148))

  - bincount ([#39947](https://github.com/PaddlePaddle/Paddle/pull/39947))

  - warpctc ([#41389](https://github.com/PaddlePaddle/Paddle/pull/41389), [#40023](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/40023))

  - multiplex ([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#40102](https://github.com/PaddlePaddle/Paddle/pull/40102))

  - qr ([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#40007](https://github.com/PaddlePaddle/Paddle/pull/40007))

  - assign_value ([#40967](https://github.com/PaddlePaddle/Paddle/pull/40967))

  - assign ([#40022](https://github.com/PaddlePaddle/Paddle/pull/40022))

  - cast ([#37610](https://github.com/PaddlePaddle/Paddle/pull/37610))

  - tril_triu ([#40007](https://github.com/PaddlePaddle/Paddle/pull/40007), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - where_index ([#40255](https://github.com/PaddlePaddle/Paddle/pull/40255))

  - index_select ([#40260](https://github.com/PaddlePaddle/Paddle/pull/40260), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - roll ([#40257](https://github.com/PaddlePaddle/Paddle/pull/40257), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - cumprod (熊昆 [#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - shard_index ([#40254](https://github.com/PaddlePaddle/Paddle/pull/40254))

  - reshape2 ([#40914](https://github.com/PaddlePaddle/Paddle/pull/40914), [#39631](https://github.com/PaddlePaddle/Paddle/pull/39631), [#38833](https://github.com/PaddlePaddle/Paddle/pull/38833), [#37164](https://github.com/PaddlePaddle/Paddle/pull/37164))

  - flip ([#39822](https://github.com/PaddlePaddle/Paddle/pull/39822), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - eye ([#39712](https://github.com/PaddlePaddle/Paddle/pull/39712), [#40105](https://github.com/PaddlePaddle/Paddle/pull/40105), [#41476](https://github.com/PaddlePaddle/Paddle/pull/41476))

  - lookup_table_v2 ([#39901](https://github.com/PaddlePaddle/Paddle/pull/39901))

  - searchsorted ([#40520](https://github.com/PaddlePaddle/Paddle/pull/40520), [#41053](https://github.com/PaddlePaddle/Paddle/pull/41053))

  - adamw ([#40351](https://github.com/PaddlePaddle/Paddle/pull/40351))

  - tanh ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - cross ([#39829](https://github.com/PaddlePaddle/Paddle/pull/39829))

  - concat ([#38955](https://github.com/PaddlePaddle/Paddle/pull/38955), [#41112](https://github.com/PaddlePaddle/Paddle/pull/41112))

  - split ([#39060](https://github.com/PaddlePaddle/Paddle/pull/39060))

  - linspace ([#40124](https://github.com/PaddlePaddle/Paddle/pull/40124))

  - huber_loss ([#39761](https://github.com/PaddlePaddle/Paddle/pull/39761))

  - hierarchical_sigmoid ([#40553](https://github.com/PaddlePaddle/Paddle/pull/40553))

  - nll_loss ([#39936](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/39936))

  - graph_send_recv ([#40092](https://github.com/PaddlePaddle/Paddle/pull/40092), [#40320](https://github.com/PaddlePaddle/Paddle/pull/40320))

  - abs ([#39492](https://github.com/PaddlePaddle/Paddle/pull/39492), [#39762](https://github.com/PaddlePaddle/Paddle/pull/39762))

  - exp ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - rsqrt ([#40727](https://github.com/PaddlePaddle/Paddle/pull/40727))

  - viterbi_decode ([#40186](https://github.com/PaddlePaddle/Paddle/pull/40186))

  - conj ([#38247](https://github.com/PaddlePaddle/Paddle/pull/38247))

  - real ([#39777](https://github.com/PaddlePaddle/Paddle/pull/39777), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - imag ([#39777](https://github.com/PaddlePaddle/Paddle/pull/39777), [#41173](https://github.com/PaddlePaddle/Paddle/pull/41173))

  - take_along_axis ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40270](https://github.com/PaddlePaddle/Paddle/pull/40270), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - put_along_axis ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - lgamma ([#39770](https://github.com/PaddlePaddle/Paddle/pull/39770))

  - relu ([#40175](https://github.com/PaddlePaddle/Paddle/pull/40175))

  - maxout ([#39959](https://github.com/PaddlePaddle/Paddle/pull/39959), [#40974](https://github.com/PaddlePaddle/Paddle/pull/40974))

  - log ([#40785](https://github.com/PaddlePaddle/Paddle/pull/40785))

  - bilinear_tensor_product ([#39903](https://github.com/PaddlePaddle/Paddle/pull/39903))

  - flatten_contiguous_range ([#38712](https://github.com/PaddlePaddle/Paddle/pull/38712), [#36957](https://github.com/PaddlePaddle/Paddle/pull/36957), [#41345](https://github.com/PaddlePaddle/Paddle/pull/41345))

  - matrix_rank ([#40074](https://github.com/PaddlePaddle/Paddle/pull/40074), [#40519](https://github.com/PaddlePaddle/Paddle/pull/40519), [#41466](https://github.com/PaddlePaddle/Paddle/pull/41466))

  - logit ([#37844](https://github.com/PaddlePaddle/Paddle/pull/37844))

  - lerp ([#40105](https://github.com/PaddlePaddle/Paddle/pull/40105), [#39524](https://github.com/PaddlePaddle/Paddle/pull/39524))

  - erfinv ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - broadcast_tensors ([#40047](https://github.com/PaddlePaddle/Paddle/pull/40047))

  - gumbel_softmax ([#39873](https://github.com/PaddlePaddle/Paddle/pull/39873))

  - diagonal ([#39575](https://github.com/PaddlePaddle/Paddle/pull/39575))

  - trunc ([#39543](https://github.com/PaddlePaddle/Paddle/pull/39543), [#39772](https://github.com/PaddlePaddle/Paddle/pull/39772))

  - multi_dot ([#40038](https://github.com/PaddlePaddle/Paddle/pull/40038))

  - matrix_power ([#40231](https://github.com/PaddlePaddle/Paddle/pull/40231))

  - digamma ([#39240](https://github.com/PaddlePaddle/Paddle/pull/39240))

  - masked_select ([#39193](https://github.com/PaddlePaddle/Paddle/pull/39193))

  - determinant ([#40539](https://github.com/PaddlePaddle/Paddle/pull/40539))

  - eigh ([#40213](https://github.com/PaddlePaddle/Paddle/pull/40213))

  - size ([#39949](https://github.com/PaddlePaddle/Paddle/pull/39949), [#39712](https://github.com/PaddlePaddle/Paddle/pull/39712))

  - shape ([#40248](https://github.com/PaddlePaddle/Paddle/pull/40248))

  - reduce_sum ([#37559](https://github.com/PaddlePaddle/Paddle/pull/37559), [#41295](https://github.com/PaddlePaddle/Paddle/pull/41295))

  - reduce_prod ([#39844](https://github.com/PaddlePaddle/Paddle/pull/39844))

  - histogram ([#39496](https://github.com/PaddlePaddle/Paddle/pull/39496))

  - meshgrid ([#41411](https://github.com/PaddlePaddle/Paddle/pull/41411))

  - brelu ([#40385](https://github.com/PaddlePaddle/Paddle/pull/40385))

  - hard_swish ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - hard_shrink ([#40565](https://github.com/PaddlePaddle/Paddle/pull/40565))

  - selu (熊昆 [#39819](https://github.com/PaddlePaddle/Paddle/pull/39819))

  - expand_v2 ([#39471](https://github.com/PaddlePaddle/Paddle/pull/39471))

  - top_k_v2 ([#40064](https://github.com/PaddlePaddle/Paddle/pull/40064))

  - expand_as_v2 ([#40373](https://github.com/PaddlePaddle/Paddle/pull/40373))

  - swish ([#40913](https://github.com/PaddlePaddle/Paddle/pull/40913))

  - hard_sigmoid ([#40626](https://github.com/PaddlePaddle/Paddle/pull/40626))

  - exp, det, assign, gaussian_random, matrix_rank, eye, deformable_conv。([#41755]exp, det, assign, gaussian_random, matrix_rank, eye, deformable_conv。([#41755](https://github.com/PaddlePaddle/Paddle/pull/41755), [#41737](https://github.com/PaddlePaddle/Paddle/pull/41737)

#### 新动态图执行机制

针对飞桨原动态图执行机制的调度性能、二次开发能力差的问题，我们重构了动态图的底层执行机制。通过全新的调用执行方式，配合 Phi 算子库进行高效的运行时执行，对于 Phi 算子库支持的算子，切换到新动态图模式能体验到调度性能有较大幅度的提升。但是由于整体框架执行机制升级的工作量巨大，且该部分工作耦合了大量 Phi 算子库的工作， 因此在这个版本下我们仍未默认使用该执行方式。如果想要试用可以通过设置环境变量 `FLAGS_enable_eager_mode=1` 来切换使用。具体包括如下内容：

- **新动态图执行机制基础架构、核心组件与机制实现**：静态化动态图相关执行代码，将原本的同质化的算子构建变成针对不同 Phi API 的特异化调用从而极大的优化了调度开销。([#36059](https://github.com/PaddlePaddle/Paddle/pull/36059), [#37323](https://github.com/PaddlePaddle/Paddle/pull/37323), [#37556](https://github.com/PaddlePaddle/Paddle/pull/37556), [#37555](https://github.com/PaddlePaddle/Paddle/pull/37555), [#37478](https://github.com/PaddlePaddle/Paddle/pull/37478), [#37458](https://github.com/PaddlePaddle/Paddle/pull/37458), [#37479](https://github.com/PaddlePaddle/Paddle/pull/37479), [#37599](https://github.com/PaddlePaddle/Paddle/pull/37599), [#37659](https://github.com/PaddlePaddle/Paddle/pull/37659), [#37654](https://github.com/PaddlePaddle/Paddle/pull/37654), [#39200](https://github.com/PaddlePaddle/Paddle/pull/39200), [#39309](https://github.com/PaddlePaddle/Paddle/pull/39309), [#39319](https://github.com/PaddlePaddle/Paddle/pull/39319), [#39414](https://github.com/PaddlePaddle/Paddle/pull/39414), [#39504](https://github.com/PaddlePaddle/Paddle/pull/39504), [#39526](https://github.com/PaddlePaddle/Paddle/pull/39526), [#39878](https://github.com/PaddlePaddle/Paddle/pull/39878), [#39963](https://github.com/PaddlePaddle/Paddle/pull/39963))

- **新动态图执行机制子功能开发、适配**：支持了更加灵活，更加完备的动态图子功能例如 hook，pylayer，double_grad, inplace，amp 等等。([#41396](https://github.com/PaddlePaddle/Paddle/pull/41396), [#40400](https://github.com/PaddlePaddle/Paddle/pull/40400), [#40695](https://github.com/PaddlePaddle/Paddle/pull/40695), [#41043](https://github.com/PaddlePaddle/Paddle/pull/41043), [#40915](https://github.com/PaddlePaddle/Paddle/pull/40915), [#41104](https://github.com/PaddlePaddle/Paddle/pull/41104), [#41350](https://github.com/PaddlePaddle/Paddle/pull/41350), [#41209](https://github.com/PaddlePaddle/Paddle/pull/41209), [#40830](https://github.com/PaddlePaddle/Paddle/pull/40830), [#40891](https://github.com/PaddlePaddle/Paddle/pull/40891), [#36814](https://github.com/PaddlePaddle/Paddle/pull/36814), [#37377](https://github.com/PaddlePaddle/Paddle/pull/37377), [#37193](https://github.com/PaddlePaddle/Paddle/pull/37193), [#36965](https://github.com/PaddlePaddle/Paddle/pull/36965), [#37810](https://github.com/PaddlePaddle/Paddle/pull/37810), [#36837](https://github.com/PaddlePaddle/Paddle/pull/36837), [#38488](https://github.com/PaddlePaddle/Paddle/pull/38488), [#39282](https://github.com/PaddlePaddle/Paddle/pull/39282), [#39449](https://github.com/PaddlePaddle/Paddle/pull/39449), [#39531](https://github.com/PaddlePaddle/Paddle/pull/39531), [#39638](https://github.com/PaddlePaddle/Paddle/pull/39638), [#39674](https://github.com/PaddlePaddle/Paddle/pull/39674), [#39893](https://github.com/PaddlePaddle/Paddle/pull/39893), [#40170](https://github.com/PaddlePaddle/Paddle/pull/40170), [#40693](https://github.com/PaddlePaddle/Paddle/pull/40693), [#40937](https://github.com/PaddlePaddle/Paddle/pull/40937), [#41016](https://github.com/PaddlePaddle/Paddle/pull/41016), [#41051](https://github.com/PaddlePaddle/Paddle/pull/41051), [#41121](https://github.com/PaddlePaddle/Paddle/pull/41121), [#41198](https://github.com/PaddlePaddle/Paddle/pull/41198), [#41287](https://github.com/PaddlePaddle/Paddle/pull/41287), [#41380](https://github.com/PaddlePaddle/Paddle/pull/41380), [#41306](https://github.com/PaddlePaddle/Paddle/pull/41306), [#41387](https://github.com/PaddlePaddle/Paddle/pull/41387), [#40623](https://github.com/PaddlePaddle/Paddle/pull/40623), [#40945](https://github.com/PaddlePaddle/Paddle/pull/40945), [#39282](https://github.com/PaddlePaddle/Paddle/pull/39282), [#39449](https://github.com/PaddlePaddle/Paddle/pull/39449), [#38488](https://github.com/PaddlePaddle/Paddle/pull/38488))

- **新动态图执行的自动代码生成机制**：当我们为了将大量的同质化算子的计算和调度逻辑分化成不同的特异化的调度逻辑时，我们发现这是一个非常庞大的工作，因此我们引入了全新的自动代码生成逻辑来生成代码从而简化动态图的运行时逻辑。同时，为了能够适配之前框架中的各类运行时逻辑，我们也利用了一些复杂的编译手段来运行时的获取信息从而生成更加准确的调度代码。([#37574](https://github.com/PaddlePaddle/Paddle/pull/37574), [#37575](https://github.com/PaddlePaddle/Paddle/pull/37575), [#37639](https://github.com/PaddlePaddle/Paddle/pull/37639), [#37723](https://github.com/PaddlePaddle/Paddle/pull/37723), [#37753](https://github.com/PaddlePaddle/Paddle/pull/37753), [#37812](https://github.com/PaddlePaddle/Paddle/pull/37812), [#37837](https://github.com/PaddlePaddle/Paddle/pull/37837), [#37910](https://github.com/PaddlePaddle/Paddle/pull/37910), [#37943](https://github.com/PaddlePaddle/Paddle/pull/37943), [#37992](https://github.com/PaddlePaddle/Paddle/pull/37992), [#37959](https://github.com/PaddlePaddle/Paddle/pull/37959), [#38017](https://github.com/PaddlePaddle/Paddle/pull/38017), [#37969](https://github.com/PaddlePaddle/Paddle/pull/37969), [#38160](https://github.com/PaddlePaddle/Paddle/pull/38160), [#38085](https://github.com/PaddlePaddle/Paddle/pull/38085), [#38562](https://github.com/PaddlePaddle/Paddle/pull/38562), [#38573](https://github.com/PaddlePaddle/Paddle/pull/38573), [#39192](https://github.com/PaddlePaddle/Paddle/pull/39192), [#39215](https://github.com/PaddlePaddle/Paddle/pull/39215), [#39355](https://github.com/PaddlePaddle/Paddle/pull/39355), [#39358](https://github.com/PaddlePaddle/Paddle/pull/39358), [#39328](https://github.com/PaddlePaddle/Paddle/pull/39328), [#39233](https://github.com/PaddlePaddle/Paddle/pull/39233), [#39628](https://github.com/PaddlePaddle/Paddle/pull/39628), [#39767](https://github.com/PaddlePaddle/Paddle/pull/39767), [#39743](https://github.com/PaddlePaddle/Paddle/pull/39743), [#39897](https://github.com/PaddlePaddle/Paddle/pull/39897), [#39797](https://github.com/PaddlePaddle/Paddle/pull/39797), [#39997](https://github.com/PaddlePaddle/Paddle/pull/39997), [#40058](https://github.com/PaddlePaddle/Paddle/pull/40058), [#40080](https://github.com/PaddlePaddle/Paddle/pull/40080), [#40107](https://github.com/PaddlePaddle/Paddle/pull/40107), [#39962](https://github.com/PaddlePaddle/Paddle/pull/39962), [#40132](https://github.com/PaddlePaddle/Paddle/pull/40132), [#40276](https://github.com/PaddlePaddle/Paddle/pull/40276), [#40266](https://github.com/PaddlePaddle/Paddle/pull/40266), [#40480](https://github.com/PaddlePaddle/Paddle/pull/40480), [#40482](https://github.com/PaddlePaddle/Paddle/pull/40482), [#40368](https://github.com/PaddlePaddle/Paddle/pull/40368), [#40650](https://github.com/PaddlePaddle/Paddle/pull/40650), [#40815](https://github.com/PaddlePaddle/Paddle/pull/40815), [#40907](https://github.com/PaddlePaddle/Paddle/pull/40907), [#40935](https://github.com/PaddlePaddle/Paddle/pull/40935), [#41089](https://github.com/PaddlePaddle/Paddle/pull/41089))

- **新动态图执行机制接入主框架，联合调试**：我们目前利用一些环境变量区分静态图模式和动态图模式（含新动态图和老动态图模式），这些模式下我们已经适配了大部分的动态图的逻辑，但是仍有大量问题正在修复中。([#37638](https://github.com/PaddlePaddle/Paddle/pull/37638), [#37643](https://github.com/PaddlePaddle/Paddle/pull/37643), [#37653](https://github.com/PaddlePaddle/Paddle/pull/37653), [#38314](https://github.com/PaddlePaddle/Paddle/pull/38314), [#38337](https://github.com/PaddlePaddle/Paddle/pull/38337), [#38338](https://github.com/PaddlePaddle/Paddle/pull/38338), [#39164](https://github.com/PaddlePaddle/Paddle/pull/39164), [#39326](https://github.com/PaddlePaddle/Paddle/pull/39326), [#40391](https://github.com/PaddlePaddle/Paddle/pull/40391), [#40201](https://github.com/PaddlePaddle/Paddle/pull/40201), [#40854](https://github.com/PaddlePaddle/Paddle/pull/40854), [#40887](https://github.com/PaddlePaddle/Paddle/pull/40887))

- **更新了动态图下的一些判断逻辑，支持兼容形态下的动态图快速执行路径**：([#40786](https://github.com/PaddlePaddle/Paddle/pull/40786))

  - 非静态图模式（目前的过渡方案）：`_non_static_mode()`。

  - 在动态图模式下且判断在新动态图（推荐的判断逻辑）：`_in_dygrah_mode()`。

  - 在动态图模式下且判断在老动态图（不推荐的判断逻辑，在将来的版本中将废弃）：`_in_legacy_dygraph()`。

  - 在动态图模式下开启老动态图并关闭新动态图：`_enable_legacy_dygraph()` 或者退出 `_test_eager_guard()`。

  - 在动态图模式下开启新动态图并关闭老动态图：`_disable_legacy_dygraph()` 或者 `with _test_eager_guard()`。

  - 在静态图或者动态图模式下判断在新动态图：`_in_eager_without_dygraph_check()`。

- **动态图重构后支持 inplace 策略**：输入与输出为同一个 Tensor。

  - 为动态图重构中间态适配 inplace 策略。([#40400](https://github.com/PaddlePaddle/Paddle/pull/40400))

  - 为动态图重构最终态适配 inplace 策略。([#40695](https://github.com/PaddlePaddle/Paddle/pull/40695))

  - 动态图重构后，为 PyLayer 功能添加 inplace 策略。([#41043](https://github.com/PaddlePaddle/Paddle/pull/41043))

  - 动态图重构后，为 Tensor 的 setitem 功能添加 inplace 策略。([#40915](https://github.com/PaddlePaddle/Paddle/pull/40915))

  - 动态图重构后添加`_reset_grad_inplace_version`接口，将 Tensor 的梯度的 inplace version 置为 0。([#41101](https://github.com/PaddlePaddle/Paddle/pull/41101))

  - 反向计算过程中如果不需要前向 Tensor 的值（no need buffer 属性），则不需要对该 Tensor 进行 inplace version 的检测操作。 为 no_need_buffer 的 Tensor 跳过 inplace version 的检查。([#41350](https://github.com/PaddlePaddle/Paddle/pull/41350))

  - 统一动态图重构后与重构前对 inplace version 检查的报错信息。([#41209](https://github.com/PaddlePaddle/Paddle/pull/41209))

- **动态图重构后支持 view 策略**：输入与输出 Tensor 共享底层数据。

  - 为动态图重构中间态适配 view 机制。包括`reshape`、`squeeze`、`unsqueeze`、`flatten` API。([#40830](https://github.com/PaddlePaddle/Paddle/pull/40830))

  - 为动态图重构最终态适配 view 机制。包括`reshape` API。([#40891](https://github.com/PaddlePaddle/Paddle/pull/40891))

- **添加支持新动态图 eager Tensor 在 python 端的 weakref**。([#41797](https://github.com/PaddlePaddle/Paddle/pull/41797))

- **增强新动态图 DoubleGrad 功能**，支持基础的 DoubleGrad 功能。([#41893](https://github.com/PaddlePaddle/Paddle/pull/41893), [#41894](https://github.com/PaddlePaddle/Paddle/pull/41894), [#41895](https://github.com/PaddlePaddle/Paddle/pull/41895))

- **新增 `core.eager.StringTensor` 接口**，支持在 python 端构造 StringTensor 以及使用 StringTensor 相关 API。([#41039](https://github.com/PaddlePaddle/Paddle/pull/41039))

- **为 `core.eager.Tensor` 新增 `*grad_name` 和 `_grad_value` API**，返回梯度的名称和值。([#41990](https://github.com/PaddlePaddle/Paddle/pull/41990))

- **为动态图中间态添加对 no_need_buffer 属性的处理**。在 inplace 反向检查操作中，会跳过具有 no_need_buffer 属性的 Tensor 的检查。([#41720](https://github.com/PaddlePaddle/Paddle/pull/41720))


#### 全新静态图执行器
为了解决飞桨原静态图执行器在部分场景下调度性能不够理想，不便于扩展多 stream 等问题，我们实现了全新的性能优越，易于扩展的静态图执行器，充分利用了多 stream、多线程的异步调度能力。新执行器相当于原执行器是兼容升级，目前已在单机单卡场景下默认使用，用户不需要在训练代码中做任何修改即可自动使用。当然，我们也提供了接口来切换回原执行器，用户可以通过设置环境变量 `FLAGS_USE_STANDALONE_EXECUTOR=false` 来切换回原执行器。([#41179](https://github.com/PaddlePaddle/Paddle/pull/41179)) 主要内容如下：

- 基础组件：用于执行器中多线程算子调度的高性能线程池 ([#35470](https://github.com/PaddlePaddle/Paddle/pull/35470), [#35930](https://github.com/PaddlePaddle/Paddle/pull/35930), [#36030](https://github.com/PaddlePaddle/Paddle/pull/36030), [#36480](https://github.com/PaddlePaddle/Paddle/pull/36480), [#36688](https://github.com/PaddlePaddle/Paddle/pull/36688), [#36740](https://github.com/PaddlePaddle/Paddle/pull/36740), [#38335](https://github.com/PaddlePaddle/Paddle/pull/38335), [#40770](https://github.com/PaddlePaddle/Paddle/pull/40770)) 及线程协同组件 ([#38779](https://github.com/PaddlePaddle/Paddle/pull/38779), [#40876](https://github.com/PaddlePaddle/Paddle/pull/40876), [#40912](https://github.com/PaddlePaddle/Paddle/pull/40912))，算子执行后及时地显存回收 ([#37642](https://github.com/PaddlePaddle/Paddle/pull/37642), [#39617](https://github.com/PaddlePaddle/Paddle/pull/39617), [#40859](https://github.com/PaddlePaddle/Paddle/pull/40859))，并行执行器新依赖分析算法 ([#37231](https://github.com/PaddlePaddle/Paddle/pull/37231)) 等。

- 调度逻辑：优化执行器中算子的调度方法，支持多 stream 的多线程异步调度机制，将数据类型、设备、布局等转换改为算子调度以提升性能，支持缓存算子 Kernel 选择，支持选择全新 Phi 算子等。([#35024](https://github.com/PaddlePaddle/Paddle/pull/35024), [#34922](https://github.com/PaddlePaddle/Paddle/pull/34922), [#35711](https://github.com/PaddlePaddle/Paddle/pull/35711), [#35928](https://github.com/PaddlePaddle/Paddle/pull/35928), [#39458](https://github.com/PaddlePaddle/Paddle/pull/39458)，[#36899](https://github.com/PaddlePaddle/Paddle/pull/36899))。

- 接口兼容：兼容原执行器的用户接口和功能，如对齐 python 端 Executor.run()、支持 Scope 中管理 Tensor 等，确保用户可以无感知地切换新执行器。([#37278](https://github.com/PaddlePaddle/Paddle/pull/37278), [#37379](https://github.com/PaddlePaddle/Paddle/pull/37379), [#37445](https://github.com/PaddlePaddle/Paddle/pull/37445), [#37510](https://github.com/PaddlePaddle/Paddle/pull/37510), [#40955](https://github.com/PaddlePaddle/Paddle/pull/40955), [#41778](https://github.com/PaddlePaddle/Paddle/pull/41178), [#41058](https://github.com/PaddlePaddle/Paddle/pull/41058), [#38584](https://github.com/PaddlePaddle/Paddle/pull/38584), [#37957](https://github.com/PaddlePaddle/Paddle/pull/37957), [#37672](https://github.com/PaddlePaddle/Paddle/pull/37672), [#37474](https://github.com/PaddlePaddle/Paddle/pull/37474), [#37085](https://github.com/PaddlePaddle/Paddle/pull/37085), [#37061](https://github.com/PaddlePaddle/Paddle/pull/37061), [#36945](https://github.com/PaddlePaddle/Paddle/pull/36945))

- 增强多线程场景下调试和报错功能，将子线程的报错捕获到主线程中统一抛出，以提升用户体验。([#36692](https://github.com/PaddlePaddle/Paddle/pull/36692)，[#36802](https://github.com/PaddlePaddle/Paddle/pull/36802))

- 修复新执行器通信流重置 Allocator 中 stream 缓存信息的问题，减少跨 stream 场景下的 RecordStream 开销，优化后 DeepFM 模型性能提升约 8%。([#42046](https://github.com/PaddlePaddle/Paddle/pull/42046))

- 优化新执行器算子间的依赖分析方法，提升运行性能；为 send/recv 通信算子建立正确依赖以支持流水线并行。([#42009](https://github.com/PaddlePaddle/Paddle/pull/42009))


#### 分布式训练

- 集合通信多机多卡训练基础功能

  - 新增弹性功能（含节点故障、扩容、缩容），提升分布式的容错能力。([#36684](https://github.com/PaddlePaddle/Paddle/pull/36684), [#37177](https://github.com/PaddlePaddle/Paddle/pull/37177), [#37781](https://github.com/PaddlePaddle/Paddle/pull/37781))

  - Launch 启动模块，重构并新增 `master` 协同和节点个数 `nnodes` 定义，提升分布式启动易用性。([#40086](https://github.com/PaddlePaddle/Paddle/pull/40086), [#40568](https://github.com/PaddlePaddle/Paddle/pull/40568), [#40782](https://github.com/PaddlePaddle/Paddle/pull/40782), [#40844](https://github.com/PaddlePaddle/Paddle/pull/40844), [#40936](https://github.com/PaddlePaddle/Paddle/pull/40936), [#41190](https://github.com/PaddlePaddle/Paddle/pull/41190), [#41314](https://github.com/PaddlePaddle/Paddle/pull/41314))

  - 新增对 GPU/NPU/XPU 多种硬件的异构训练的支持。([#37613](https://github.com/PaddlePaddle/Paddle/pull/37613), [#37998](https://github.com/PaddlePaddle/Paddle/pull/37998))

  - 新增 fleet_executor 异步流水执行器。([#36966](https://github.com/PaddlePaddle/Paddle/pull/36966), [#37049](https://github.com/PaddlePaddle/Paddle/pull/37049), [#37087](https://github.com/PaddlePaddle/Paddle/pull/37087), [#37126](https://github.com/PaddlePaddle/Paddle/pull/37126), [#37150](https://github.com/PaddlePaddle/Paddle/pull/37150), [#37203](https://github.com/PaddlePaddle/Paddle/pull/37203), [#37167](https://github.com/PaddlePaddle/Paddle/pull/37167), [#37282](https://github.com/PaddlePaddle/Paddle/pull/37282), [#37319](https://github.com/PaddlePaddle/Paddle/pull/37319), [#37462](https://github.com/PaddlePaddle/Paddle/pull/37462), [#37507](https://github.com/PaddlePaddle/Paddle/pull/37507), [#37533](https://github.com/PaddlePaddle/Paddle/pull/37533), [#37576](https://github.com/PaddlePaddle/Paddle/pull/37576), [#37605](https://github.com/PaddlePaddle/Paddle/pull/37605), [#37691](https://github.com/PaddlePaddle/Paddle/pull/37691), [#37742](https://github.com/PaddlePaddle/Paddle/pull/37742), [#37783](https://github.com/PaddlePaddle/Paddle/pull/37783), [#37809](https://github.com/PaddlePaddle/Paddle/pull/37809), [#37862](https://github.com/PaddlePaddle/Paddle/pull/37862), [#37882](https://github.com/PaddlePaddle/Paddle/pull/37882), [#37934](https://github.com/PaddlePaddle/Paddle/pull/37934), [#38024](https://github.com/PaddlePaddle/Paddle/pull/38024), [#38083](https://github.com/PaddlePaddle/Paddle/pull/38083), [#38164](https://github.com/PaddlePaddle/Paddle/pull/38164), [#38261](https://github.com/PaddlePaddle/Paddle/pull/38261), [#38290](https://github.com/PaddlePaddle/Paddle/pull/38290), [#40607](https://github.com/PaddlePaddle/Paddle/pull/40607), [#37093](https://github.com/PaddlePaddle/Paddle/pull/37093), [#37106](https://github.com/PaddlePaddle/Paddle/pull/37106), [#37143](https://github.com/PaddlePaddle/Paddle/pull/37143), [#37338](https://github.com/PaddlePaddle/Paddle/pull/37338), [#37376](https://github.com/PaddlePaddle/Paddle/pull/37376), [#37485](https://github.com/PaddlePaddle/Paddle/pull/37485), [#37531](https://github.com/PaddlePaddle/Paddle/pull/37531), [#37623](https://github.com/PaddlePaddle/Paddle/pull/37623), [#37693](https://github.com/PaddlePaddle/Paddle/pull/37693), [#37755](https://github.com/PaddlePaddle/Paddle/pull/37755), [#37807](https://github.com/PaddlePaddle/Paddle/pull/37807), [#37889](https://github.com/PaddlePaddle/Paddle/pull/37889), [#38420](https://github.com/PaddlePaddle/Paddle/pull/38420), [#38539](https://github.com/PaddlePaddle/Paddle/pull/38539), [#36892](https://github.com/PaddlePaddle/Paddle/pull/36892), [#37084](https://github.com/PaddlePaddle/Paddle/pull/37084), [#37158](https://github.com/PaddlePaddle/Paddle/pull/37158), [#37361](https://github.com/PaddlePaddle/Paddle/pull/37361), [#37509](https://github.com/PaddlePaddle/Paddle/pull/37509), [#37603](https://github.com/PaddlePaddle/Paddle/pull/37603), [#37703](https://github.com/PaddlePaddle/Paddle/pull/37703), [#37824](https://github.com/PaddlePaddle/Paddle/pull/37824), [#38114](https://github.com/PaddlePaddle/Paddle/pull/38114), [#38322](https://github.com/PaddlePaddle/Paddle/pull/38322), [#38535](https://github.com/PaddlePaddle/Paddle/pull/38535), [#38650](https://github.com/PaddlePaddle/Paddle/pull/38650), [#38709](https://github.com/PaddlePaddle/Paddle/pull/38709), [#38799](https://github.com/PaddlePaddle/Paddle/pull/38799), [#38839](https://github.com/PaddlePaddle/Paddle/pull/38839), [#38904](https://github.com/PaddlePaddle/Paddle/pull/38904))

  - 新增分布式大模型推理功能。([#38795](https://github.com/PaddlePaddle/Paddle/pull/38795), [#39012](https://github.com/PaddlePaddle/Paddle/pull/39012), [#39032](https://github.com/PaddlePaddle/Paddle/pull/39032), [#39076](https://github.com/PaddlePaddle/Paddle/pull/39076), [#39194](https://github.com/PaddlePaddle/Paddle/pull/39194), [#39207](https://github.com/PaddlePaddle/Paddle/pull/39207), [#39241](https://github.com/PaddlePaddle/Paddle/pull/39241), [#39603](https://github.com/PaddlePaddle/Paddle/pull/39603), [#39758](https://github.com/PaddlePaddle/Paddle/pull/39758), [#39992](https://github.com/PaddlePaddle/Paddle/pull/39992))

- 动态图混合并行

  - 重构 `paddle.distributed.fleet.utils.recompute`，支持新动态图。([#41396](https://github.com/PaddlePaddle/Paddle/pull/41396))

  - 支持 Pure FP16 训练。([#36420](https://github.com/PaddlePaddle/Paddle/pull/36420))

  - 新增 MoE（Mixture of Experts）并行策略, 支持超大 MoE 模型训练。([#41092](https://github.com/PaddlePaddle/Paddle/pull/41092), [#40895](https://github.com/PaddlePaddle/Paddle/pull/40895), [#40850](https://github.com/PaddlePaddle/Paddle/pull/40580), [#39224](https://github.com/PaddlePaddle/Paddle/pull/39224))

  - 新增 GroupSharded 并行策略，支持 stage1、stage2、stage3 三个阶段模型状态分组切片训练策略，支持同、异步通信，并可与 Recompute、AMP O1\O2、Offload、GroupShardedClipGrad、GroupShardedScaler 等基础功能组合使用。([#37489](https://github.com/PaddlePaddle/Paddle/pull/37489), [#37568](https://github.com/PaddlePaddle/Paddle/pull/37568), [#37707](https://github.com/PaddlePaddle/Paddle/pull/37707), [#37836](https://github.com/PaddlePaddle/Paddle/pull/37836), [#37947](https://github.com/PaddlePaddle/Paddle/pull/37947), [#38151](https://github.com/PaddlePaddle/Paddle/pull/38151), [#38407](https://github.com/PaddlePaddle/Paddle/pull/38407), [#38052](https://github.com/PaddlePaddle/Paddle/pull/38052), [#39112](https://github.com/PaddlePaddle/Paddle/pull/39112), [#38989](https://github.com/PaddlePaddle/Paddle/pull/38989), [#39171](https://github.com/PaddlePaddle/Paddle/pull/39171), [#39285](https://github.com/PaddlePaddle/Paddle/pull/39285), [#39334](https://github.com/PaddlePaddle/Paddle/pull/39334), [#39397](https://github.com/PaddlePaddle/Paddle/pull/39397), [#39581](https://github.com/PaddlePaddle/Paddle/pull/39581), [#39668](https://github.com/PaddlePaddle/Paddle/pull/39668), [#40129](https://github.com/PaddlePaddle/Paddle/pull/40129), [#40396](https://github.com/PaddlePaddle/Paddle/pull/40396), [#40488](https://github.com/PaddlePaddle/Paddle/pull/40488), [#40601](https://github.com/PaddlePaddle/Paddle/pull/40601)，[#37725](https://github.com/PaddlePaddle/Paddle/pull/37725)，[#37904](https://github.com/PaddlePaddle/Paddle/pull/37904), [#38064](https://github.com/PaddlePaddle/Paddle/pull/38064))

- 静态图混合并行

  - 新增`scale_gradient`标志位至`gradient_scale_configs`，用于控制流水线并行下梯度聚合运算对梯度进行求平均运算的位置。([#36384](https://github.com/PaddlePaddle/Paddle/pull/36384))

  - 张量模型并行下，dropout 支持设置确定性随机种子生成器，以确保非分布式变量的随机一致性和分布式变量的随机性。([#36228](https://github.com/PaddlePaddle/Paddle/pull/36228))

  - NPU 混合并行支持 Offload，可节约 40%显存。([#37224](https://github.com/PaddlePaddle/Paddle/pull/37224))

  - 为 seed op 增加 `force_cpu` 可选参数，使 dropout 可以直接从 CPU 读取 seed 的值。([#35820](https://github.com/PaddlePaddle/Paddle/pull/35820))

  - 完善 Automatic Sparsity (ASP)sharding 策略，支持根据 program 选择 sharding 策略。(#[#40028](https://github.com/PaddlePaddle/Paddle/pull/40028))

- 自动并行

  - 新增逻辑进程与物理设备自动映射后的进程重新启动（relaunch）。([#37523](https://github.com/PaddlePaddle/Paddle/pull/37523), [#37326](https://github.com/PaddlePaddle/Paddle/pull/37326))

  - 完善自动并行底层机制和接口，利于各个模块统一和添加优化 pass。([#36617](https://github.com/PaddlePaddle/Paddle/pull/36617), [#38132](https://github.com/PaddlePaddle/Paddle/pull/38132))

  - 新增统一资源表示，支持逻辑进程与物理设备自动映射功能。([#37091](https://github.com/PaddlePaddle/Paddle/pull/37091), [#37482](https://github.com/PaddlePaddle/Paddle/pull/37482), [#37094](https://github.com/PaddlePaddle/Paddle/pull/37094))

  - 完善自动并行计算图反向和更新部分的分布式属性补全功能。([#36744](https://github.com/PaddlePaddle/Paddle/pull/36744))

  - 新增数据切分功能。([#36055](https://github.com/PaddlePaddle/Paddle/pull/36055))

  - 新增张量重切分功能，根据张量和算子的分布式属性对张量进行重新切分。([#40865](https://github.com/PaddlePaddle/Paddle/pull/40865), [#41106](https://github.com/PaddlePaddle/Paddle/pull/41106))

  - 新增资源数量或并行策略变化时分布式参数的自动转换功能。([#40434](https://github.com/PaddlePaddle/Paddle/pull/40434))

  - 新增梯度累加功能（GradientMerge），减少通信次数，提升训练效率。([#38259](https://github.com/PaddlePaddle/Paddle/pull/38259), [#40737](https://github.com/PaddlePaddle/Paddle/pull/40737))

  - 新增重计算功能(Recompute)，优化显存。([#38920](https://github.com/PaddlePaddle/Paddle/pull/38920))

  - 新增 Sharding 优化 pass， 支持 p-g-os 3 个 stage 的切分优化。([#38502](https://github.com/PaddlePaddle/Paddle/pull/38502))

  - 新增 AMP + FP16 优化 pass。([#38764](https://github.com/PaddlePaddle/Paddle/pull/38764), [#40615](https://github.com/PaddlePaddle/Paddle/pull/40615))

  - 新增 Transformer 类模型的 QKV fuse 切分。([#39080](https://github.com/PaddlePaddle/Paddle/pull/39080))

  - 新增 while op 的分布式属性推导功能，确保迭代推导算法能收敛。([#39939](https://github.com/PaddlePaddle/Paddle/pull/39939), [#39086](https://github.com/PaddlePaddle/Paddle/pull/39086), [#39014](https://github.com/PaddlePaddle/Paddle/pull/39014))

  - 支持子 block 和 while op 控制流的训练和推理。([#39612](https://github.com/PaddlePaddle/Paddle/pull/39612), [#39895](https://github.com/PaddlePaddle/Paddle/pull/39895), [#40077](https://github.com/PaddlePaddle/Paddle/pull/40077))

- 参数服务器

  - GPUPS 下，新增 NAN/INF 值检查工具。([#38131](https://github.com/PaddlePaddle/Paddle/pull/38131))

  - GPUPS 下，新增 set_date 接口，适配增量训练。([#36194](https://github.com/PaddlePaddle/Paddle/pull/36194))

  - GPUPS 下，新增异步 release dataset 功能。([#37790](https://github.com/PaddlePaddle/Paddle/pull/37790))

  - GPUPS 下，支持 Dump 参数和中间层 ([#36157](https://github.com/PaddlePaddle/Paddle/pull/36157))；

  - GPUPS 下，支持优化器参数配置。([#39783](https://github.com/PaddlePaddle/Paddle/pull/39783), [#39849](https://github.com/PaddlePaddle/Paddle/pull/39849))

  - 统一参数服务器下，重构通信、存储等各个模块基类，提升各个模块的易二次开发性。([#41207](https://github.com/PaddlePaddle/Paddle/pull/41207), [#41022](https://github.com/PaddlePaddle/Paddle/pull/41022), [#40702](https://github.com/PaddlePaddle/Paddle/pull/40702), [#39341](https://github.com/PaddlePaddle/Paddle/pull/39341) [#39377](https://github.com/PaddlePaddle/Paddle/pull/39377), [#39191](https://github.com/PaddlePaddle/Paddle/pull/39191), [#39064](https://github.com/PaddlePaddle/Paddle/pull/39064))

  - 统一参数服务器下，新增评估指标模块，支持 AUC/WuAUC/MaskAuc 等评估指标计算及可自定义扩展。([#38789](https://github.com/PaddlePaddle/Paddle/pull/38789))

  - 支持在昆仑 2 芯片上的 XPU 参数服务器训练。([#41917](https://github.com/PaddlePaddle/Paddle/pull/41917), [#42266](https://github.com/PaddlePaddle/Paddle/pull/42266), [#41916](https://github.com/PaddlePaddle/Paddle/pull/41916))

#### Profiler

- Python 层新增性能分析模块 `paddle.profiler`：提供对训推过程中性能数据的收集，导出和统计的功能。([#40065](https://github.com/PaddlePaddle/Paddle/pull/40065), [#40357](https://github.com/PaddlePaddle/Paddle/pull/40357), [#40888](https://github.com/PaddlePaddle/Paddle/pull/40888))

  - `paddle.profiler.Profiler`，性能分析器，用户交互的接口。([#41029](https://github.com/PaddlePaddle/Paddle/pull/41029), [#41524](https://github.com/PaddlePaddle/Paddle/pull/41524), [#41157](https://github.com/PaddlePaddle/Paddle/pull/41157), [#40249](https://github.com/PaddlePaddle/Paddle/pull/40249), [#40111](https://github.com/PaddlePaddle/Paddle/pull/40111), [#39964](https://github.com/PaddlePaddle/Paddle/pull/39964), [#40133](https://github.com/PaddlePaddle/Paddle/pull/40133))

  - `paddle.profiler.RecordEvent`，提供自定义打点来记录时间的功能。([#39693](https://github.com/PaddlePaddle/Paddle/pull/39693), [#39694](https://github.com/PaddlePaddle/Paddle/pull/39694), [#39695](https://github.com/PaddlePaddle/Paddle/pull/39695), [#39675](https://github.com/PaddlePaddle/Paddle/pull/39675),[#41445](https://github.com/PaddlePaddle/Paddle/pull/41445), [#41132](https://github.com/PaddlePaddle/Paddle/pull/41132))

  - `paddle.profiler.ProfilerTarget`，指定性能分析的目标设备。

  - `paddle.profiler.ProfilerState`，表示性能分析器的状态。

  - `paddle.profiler.SortedKeys`，指定统计表单内数据的排序方式。

  - `paddle.profiler.make_scheduler`，生成性能分析器状态的调度器，实现采集范围的周期性控制。

  - `paddle.profiler.export_chrome_tracing`，将性能数据保存到可供 chrome://tracing 插件查看的 google chrome tracing 文件。([#39316](https://github.com/PaddlePaddle/Paddle/pull/39316), [#39984](https://github.com/PaddlePaddle/Paddle/pull/39984), [#41029](https://github.com/PaddlePaddle/Paddle/pull/41029))

  - `paddle.profiler.export_protobuf`，将性能数据保存到内部结构表示的 protobuf 文件。([#39519](https://github.com/PaddlePaddle/Paddle/pull/39519), [#39109](https://github.com/PaddlePaddle/Paddle/pull/39109), [#39474](https://github.com/PaddlePaddle/Paddle/pull/39474))

  - `paddle.profiler.load_profiler_result`，载入所保存到 protobuf 文件的性能数据。

  - `paddle.profiler.Profiler`通过指定 `timer_only` 参数，对模型进行数据读取、step 开销和吞吐量的统计。([#40386](https://github.com/PaddlePaddle/Paddle/pull/40386))

- C++层重构 Profiler 底层基础设施

  - 重构 Profiler 的控制器架构。([#38826](https://github.com/PaddlePaddle/Paddle/pull/38826), [#39230](https://github.com/PaddlePaddle/Paddle/pull/39230), [#39779](https://github.com/PaddlePaddle/Paddle/pull/39779))

  - 新增 Host Tracer，收集主机侧性能指标。([#37629](https://github.com/PaddlePaddle/Paddle/pull/39629), [#37766](https://github.com/PaddlePaddle/Paddle/pull/37766), [#37944](https://github.com/PaddlePaddle/Paddle/pull/37944), [#38280](https://github.com/PaddlePaddle/Paddle/pull/38280), [#39975](https://github.com/PaddlePaddle/Paddle/pull/39975), [#40460](https://github.com/PaddlePaddle/Paddle/pull/40460))

  - 新增 CUDA Tracer，收集设备侧性能指标。([#39488](https://github.com/PaddlePaddle/Paddle/pull/39488))

  - Profiler 支持分级。([#39926](https://github.com/PaddlePaddle/Paddle/pull/39926))

- 修改新动态图下 op 的打点名称和类型。（[#41771](https://github.com/PaddlePaddle/Paddle/pull/41771/)

- 添加 Kernel 表单，以及优化表单内容的展示方式。([#41989](https://github.com/PaddlePaddle/Paddle/pull/41989))

- 消除 Profiler 关闭情况下对模型前向计算造成性能下降的影响。([#42142](https://github.com/PaddlePaddle/Paddle/pull/42142))

#### CINN 编译器接入

飞桨的编译器功能在逐步丰富中，针对 CINN ([GitHub - PaddlePaddle/CINN: Compiler Infrastructure for Neural Networks](https://github.com/PaddlePaddle/CINN)) 的变更，Paddle 侧接入也进行了相对应的更改，以适配编译器 CINN 的功能。其中主要包括增加 Paddle-CINN 运行流程的子图管理相关功能，显存和速度性能的优化、开发过程发现的 bug 修复。

- 功能开发：

  - 子图 op 相关：

    - 添加从计算图中找到并生成 CINN 子图的功能。([#36345](https://github.com/PaddlePaddle/Paddle/pull/36345))

    - 新增 cinn_launch op 作为运行时接入 CINN 的入口，负责调度 CINN 对子图进行编译、初始化数据空间、调度生成 Kernel 的执行。([#36600](https://github.com/PaddlePaddle/Paddle/pull/36600))

    - 为 cinn_launch op 的 Kernel 实现添加辅助类 CinnLaunchContext 管理子图编译、运行的中间数据，提升可扩展性和代码可读性。([#37938](https://github.com/PaddlePaddle/Paddle/pull/37938))

    - 为 CINN 子图添加额外的 fetch 结点，从而保证 CINN 外部结点能取到待 fetch 变量的值。([#37172](https://github.com/PaddlePaddle/Paddle/pull/37172), [#37190](https://github.com/PaddlePaddle/Paddle/pull/37190))

    - 添加对 CINN 子图符号化的功能，符号化用于拓扑排序子图并返回 CINN 执行序列。([#36417](https://github.com/PaddlePaddle/Paddle/pull/36417))

    - 新增 CinnCompiler 类，用于调用 CINN 编译模型中可使用 CINN 算子替换的子图。([#36562](https://github.com/PaddlePaddle/Paddle/pull/36562), [#36975](https://github.com/PaddlePaddle/Paddle/pull/36975))

    - 为 CINN 符号化类新增获取子图 fetch 变量名的接口，防止编译优化中将 fetch 变量融合消除。([#37218](https://github.com/PaddlePaddle/Paddle/pull/37218))

  - 程序开发检查、debug、API 变更相关：

    - 同步更新 CINN 中 NetBuilder API 名称的变化。([#40392](https://github.com/PaddlePaddle/Paddle/pull/40392))

    - 为 Paddle-CINN 添加必要的用于 debug 的日志信息。([#36867](https://github.com/PaddlePaddle/Paddle/pull/36867))

    - 添加 Paddle desc 与 CINN desc 互转函数。([#36100](https://github.com/PaddlePaddle/Paddle/pull/36100))

    - 相比 Paddle，CINN 中实现的算子可能存在未使用到某些输入变量，因此在 cinn_launch op 中去除对输入变量必须被使用的检查。([#37119](https://github.com/PaddlePaddle/Paddle/pull/37119))

    - 新增 cinn_instruction_run op 用于调用 CINN 执行单个生成指令，便于 Paddle 侧构建 Graph 调度运行子图。([#39435](https://github.com/PaddlePaddle/Paddle/pull/39435), [#39576](https://github.com/PaddlePaddle/Paddle/pull/39576))

    - 在 Paddle 中添加编译 CINN 所需的 CUDA/CUBLAS/MKL/CINN pass 应用等控制宏。([#37066](https://github.com/PaddlePaddle/Paddle/pull/37066), [#36660](https://github.com/PaddlePaddle/Paddle/pull/36660))

    - 增加 FLAGS_allow_cinn_ops 和 FLAGS_deny_cinn_ops 两个控制标记，用于控制 Paddle 训练中使用 CINN 算子代替原生算子的种类。([#36842](https://github.com/PaddlePaddle/Paddle/pull/36842))

- 性能优化：

  - 速度优化

    - 优化 CinnCacheKey 的计算耗时。([#37786](https://github.com/PaddlePaddle/Paddle/pull/37786), [#37317](https://github.com/PaddlePaddle/Paddle/pull/37317))

    - 缓存 CINN 编译子图的变量 scope，降低运行参数构造开销。([#37983](https://github.com/PaddlePaddle/Paddle/pull/37983))

    - 子图编译时接入 CINN 自动调优，支持通过 flag 启用，便于后续进一步调优训练性能。([#41795](https://github.com/PaddlePaddle/Paddle/pull/41795))

    - 重构子图编译时对编译结果的正确性校验，避免运行时重复检查，降低调度开销。([#41777](https://github.com/PaddlePaddle/Paddle/pull/41777))

    - 在 Paddle-CINN 训练功能中默认启用 TransposeFolding 和 GemmRewriter 优化 pass。([#41084](https://github.com/PaddlePaddle/Paddle/pull/41084))

    - 将 Paddle 中创建的 cuda stream 传入 CINN，使得 Paddle 和 CINN 执行计算时共用同一个 CUDA stream。([#37337](https://github.com/PaddlePaddle/Paddle/pull/37337))

    - 将 CINN 优化 pass 应用逻辑从 Paddle 中移动到 CINN 中。([#42047](https://github.com/PaddlePaddle/Paddle/pull/42047), [#42070](https://github.com/PaddlePaddle/Paddle/pull/42070))

  - 显存优化

    - 为 cinn_launch op 添加 NoNeedBufferVars 声明无须 buffer 的输入变量列表，以便显存优化提前释放无效空间。([#38367](https://github.com/PaddlePaddle/Paddle/pull/38367))

    - 传入子图外部变量的引用计数信息，便于 cinn_launch 内子图复用显存优化 pass，降低使用 CINN 的显存开销。([#39209](https://github.com/PaddlePaddle/Paddle/pull/39209), [#39622](https://github.com/PaddlePaddle/Paddle/pull/39622))

    - 添加 CINN 编译生成的可执行指令集合转换为 Paddle Graph 的功能，支持复用 Paddle 调度器及显存优化 pass，进一步降低使用 CINN 的显存开销。([#39724](https://github.com/PaddlePaddle/Paddle/pull/39724), [#39911](https://github.com/PaddlePaddle/Paddle/pull/39911))

    - 添加 cinn_instruction_run op 的 Kernel 支持根据编译结果推断的数据类型动态申请空间。([#40920](https://github.com/PaddlePaddle/Paddle/pull/40920))

- 问题修复：

  - 修复并优化 CINN 子图的生成逻辑。([#36503](https://github.com/PaddlePaddle/Paddle/pull/36503))

  - 修复 Paddle-CINN 不支持无输入子图的问题。([#40814](https://github.com/PaddlePaddle/Paddle/pull/40814))

  - 修复由于 CINN 无法处理 batch_norm 等算子中存在的无用输出而报错的问题。([#36996](https://github.com/PaddlePaddle/Paddle/pull/36996))

  - 修复若干 CINN 子图划分以及符号化中存在的 bug，解决 Paddle 训练接入 CINN 全流程打通过程中遇到的问题。([#36739](https://github.com/PaddlePaddle/Paddle/pull/36739), [#36698](https://github.com/PaddlePaddle/Paddle/pull/36698) )

  - CINN 尚不支持控制流，添加遇控制流跳过的逻辑。([#40812](https://github.com/PaddlePaddle/Paddle/pull/40812))

#### 其他

- 模型量化

  - 升级量化存储格式，并统一动、静态图量化格式。([#41041](https://github.com/PaddlePaddle/Paddle/pull/41041))

  - 新增离线量化方法：EMD、Adaround。([#40421](https://github.com/PaddlePaddle/Paddle/pull/40421), [#38460](https://github.com/PaddlePaddle/Paddle/pull/38460))

  - 支持更多 op 适配模 op 量化。([#40083](https://github.com/PaddlePaddle/Paddle/pull/40083))

  - 支持控制流中的 OP 量化。([#37498](https://github.com/PaddlePaddle/Paddle/pull/37498))

  - 新增支持 matmul_v2 OP 的量化。([#36469](https://github.com/PaddlePaddle/Paddle/pull/36469))

  - 新增支持量化后的 matmul_v2 在 TensorRT 上的推理。([#36594](https://github.com/PaddlePaddle/Paddle/pull/36594))

- 显存优化

  - 实现多 stream 安全 Allocator，支持在多 stream 异步计算场景下安全高效地使用显存。([#37290](https://github.com/PaddlePaddle/Paddle/pull/37290))

  - 新增运行时显存监控模块(paddle.device.cuda.max_memory_allocated, paddle.device.cuda.max_memory_reserved, paddle.device.cuda.memory_allocated and paddle.device.cuda.memory_reserved)，支持高性能地实时统计显存数据。([#38657](https://github.com/PaddlePaddle/Paddle/pull/38657))

  - 实现 CPU-GPU 统一内存寻址（CUDA Managed Memory），支持在显存受限场景下训练超大模型。([#39075](https://github.com/PaddlePaddle/Paddle/pull/39075))

  - C++底层新增 GetBasePtr 接口，用来获取设备接口 CUDAMalloc 创建的设备地址。([#37978](https://github.com/PaddlePaddle/Paddle/pull/37978))

  - 减少 AutoGrowth Allocator 中 free blocks 的数量，提升显存分配性能。([#35732](https://github.com/PaddlePaddle/Paddle/pull/35732))

  - 对于 `initializer.Normal` 和 `initializer.Constant` 数据类型是 FP16 的 Tensor 去除多余的 float32 临时 Tensor 以及 cast，节省 2 倍显存。([#38818](https://github.com/PaddlePaddle/Paddle/pull/38818))

- 动态图高阶导数组网测试

  - 为动态图增加三阶导数组网测试，以及 Broadcast 情况的测试。([#36814](https://github.com/PaddlePaddle/Paddle/pull/36814), [#37377](https://github.com/PaddlePaddle/Paddle/pull/37377))

- 自定义 op：支持 ROCm(HIP) 平台进行自定义 op 注册。([#36771](https://github.com/PaddlePaddle/Paddle/pull/36771))

- Cost Model：增加基于运行 Profile 的 Cost Model。([#35774](https://github.com/PaddlePaddle/Paddle/pull/35774))

- 提供定制化层 (nn.Layer)的自动稀疏训练支持，让用戶可根据自定义的 Prune 函数来对其设计的层进行稀疏剪枝。([#40253](https://github.com/PaddlePaddle/Paddle/pull/40253))

- 新增字符串张量底层数据结构表示，使框架具备字符串张量表示和计算的能力。([#39830](https://github.com/PaddlePaddle/Paddle/pull/39830), [#40992](https://github.com/PaddlePaddle/Paddle/pull/40992))

- 新增或者升级 oneDNN FP32/int8/bfloat16 Kernel，包括：

  - ELU ([#37149](https://github.com/PaddlePaddle/Paddle/pull/37149))

  - exp ([#38624](https://github.com/PaddlePaddle/Paddle/pull/38624))

  - stack ([#37002](https://github.com/PaddlePaddle/Paddle/pull/37002))

  - softplus ([#36382](https://github.com/PaddlePaddle/Paddle/pull/36382))

  - round ([#39653](https://github.com/PaddlePaddle/Paddle/pull/39653))

  - shape ([#36033](https://github.com/PaddlePaddle/Paddle/pull/36033))

  - flatten and flatten2 ([#35892](https://github.com/PaddlePaddle/Paddle/pull/35892))

  - slice ([#37630](https://github.com/PaddlePaddle/Paddle/pull/37630))

  - elementwise_mul ([#40546](https://github.com/PaddlePaddle/Paddle/pull/40546))

  - elementwise_add ([#38176](https://github.com/PaddlePaddle/Paddle/pull/38176))

  - ementwise_div ([#36158](https://github.com/PaddlePaddle/Paddle/pull/36158))

  - elementwise_sub ([#35662](https://github.com/PaddlePaddle/Paddle/pull/35662))

  - roi_align ([#37848](https://github.com/PaddlePaddle/Paddle/pull/37848))

  - nearest_interp and nearest_interp_v2 ([#37985](https://github.com/PaddlePaddle/Paddle/pull/37985)，[#38622](https://github.com/PaddlePaddle/Paddle/pull/38622)，[#39490](https://github.com/PaddlePaddle/Paddle/pull/39490))

  - assembly optimized Adam ([#39158](https://github.com/PaddlePaddle/Paddle/pull/39158))

  - logsoftmax ([#39793](https://github.com/PaddlePaddle/Paddle/pull/39793))

  - activation ([#40721](https://github.com/PaddlePaddle/Paddle/pull/40721))

  - mul ([#38552](https://github.com/PaddlePaddle/Paddle/pull/38552))

  - mean ([#37104](https://github.com/PaddlePaddle/Paddle/pull/37104))

  - relu ([#36265](https://github.com/PaddlePaddle/Paddle/pull/36265))

  - pool2d ([#37081](https://github.com/PaddlePaddle/Paddle/pull/37081))

  - concat ([#35889](https://github.com/PaddlePaddle/Paddle/pull/35889))

  - conv2d ([#38507](https://github.com/PaddlePaddle/Paddle/pull/38507)，[#38938](https://github.com/PaddlePaddle/Paddle/pull/38938)，[#36284](https://github.com/PaddlePaddle/Paddle/pull/36284))

  - LayerNorm ([#40418](https://github.com/PaddlePaddle/Paddle/pull/40418))

- 增加基于 SSD-内存-GPU 显存 的 3 级存储图检索引擎，支持大规模图神经网络训练。([#42472](https://github.com/PaddlePaddle/Paddle/pull/42472), [#42321](https://github.com/PaddlePaddle/Paddle/pull/42321), [#42027](https://github.com/PaddlePaddle/Paddle/pull/42027))

- 增加异构多云训练通信模块 switch，实现 Send/Recv 接口，支持多云异构通信。([#40965](https://github.com/PaddlePaddle/Paddle/pull/40965) [40911](https://github.com/PaddlePaddle/Paddle/pull/40911))

### （2）功能优化

#### API

- 为 `paddle.Model`新增支持混合精度训练 O2 模式，即支持原来动/静态图的 Pure FP16 训练模式。([#36441](https://github.com/PaddlePaddle/Paddle/pull/40962441))

- 为 `paddle.nn.Layer` 支持 self chain 调用。([#36609](https://github.com/PaddlePaddle/Paddle/pull/36609))

- 为 `paddle.nn.Layer`的`to`方法添加`is_distributed`属性的设置，保证网络参数转换前后分布式属性保持一致。([#36221](https://github.com/PaddlePaddle/Paddle/pull/36221))

- 完善 `paddle.nn.Layer`的`to` 方法的参数转换逻辑，降低转换过程占用的峰值显存，提高转换成功率。([#36862](https://github.com/PaddlePaddle/Paddle/pull/36862))

- 为 `paddle.incubate.graph_send_recv`支持设置输出 Tensor 的 shape，有利于减少实际计算过程的显存占用。([#40509](https://github.com/PaddlePaddle/Paddle/pull/40509))

- 为 `paddle.incubate.segment_sum`、`segment_mean`、`segment_max`、`segment_min` 新增 int32、int64 数据类型支持。([#40577](https://github.com/PaddlePaddle/Paddle/pull/40577))

- 为 transpose op 新增 bool 类型支持。([#35886](https://github.com/PaddlePaddle/Paddle/pull/35886))

- 将 `paddle.mm` 底层算子从 matmul 切换到 matmul_v2。([#35770](https://github.com/PaddlePaddle/Paddle/pull/35770))

- 为 `paddle.einsum` 支持静态图模式调用，支持未知 shape。([#40360](https://github.com/PaddlePaddle/Paddle/pull/40360))

- 为 `paddle.nn.functional.margin_cross_entropy` 和 `paddle.nn.functional.class_center_sample` 支持数据并行。([#39852](https://github.com/PaddlePaddle/Paddle/pull/39852))

- 为 `paddle.nn.functional.grid_sample`支持形状为[1]的输入。([#36183](https://github.com/PaddlePaddle/Paddle/pull/36183))

- 为 `paddle.nn.PRelu` 支持 `NHWC` 数据格式。([#37019](https://github.com/PaddlePaddle/Paddle/pull/37019))

- 为 `paddle.nn.functional.class_center_sample` 支持使用 `paddle.seed` 固定随机状态。([#38248](https://github.com/PaddlePaddle/Paddle/pull/38248))

- 为 `paddle.fft` 下所有 API 新增 ROCM 后端支持，并优化 CUFFT 后端报错信息。([#36415](https://github.com/PaddlePaddle/Paddle/pull/36415), [#36114](https://github.com/PaddlePaddle/Paddle/pull/36114/files))

- 为 `Tensor.getitem` 增加对切片部分维度为 0 的功能支持，即允许切片索引结果为空。([#37313](https://github.com/PaddlePaddle/Paddle/pull/37313))

- 为 `Tensor.setitem` 支持 int 和 bool 类型 Tensor 使用 bool 索引。([#37761](https://github.com/PaddlePaddle/Paddle/pull/37761))

- 为 `paddle.nn.functional.interpolate` 支持 nearest 模式时输入 shape 为 5D。([#38868](https://github.com/PaddlePaddle/Paddle/pull/38868))

- 为 `paddle.nn.Embedding`、`paddle.gather` 增加 int16 支持。([#40964](https://github.com/PaddlePaddle/Paddle/pull/40964), [#40052](https://github.com/PaddlePaddle/Paddle/pull/40052))

- 为 `paddle.distributed.spawn`添加 CPU 单机数据并行。([#35745](https://github.com/PaddlePaddle/Paddle/pull/35745), [#36758](https://github.com/PaddlePaddle/Paddle/pull/36758), [#36637](https://github.com/PaddlePaddle/Paddle/pull/36637))

- 新增`depthwise_conv2d`MKLDNN 算子。([#38484](https://github.com/PaddlePaddle/Paddle/pull/38484))

- 为`paddle.abs`、`paddle.transpose`、`paddle.squeeze`、`paddle.unsqueeze`、 `paddle.matmul`、`paddle.full` 静态图数据类型检测中增加复数类型。([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))

- 为 `paddle.autograd.PyLayer` 支持 tuple/list 类型的参数。([#38146](https://github.com/PaddlePaddle/Paddle/pull/38146))

- 为 `paddle.autograd.PyLayer` 增加检查 inplace 策略下，输入叶子节点的 Tensor 的检查报错机制。([#37931](https://github.com/PaddlePaddle/Paddle/pull/37931))

- 为 `paddle.autograd.PyLayer` 支持 HIP 库。([#38184](https://github.com/PaddlePaddle/Paddle/pull/38184))

- 为 `paddle.take_along_axis`、`paddle.put_along_axis` 支持更多 size 的输入，允许 index 矩阵的 shape size 大于 arr 矩阵的 shape size。([#39072](https://github.com/PaddlePaddle/Paddle/pull/39072))

- 优化 API `paddle.nn.Pad2D`在 replicate 为 0 时的报错信息。([#36510](https://github.com/PaddlePaddle/Paddle/pull/36510/files))

- 支持 API `paddle.nn.Pad2D`在 tuple 格式的 pad 输入。([#35985](https://github.com/PaddlePaddle/Paddle/pull/35985/files))

- 新增 `paddle.distributed.InMemoryDataset` 中 tdm_sample API 以支持 TDM 算法中的采样操作。([#37044](https://github.com/PaddlePaddle/Paddle/pull/37044))

- 新增对于`paddle.jit.save`的 Pre-saving Hooks 机制。([#38186](https://github.com/PaddlePaddle/Paddle/pull/38186))

- 新增高阶微分相关 API：

  - `elementwise_add` 增加三阶 Kernel，支持三阶微分的计算。([#36508](https://github.com/PaddlePaddle/Paddle/pull/36508), [#36618](https://github.com/PaddlePaddle/Paddle/pull/36618))

  - `matmul_v2` 增加三阶 Kernel，支持三阶微分的计算。([#36459](https://github.com/PaddlePaddle/Paddle/pull/36459))

  - `elementwise_mul` 增加三阶 Kernel，支持三阶微分的计算。([#37152](https://github.com/PaddlePaddle/Paddle/pull/37547))

- 完善`paddle.amp.GradScaler`调用 check_finite_and_unscale op 的逻辑，消除该处创建 bool 变量所引入的 cudaMemcpy。([#37770](https://github.com/PaddlePaddle/Paddle/pull/37770))

- 新增对 unstack 和 unique op 元素个数为 0 的 Tensor 增加检查。([#36021](https://github.com/PaddlePaddle/Paddle/pull/36021))

- 新增支持昆仑 2 的多层、双向 LSTM 功能，完善 RNN 前反向 op，支持时序类模型训练使用。([#](https://github.com/PaddlePaddle/Paddle/pull/41781)[42076](https://github.com/PaddlePaddle/Paddle/pull/42076))

- 新增支持昆仑 2 的 bce_loss 前反向 op。([#41610](https://github.com/PaddlePaddle/Paddle/pull/41610))

- 添加 `paddle.linalg.det` 的反向实现。([#36013](https://github.com/PaddlePaddle/Paddle/pull/36013))

#### IR(Intermediate Representation)

- 动态图转静态图

  - 优化动转静下 `ProgramCache.last` 接口行为，使其返回最近使用的 Program，而非最后生成的 Program。([#39541](https://github.com/PaddlePaddle/Paddle/pull/39541))

  - 优化动转静下 `paddle.reshape` API 的报错信息，新增推荐用法提示。([#40599](https://github.com/PaddlePaddle/Paddle/pull/40599))

  - 优化动转静代码转写时 `is_api_in_module` 函数中异常捕获类型。([#40243](https://github.com/PaddlePaddle/Paddle/pull/40243))

  - 优化动转静模块报错提示，默认隐藏 warning 信息。([#39730](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/39730))

  - 增加动转静对于 type hint 语法的支持，提高变量类型分析的准确性。([#39572](https://github.com/PaddlePaddle/Paddle/pull/39572))

  - 优化 `paddle.cond` 功能，允许 bool、int 等基本类型支持值相等。([#37888](https://github.com/PaddlePaddle/Paddle/pull/37888))

  - 优化动转静`@to_static` 装饰普通函数时，允许切换 train/eval 模式。([#37383](https://github.com/PaddlePaddle/Paddle/pull/37383))

  - 优化动转静报错栈，突出用户相关代码，减少框架冗余报错栈。([#36741](https://github.com/PaddlePaddle/Paddle/pull/36741))

  - 移除`paddle.cond` 返回值中 `no_value` 占位符。([#36513](https://github.com/PaddlePaddle/Paddle/pull/36513)、[#36826](https://github.com/PaddlePaddle/Paddle/pull/36826))

  - 为动转静 run_program op 适配新动态图模式。([#40198](https://github.com/PaddlePaddle/Paddle/pull/40198), [#40355](https://github.com/PaddlePaddle/Paddle/pull/40355))

  - 新增对于 zip 语法的检查。([#37846](https://github.com/PaddlePaddle/Paddle/pull/https://github.com/PaddlePaddle/Paddle/pull/37846))

  - 修复 `paddle.signal.frame`、`paddle.signal.stft`、`paddle.signal.istft` 因维度和类型判断错误导致的动转静失败问题。([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))

  - 为 mean、pad3d ops 新增注册复数类型 Kernel。([#40113](https://github.com/PaddlePaddle/Paddle/pull/40113))

#### 混合精度训练

- 为 amp 添加 GPU Compute Capability 环境检查，对无法产生训练加速效果的 GPU 环境添加使用警告。([#38086](https://github.com/PaddlePaddle/Paddle/pull/38086))

- 添加`paddle.amp.decorate`与`paddle.DataParallel`同时使用时调用顺序的检查。([#38785](https://github.com/PaddlePaddle/Paddle/pull/38785))

#### 分布式训练

- 分布式训练基础功能

  - 优化 Fleet API 和 DistributedStrategy 配置以使用动态图并行功能，提升动态图易用性。([#40408](https://github.com/PaddlePaddle/Paddle/pull/40408))

  - 优化动态图混合并行 HybridParallelClipGrad 策略，支持 4D 混合并行 + Pure FP16 训练。([#36237](https://github.com/PaddlePaddle/Paddle/pull/36237), [#36555](https://github.com/PaddlePaddle/Paddle/pull/36555))

  - 重构动态图数据并行策略，以支持新动态图和新通信库功能。([#40389](https://github.com/PaddlePaddle/Paddle/pull/40389), [#40593](https://github.com/PaddlePaddle/Paddle/pull/40593), [#40836](https://github.com/PaddlePaddle/Paddle/pull/40836), [#41119](https://github.com/PaddlePaddle/Paddle/pull/41119), [#41413](https://github.com/PaddlePaddle/Paddle/pull/41413), [#39987](https://github.com/PaddlePaddle/Paddle/pull/39987))

  - 为 fused_attention op 支持分布式张量模型并行。([#40101](https://github.com/PaddlePaddle/Paddle/pull/40101))

  - 为 fused_feedforward op 支持分布式张量模型并行。([#40160](https://github.com/PaddlePaddle/Paddle/pull/40160))

- 图检索引擎

  - 优化图引擎的图采样接口返回的数据格式，采样速度提升 3 倍。([#37315](https://github.com/PaddlePaddle/Paddle/pull/37315))

  - 减少图引擎线程量以提升性能。([#37098](https://github.com/PaddlePaddle/Paddle/pull/37098))

  - 优化图引擎数据传输以提升性能。([#37341](https://github.com/PaddlePaddle/Paddle/pull/37341))

  - 利用模型中 embedding op 的拓扑关系，优化 embedding op 的合并逻辑以提升性能。[(#35942)](https://github.com/PaddlePaddle/Paddle/pull/35942)

- 通信库：重构通信库，提升通信库的易扩展性和二次开发性，支持异构通信。([#41398](https://github.com/PaddlePaddle/Paddle/pull/41398), [#39720](https://github.com/PaddlePaddle/Paddle/pull/39720), [#40911](https://github.com/PaddlePaddle/Paddle/pull/40911), [#40579](https://github.com/PaddlePaddle/Paddle/pull/40579), [#40629](https://github.com/PaddlePaddle/Paddle/pull/40629), [#40437](https://github.com/PaddlePaddle/Paddle/pull/40437), [#40430](https://github.com/PaddlePaddle/Paddle/pull/40430), [#40228](https://github.com/PaddlePaddle/Paddle/pull/40228), [#40181](https://github.com/PaddlePaddle/Paddle/pull/40181), [#40100](https://github.com/PaddlePaddle/Paddle/pull/40100), [#40097](https://github.com/PaddlePaddle/Paddle/pull/40097), [#39892](https://github.com/PaddlePaddle/Paddle/pull/39892), [#39384](https://github.com/PaddlePaddle/Paddle/pull/39384), [#39737](https://github.com/PaddlePaddle/Paddle/pull/39737), [#40040](https://github.com/PaddlePaddle/Paddle/pull/40040))

- 支持 `paddle.incubate.distributed.models.moe`中 MoE 相关接口(`moe.GShardGate`, `moe.BaseGate`, `moe.SwitchGate`, `moe.MoELayer`, `moe.ClipGradForMOEByGlobalNorm` )的公开。([#42300](https://github.com/PaddlePaddle/Paddle/pull/42300))

- 修复 `paddle.incubate.distributed.models.moe.MoELayer` 中使用 recomputing 可能报错的问题。([#42128](https://github.com/PaddlePaddle/Paddle/pull/42128))

- 修复新动态图流水线并行因为数据类型不同导致的报错 ([#41937](https://github.com/PaddlePaddle/Paddle/pull/41937) [#42053](https://github.com/PaddlePaddle/Paddle/pull/42053))

- 修复新动态图张量模型并行因为数据类型不同导致的报错 ([#41960](https://github.com/PaddlePaddle/Paddle/pull/41960))

#### 自定义算子

- 增强 C++自定义算子机制对二阶反向算子编写功能，支持为二阶反向算子的梯度输入变量添加后缀作为输出使用。([#41781](https://github.com/PaddlePaddle/Paddle/pull/41781))

- 移除 Tensor API 成员方法中对废弃的枚举类型 PlaceType 的使用，进行相应兼容处理，并添加 deprecated warning 提示。([#41882](https://github.com/PaddlePaddle/Paddle/pull/41882))

- 为原 Tensor API 的一系列废弃接口，包括不完整构造函数、reshape、mutable_data、copy_to 方法添加 deprecated warning 提示。([#41882](https://github.com/PaddlePaddle/Paddle/pull/41882))

#### 其他

- 报错调试优化

  - 优化 cross_entropy op 对 `label` 的边界检查报错信息。([#40001](https://github.com/PaddlePaddle/Paddle/pull/40001))

  - 为动态图添加 op 执行时`infer_shape`和`compute`方法的 profile record，用于在 timeline 中展示其开销。([#39023](https://github.com/PaddlePaddle/Paddle/pull/39023))

  - 替换了 Windows 下容易出现未知异常的 `pybind::index_error` 报错提示。([#40538](https://github.com/PaddlePaddle/Paddle/pull/40538))

  - 添加用户 scatter op 越界检查的报错信息。([#37429](https://github.com/PaddlePaddle/Paddle/pull/37429))

- 下载工具：针对`paddle.utils.download.get_path_from_url`中解压含多文件目录速度慢的问题，将原先循环遍历目录下文件逐一解压的方式替换为在目录上调用 extractall 一次解压的方式，解压速度大幅提升。([#37311](https://github.com/PaddlePaddle/Paddle/pull/37311))

- 加速 `fake_quantize_range_abs_max`、`fake_quantize_abs_max`、`fake_quantize_dequantize_abs_max`、 `fake_quantize_moving_average_abs_max` 等量化训练。([#40491](https://github.com/PaddlePaddle/Paddle/pull/40491))

### （3）性能优化

#### 分布式训练

- 混合并行优化器 sharding 支持 optimize_cast 优化，将前反向参数 cast 移到优化器阶段，性能提升 7%。([#35878](https://github.com/PaddlePaddle/Paddle/pull/35878))

- GPUPS 优化：支持梯度 fuse allreduce 训练，训练提升 20%。([#35131](https://github.com/PaddlePaddle/Paddle/pull/35131))

- GPUPS 优化：dump CPU 优化提速 3.21 倍。([#40068](https://github.com/PaddlePaddle/Paddle/pull/40068))

- CPU 参数服务器流式训练优化：支持稀疏参数统计量自动统计、稀疏参数增量保存等功能，训练性能提升 20%。([#36465](https://github.com/PaddlePaddle/Paddle/pull/36465), [#36601](https://github.com/PaddlePaddle/Paddle/pull/36601), [#36734](https://github.com/PaddlePaddle/Paddle/pull/36734), [#36909](https://github.com/PaddlePaddle/Paddle/pull/36909), [#36943](https://github.com/PaddlePaddle/Paddle/pull/36943), [#37181](https://github.com/PaddlePaddle/Paddle/pull/37181), [#37194](https://github.com/PaddlePaddle/Paddle/pull/37194), [#37515](https://github.com/PaddlePaddle/Paddle/pull/37515), [#37626](https://github.com/PaddlePaddle/Paddle/pull/37626), [#37995](https://github.com/PaddlePaddle/Paddle/pull/37995), [#38582](https://github.com/PaddlePaddle/Paddle/pull/38582), [#39250](https://github.com/PaddlePaddle/Paddle/pull/39250), [#40762](https://github.com/PaddlePaddle/Paddle/pull/40762), [#41234](https://github.com/PaddlePaddle/Paddle/pull/41234), [#41320](https://github.com/PaddlePaddle/Paddle/pull/41320), [#41400](https://github.com/PaddlePaddle/Paddle/pull/41400))

#### 算子优化

- 优化 `FasterTokenizer` 性能，性能与优化前相比提升 10%。([#36701](https://github.com/PaddlePaddle/Paddle/pull/36701))

- 优化 `index_select` 反向计算，性能较优化前有 3.7~25.2 倍提升。([#37055](https://github.com/PaddlePaddle/Paddle/pull/37055))

- 优化 `paddle.nn.ClipByGlobalNorm` 的性能，以 10*10 的 `paddle.nn.Linear` 为例，性能与优化前相比提升 30%左右。([#38209](https://github.com/PaddlePaddle/Paddle/pull/38209))

- 优化 `pnorm` 在 `axis` 维度极大或极小情况下的性能，前向速度提升 31~96 倍，反向速度提升 1.1~19 倍。([#37685](https://github.com/PaddlePaddle/Paddle/pull/37685), [#38215](https://github.com/PaddlePaddle/Paddle/pull/38215), [#39011](https://github.com/PaddlePaddle/Paddle/pull/39011))

- 优化 `softmax` 前、反向性能，对于 `axis!=-1` 的配置加速比为 2 倍左右。([#38602](https://github.com/PaddlePaddle/Paddle/pull/38602), [#38609](https://github.com/PaddlePaddle/Paddle/pull/38609), [#32387](https://github.com/PaddlePaddle/Paddle/pull/32387), [#37927](https://github.com/PaddlePaddle/Paddle/pull/37927/files))

- 优化 `log_softmax` 前、反向性能，对于 `axis!=-1`的配置加速比为 6~20 倍左右。([#38992](https://github.com/PaddlePaddle/Paddle/pull/38992), [#40612](https://github.com/PaddlePaddle/Paddle/pull/40612))

- 优化 `softmax_with_cross_entropy` 前、反向性能，对于 `hard_label` 的配置加速比为 1.3 倍左右。([#39553](https://github.com/PaddlePaddle/Paddle/pull/39553), [#40424](https://github.com/PaddlePaddle/Paddle/pull/40424), [#40643](https://github.com/PaddlePaddle/Paddle/pull/40643))

- 优化 `top_k` 性能，对于一维且 `k` 较大时(k=5000)的配置加速比为 22 倍以上。([#40941](https://github.com/PaddlePaddle/Paddle/pull/40941))

- 优化 `elementwise_mul` 反向计算，较优化前有 1.85~12.16 倍性能提升。([#37728](https://github.com/PaddlePaddle/Paddle/pull/37728))

- 优化 `elementwise_min` 反向和 `elementwise_max` 反向，较优化前打平或有 1.05~18.75 倍性能提升。([#38236](https://github.com/PaddlePaddle/Paddle/pull/38236), [#37906](https://github.com/PaddlePaddle/Paddle/pull/37906))

- 优化 `nearest_interp` 前向和反向计算，前向较优化前性能有 1.5~2.3 倍提升；反向性能较优化前有 60%~1.8 倍提升。([#38528](https://github.com/PaddlePaddle/Paddle/pull/38528), [#39067](https://github.com/PaddlePaddle/Paddle/pull/39067))

- 优化 `bilinear_interp` 前向和反向计算，前向较优化前性能有 0.4~2.3 倍提升；反向性能较优化前有 10%~30%提升。([#39243](https://github.com/PaddlePaddle/Paddle/pull/39243), [#39423](https://github.com/PaddlePaddle/Paddle/pull/39423))

- 优化 `dropout` 前向和反向计算，性能提升约 20%。([#39795](https://github.com/PaddlePaddle/Paddle/pull/39795), [#38859](https://github.com/PaddlePaddle/Paddle/pull/38859), [#38279](https://github.com/PaddlePaddle/Paddle/pull/38279), [#40053](https://github.com/PaddlePaddle/Paddle/pull/40053))

- 优化 `grid_sampler`前向和反向计算，前向较优化前性能有 10%~30%提升；反向性能较优化前有 10%~60%提升。([#39751](https://github.com/PaddlePaddle/Paddle/pull/39751))

- 优化 `group_norm` 前向和反向计算，前向性能提升 1.04~2.35 倍，反向性能提升 1.12~1.18 倍。([#39944](https://github.com/PaddlePaddle/Paddle/pull/39944), [#40657](https://github.com/PaddlePaddle/Paddle/pull/40657), [#39596](https://github.com/PaddlePaddle/Paddle/pull/39596))

- 优化 `conv1d` 前向和反向计算，前向性能提升 1.00~2.01 倍，反向性能提升 1.01~474.56 倍。([#38425](https://github.com/PaddlePaddle/Paddle/pull/38425))

- 优化 `elementwise_div` 反向计算，反向性能提升 1.02~29.25 倍。([#38044](https://github.com/PaddlePaddle/Paddle/pull/38044))

- 优化 `gelu` 前向和反向计算，前向性能提升 1.13~1.43 倍，反向性能提升 1.10～1.55 倍。([#38188](https://github.com/PaddlePaddle/Paddle/pull/38188), [#38263](https://github.com/PaddlePaddle/Paddle/pull/38263))

- 优化 `elementwise_sub` 反向计算，反向性能提升 1.04~15.64 倍。([#37754](https://github.com/PaddlePaddle/Paddle/pull/37754))

- 优化 `flip` 在输入一维数据时前向性能，性能提升 100%。([#37825](https://github.com/PaddlePaddle/Paddle/pull/37825))

- 优化 `layer_norm` 前向和反向计算，前向较优化前提升 2-5 倍，反向较优化前提升 20%~50%。([#39167](https://github.com/PaddlePaddle/Paddle/pull/39167), [#39247](https://github.com/PaddlePaddle/Paddle/pull/39247))

- 优化 `embedding` 前向和反向计算，前向较优化前最大提升 1.51 倍，反向较优化前提升 1.03~7.79 倍。([#39856](https://github.com/PaddlePaddle/Paddle/pull/39856), [#39886](https://github.com/PaddlePaddle/Paddle/pull/398866))

- 优化 `gelu` FP16 前向和反向计算，前向较优化前提升 9%~12%，反向较优化前提升 2%~9%。([#38980](https://github.com/PaddlePaddle/Paddle/pull/38980))

- 移除 `gather_nd`前反向算子中的 CPU -> GPU 显式数据传输操作，移除 `index_select` 前反向算子中的显式同步操作，将 `scatter_nd` 中的 GPU -> GPU 数据传输由同步操作改成异步操作。([#40933](https://github.com/PaddlePaddle/Paddle/pull/40933))

- 优化 `Lars optimzier` 计算，优化后 Resnet50 PF16 模型训练性能较优化前提升 5.1%。([#35652](https://github.com/PaddlePaddle/Paddle/pull/35652), [#35476](https://github.com/PaddlePaddle/Paddle/pull/35476))

- 优化 `AvgPool2dGrad` 计算，优化后性能较优化前提升 2.6 倍。([#35389](https://github.com/PaddlePaddle/Paddle/pull/35389))

- 优化 `Elementwise` 类计算对于多元输出的功能支持，优化后计算性能较优化前提升最多可达 15%。([#38329](https://github.com/PaddlePaddle/Paddle/pull/38329), [#38410](https://github.com/PaddlePaddle/Paddle/pull/38410))

- 优化 `Categorical`的 `probs`计算，简化计算逻辑，性能提升 4 ~ 5 倍。([#42178](https://github.com/PaddlePaddle/Paddle/pull/42178))

- `paddle.sum` 性能优化，性能相比优化前提升约 20%。([#42309](https://github.com/PaddlePaddle/Paddle/pull/42309))

#### 自动调优

新增训练全流程硬件感知性能自动调优功能，在图像分类、分割、检测和图像生成任务上与模型默认参数配置下的性能相比提升约 3%～50%以上。通过 `paddle.incubate.autotune.set_config` API 设置自动调优状态，当前默认关闭。自动调优具体包括三个层次：

- `paddle.io.DataLoader` 新增自动调优功能，根据训练数据和设备资源选择最佳的模型 num_workers。([#42004](https://github.com/PaddlePaddle/Paddle/pull/42004))

- 新增混合精度训练数据布局自动调优功能，根据设备类型和数据类型选择最佳数据布局，并在运行时自动转换。([#41964](https://github.com/PaddlePaddle/Paddle/pull/41964))

- 新增 Conv 运行时所需 workspace size 阈值自动调整功能，根据 GPU 当前可申请显存资源情况来自动设置；基于通用的 AlgorithmCache 设计和 Kernel 计时组件，新增 Conv cuDNN 算法自动选择功能，支持数据变长模型。([#41833](https://github.com/PaddlePaddle/Paddle/pull/41833))

#### 调度优化

- 移除 `paddle.nn.ClipGradByGlobalNorm` 中的 CudaStreamSync 隐藏操作，减少执行时的调度开销，在 ptb 模型上有 5%的性能提升。([#42170](https://github.com/PaddlePaddle/Paddle/pull/42170))

- 优化一系列底层数据结构及原动态图执行体系中的细节实现，提升原动态图的调度性能。([#42010](https://github.com/PaddlePaddle/Paddle/pull/42010), [#42171](https://github.com/PaddlePaddle/Paddle/pull/42171), [#42224](https://github.com/PaddlePaddle/Paddle/pull/42224), [#42256](https://github.com/PaddlePaddle/Paddle/pull/42256), [#42306](https://github.com/PaddlePaddle/Paddle/pull/42306), [#42329](https://github.com/PaddlePaddle/Paddle/pull/42329)[, #42340](https://github.com/PaddlePaddle/Paddle/pull/42340), [#42368](https://github.com/PaddlePaddle/Paddle/pull/42368), [#42425](https://github.com/PaddlePaddle/Paddle/pull/42425))

- 简化 `paddle.distribution.Categorical`的 probs 计算逻辑，提升性能 4 到 5 倍。([#42178](https://github.com/PaddlePaddle/Paddle/pull/42178))

### （4）问题修复

#### API

- 修复 `paddle.sum` 输入参数类型和输出参数类型不一致且 `axis` 轴对应的 reduce 元素个数为 1 时，输出类型错误问题。([#36123](https://github.com/PaddlePaddle/Paddle/pull/36123))

- 修复 `paddle.flops` 在 layer 输出类型为 tuple 时的 `AttributeError`。([#38850](https://github.com/PaddlePaddle/Paddle/pull/38850))

- 修复 `paddle.diag` 因为没有反向 Kernel 而无法传播梯度的问题。([#40447](https://github.com/PaddlePaddle/Paddle/pull/40447))

- 修复 `paddle.sort` 输入存在 NaN 值排序错误。([#41070](https://github.com/PaddlePaddle/Paddle/pull/41070))

- 修复 `paddle.full_like` 输入存在 Inf 值构建 Tensor 错误。([#40232](https://github.com/PaddlePaddle/Paddle/pull/40232))

- 修复 `paddle.strided_slice` 在输入 starts 中数据小于 -rank 时，strided_slice 结果与 slice 不一致的 bug。([#39066](https://github.com/PaddlePaddle/Paddle/pull/39066))

- 修复 `max_pool` 系列算子在返回 index 时 infer_shape 计算错误的问题，受影响的 API 有 `paddle.nn.functional.max_pool1d/2d/3d`, `paddle.nn.functional.adaptive_max_pool1d/2d/3d`, `paddle.nn.MaxPool1D/2D/3D`, `paddle.nn.AdaptiveMaxPool1D/2D/3D`。([#40139](https://github.com/PaddlePaddle/Paddle/pull/40139))

- 修复 `max_pool` 系列算子返回的 pooling_mask 的 dtype 错误的问题，现在 pooling_mask 的 dtype 为 int32，受影响的 API 有 `paddle.nn.functional.max_pool1d/2d/3d`, `paddle.nn.functional.adaptive_max_pool1d/2d/3d`, `paddle.nn.MaxPool1D/2D/3D`, `paddle.nn.AdaptiveMaxPool1D/2D/3D`。([#39314](https://github.com/PaddlePaddle/Paddle/pull/39314))

- 修复 `paddle.shape` 默认存在反向梯度导致计算错误的问题。([#37340](https://github.com/PaddlePaddle/Paddle/pull/37340))

- 修复 `paddle.nn.Layer` 的 `to` 方法同时转换 dtype 和 place 存在的 bug。([#37007](https://github.com/PaddlePaddle/Paddle/pull/38007))

- 修复 `paddle.amp.decorate` 无法对非叶子网络层的参数改写为 FP16 的 bug。([#38402](https://github.com/PaddlePaddle/Paddle/pull/38402))

- 修复 `paddle.amp.decorate` 将 `paddle.nn.BatchNorm1D`、`paddle.nn.BatchNorm2D`、`paddle.nn.BatchNorm3D` 非输入参数改写为 FP16 的 bug。([#38541](https://github.com/PaddlePaddle/Paddle/pull/38541))

- 修复 `paddle.amp.decorate` 将 `paddle.nn.SyncBatchNorm` 非输入参数改写为 FP16 的 bug。([#40943](https://github.com/PaddlePaddle/Paddle/pull/40943))

- 修复 `paddle.nn.Layer.to` 当中多余的 warning。([#36700](https://github.com/PaddlePaddle/Paddle/pull/36700))

- 修复 `paddle.nn.RNN` 在控制流下使用报错的问题。([#41162](https://github.com/PaddlePaddle/Paddle/pull/41162))

- 修复 `paddle.to_tensor` 无法指定 Tensor 的 CUDA Place 的问题。([#39662](https://github.com/PaddlePaddle/Paddle/pull/39662))

- 修复 `paddle.nn.Identity` 没有公开的问题。([#39615](https://github.com/PaddlePaddle/Paddle/pull/39615))

- 修复动态图重构后，`fill_` 和 `zero_` inplace API 的输入在 CUDAPinned Place 上时，输出值不正确的 bug。([#41229](https://github.com/PaddlePaddle/Paddle/pull/41229))

- 动态图重构后，修复使用 append op 的方式调用 assign op 导致输出 Tensor 的 inplace version 值不正确的 bug，修改为使用 `_C_ops` 的方式调用 assign op。([#41118](https://github.com/PaddlePaddle/Paddle/pull/41118))

- 移除 `elementwise_add` 三阶 Kernel 中不合理的代码，修复组网过程未初始化问题。([#36618](https://github.com/PaddlePaddle/Paddle/pull/36618))

- 修复 `conv2d` 执行 cuDNN Kernel 时属性缺失的问题。([#38827](https://github.com/PaddlePaddle/Paddle/pull/38827))

- 修复 `multiclass_nms3` 输出 shape 不正确的问题。([#40059](https://github.com/PaddlePaddle/Paddle/pull/40059))

- 修复 `yolo_box` 输出 shape 不正确的问题。([#40056](https://github.com/PaddlePaddle/Paddle/pull/40056))

- 修复高阶微分 `gradients` 接口在指定 target_grad 时未按预期生效的问题。([#40940](https://github.com/PaddlePaddle/Paddle/pull/40940/))

- 修复动态图 op`_BatchNormBase` 基类中修改了 default_dtype，导致后续组网参数类型错误的问题，受影响的 API 有 `paddle.nn.BatchNorm1D`，`paddle.nn.BatchNorm2D`，`paddle.nn.BatchNorm3D`，`paddle.nn.SyncBatchNorm`。具体原因是当 `get_default_dtype() == 'float16'` 时，通过 `set_default_dtype('float32')`修改默认参数数据类型，动态图组网的参数类型是通过 default_dtype 来创建的，因此当默认参数类型被修改后导致后续的组网参数类型错误。([#36376](https://github.com/PaddlePaddle/Paddle/pull/36376))

- 修复 batchnorm op 中，当数据类型为 FP32，且数据维度 `dims = 2，data_layout = NHWC` 时，反向 op 内中间变量未定义问题。([#37020](https://github.com/PaddlePaddle/Paddle/pull/37020))

- 修复静态图模式下，`paddle.static.nn.prelu` 对于 `NHWC` 输入格式且 `mode==channel` 权重的 shape 错误问题。([#38310](https://github.com/PaddlePaddle/Paddle/pull/38310))

- 修复多机情况下，`paddle.nn.functional.class_center_sample` CUDA 种子设置 bug。([#38815](https://github.com/PaddlePaddle/Paddle/pull/38815))

- 修复 `paddle.nn.functional.one_hot` 在输入不正确参数时，CUDA 版本无法正确报错的问题。([#41335](https://github.com/PaddlePaddle/Paddle/pull/41335))

- 修复 DCU 设备上回收显存的 callback 未及时触发导致显存 OOM 的问题。([#40445](https://github.com/PaddlePaddle/Paddle/pull/40445))

- 修复 `setitem` 索引赋值反向梯度传递异常以及动态图部分场景下 inplace 逻辑处理异常的问题。([#37023](https://github.com/PaddlePaddle/Paddle/pull/37023), [#38298](https://github.com/PaddlePaddle/Paddle/pull/38298))

- 修复动转静下 Tensor array 使用 Slice 索引异常的问题。([#39251](https://github.com/PaddlePaddle/Paddle/pull/39251))

- 修复 `paddle.Tensor.register_hook` 接口使用时临时变量未析构，从而导致内存或显存泄漏的问题。([#40716](https://github.com/PaddlePaddle/Paddle/pull/40716))

- 修复 `Tensor.getitem` 当索引是全为 False 的 bool Tensor 时无法取值的问题。([#41297](https://github.com/PaddlePaddle/Paddle/pull/41297))

- 修复 `Tensor.getitem` 当索引是 bool scalar Tensor 时无法取值的问题。([#40829](https://github.com/PaddlePaddle/Paddle/pull/40829))

- 修复 `paddle.index_select` 在 index 为 0-shape Tensor 时报错的问题。([#41383](https://github.com/PaddlePaddle/Paddle/pull/41383))

- 修复 `paddle.index_select`，`paddle.index_sample` 申请的 GPU 线程数超过有限机器资源时报错的问题。([#41127](https://github.com/PaddlePaddle/Paddle/pull/41127), [#37816](https://github.com/PaddlePaddle/Paddle/pull/37816), [#39736](https://github.com/PaddlePaddle/Paddle/pull/39736), [#41563](https://github.com/PaddlePaddle/Paddle/pull/41563))

- 修复 ReduceConfig、elemwise_grad、gather、gather_nd、scatter ops 申请 GPU 线程数超过有限机器资源时报错的问题。([#40813](https://github.com/PaddlePaddle/Paddle/pull/40813), [#41127](https://github.com/PaddlePaddle/Paddle/pull/41127))

- 修复 Kernel Primitive API 中 ReadData，ReadDataBc，ReadDataReduce 在 NX != 1 时访存越界的问题。([#36373](https://github.com/PaddlePaddle/Paddle/pull/36373))

- 修复 IndexRandom 数据类型错误导致数据溢出计算结果异常的问题。([#39867](https://github.com/PaddlePaddle/Paddle/pull/39867), [#39891](https://github.com/PaddlePaddle/Paddle/pull/39891))

- 修复 reduce op 在 reduce_num = 1 计算结果返回错误的问题。([#38771](https://github.com/PaddlePaddle/Paddle/pull/38771))

- 修复 reduce op 在 HIP 环境下 reduce 中间维度出现访存越界的问题。([#41273](https://github.com/PaddlePaddle/Paddle/pull/41273))

- 修复 matmul op 两个 FP16 一维向量计算时 Kernel 无法正常释放的问题。

- 修复部分算子在 CUDA 上因整型计算溢出导致的问题，包括：bernoulli、gaussian_random、gumbel_softmax、multinomial、truncated_gaussian_random、uniform_random_inplace、uniform_random ops。([#37670](https://github.com/PaddlePaddle/Paddle/pull/37670))

- 修复 `paddle.nn.Sequential` 在 for 循环遍历 sublayers 时会报 KeyError 错误的 bug。([#39372](https://github.com/PaddlePaddle/Paddle/pull/39372))

- 修复 `paddle.nn.functional.unfold` 在静态图下编译时检查 shape 错误的 bug。([#38907](https://github.com/PaddlePaddle/Paddle/pull/38907), [#38819](https://github.com/PaddlePaddle/Paddle/pull/38819))

- 修复静态图使用 dropout 时如果指定了 `axis` 后会报错的问题。([#37223](https://github.com/PaddlePaddle/Paddle/pull/37223))

- 迁移 `paddle.nn.MultiHeadAttention`中 matmul 算子到 matmul_v2 算子。([#36222](https://github.com/PaddlePaddle/Paddle/pull/36222))

- 修复 `paddle.nn.functional.label_smooth`在输入为空 Tensor 时抛出 FPE 的问题。([#35861](https://github.com/PaddlePaddle/Paddle/pull/35861))

- 修复 reshape op 空 Tensor 形变问题， 支持将空 Tensor rehape 成[-1]。([#36087](https://github.com/PaddlePaddle/Paddle/pull/36087))

- 修复 `fill_diagonal`参数 offset 非零时会造成修改值跨行问题。([#36212](https://github.com/PaddlePaddle/Paddle/pull/36212))

- 修改动态图模式下 range op 返回 stop gradient 设置成 True。([#37486](https://github.com/PaddlePaddle/Paddle/pull/37486))

- 修复 Lamb 优化器当 Beta1Pow 和 Beta2Pow 在 GPU 上时更新错误的 bug。([#38518](https://github.com/PaddlePaddle/Paddle/pull/38518))

- 修复 conv2d 算子 FLAGS_cudnn_deterministic 设置不生效的问题。([#37173](https://github.com/PaddlePaddle/Paddle/pull/37173))

- 修复因早期版本的 cufft 没有定义 CUFFT_VERSION 引发的问题。([#37312](https://github.com/PaddlePaddle/Paddle/pull/37312))

- 修复 `paddle.ifftshit`, `paddle.fftshift` 计算错误问题。([#36834](https://github.com/PaddlePaddle/Paddle/pull/36834), [#36748](https://github.com/PaddlePaddle/Paddle/pull/36748))

- 修复 `paddle.fft` 系列 API 中的 `axis` 计算错误。([#36321](https://github.com/PaddlePaddle/Paddle/pull/36321))

- 修复 batch_norm_grad op 在 FP16 数据类型时输出数据类型注册的 bug，该 bug 会导致部分场景下编译失败，并且对 FP16 计算精度会有一定影响。([#42461](https://github.com/PaddlePaddle/Paddle/pull/42461))

- 修复 `paddle.nn.functional.pad` API 在模型动转静时，padding 为 Tensor 条件下的 Infershape 信息错误问题。([#42414](https://github.com/PaddlePaddle/Paddle/pull/42414))

- 修复 `paddle.distribution.StickBreakingTransform` 输入维度超过 2 时异常的问题。([#41762](https://github.com/PaddlePaddle/Paddle/pull/41672))

- 修复 fused_attention op 中 QK^T 计算出 nan/inf 的问题。([#42032](https://github.com/PaddlePaddle/Paddle/pull/42032))

- 修复 fused_attention op 中 FusedResidualDropoutBias 在 V100 上计算出 nan/inf 问题。([#42398](https://github.com/PaddlePaddle/Paddle/pull/42398))

- 修复 full_like op 在执行时引入的多余的 data transform 问题。([#41973](https://github.com/PaddlePaddle/Paddle/pull/41973))

- 修复 p_norm op 在 GPU 环境上计算 nan 的问题。([#41804](https://github.com/PaddlePaddle/Paddle/pull/41804))

- 修复 split op 在参数 sections 存在为 0 的 size 情况下，段错误的问题。([#41755](https://github.com/PaddlePaddle/Paddle/pull/41755))

- 修复 6 个 elementwise op（pow、complex、divide_double、multiply_double、fmax、fmin）在需要 broadcast 的情况下，多卡训练时报 Place(gpu:0) 不支持的问题。([#42332](https://github.com/PaddlePaddle/Paddle/pull/42332))

- 修复 import paddle 时由于 PIL 版本升级导致的废弃接口报 warning 的问题。([#42307](https://github.com/PaddlePaddle/Paddle/pull/42307))

- 修复静态图下 `paddle.linalg.matrix_rank`不支持 tol 为 FP64 Tensor 的问题。([#42085](https://github.com/PaddlePaddle/Paddle/pull/42085))

#### IR(Intermediate Representation)

- 动态图转静态图

  - 修复 `tensor_array` 搭配控制流使用时，在反向梯度累加时存在的类型推导错误问题。([#39585](https://github.com/PaddlePaddle/Paddle/pull/39585), [#39689](https://github.com/PaddlePaddle/Paddle/pull/39689))

  - 修复动转静 AMP 训练时参数梯度类型未被正确设置的问题。([#40938](https://github.com/PaddlePaddle/Paddle/pull/40938))

  - 修复代码中存在错位注释时，动转静代码解析报错的问题。([#39035](https://github.com/PaddlePaddle/Paddle/pull/39035), [#38003](https://github.com/PaddlePaddle/Paddle/pull/38003))

  - 修复动转静代码中调用非 forward 函数时，Tensor 未被正确转化为 Variable 的问题。([#37296](https://github.com/PaddlePaddle/Paddle/pull/37296), [#38540](https://github.com/PaddlePaddle/Paddle/pull/38540))

  - 修复动转静代码转写时 `paddle` 被错误地作为变量传递的问题。([#37999](https://github.com/PaddlePaddle/Paddle/pull/37999))

  - 修复模型动转静后调用 `paddle.flops` 时模型参数统计错误的问题。([#36852](https://github.com/PaddlePaddle/Paddle/pull/36852))

  - 修复使用 `paddle.jit.save/load` 接口加载模型后，在 train 模式和 no_grad 上下文中，显存会一直增长的问题。([#36434](https://github.com/PaddlePaddle/Paddle/pull/36434))

  - 添加在 convert_call 对 generator function 转换时的警告。([#35369](https://github.com/PaddlePaddle/Paddle/pull/35369))

  - 修复 run_program op 依赖分析的问题。([#38470](https://github.com/PaddlePaddle/Paddle/pull/38470))

  - 修复控制流 For 中返回单值时代码转换的问题。([#40683](https://github.com/PaddlePaddle/Paddle/pull/40683))

  - 修复控制流 cond 的输入包含 LoDTensorArray 时，生成反向 op 会报错的问题。([#39585](https://github.com/PaddlePaddle/Paddle/pull/39585))

  - 修复 `padddle.jit.save`在导出动转静模型时丢失顶层 Layer 的 forward_pre_hook 和 forward_post_hook 的问题。([#42273](https://github.com/PaddlePaddle/Paddle/pull/42273))

  - 修复 `paddle.expand`中 shape 参数包含 Tensor 在动转静时会转换报错的问题。([#41973](https://github.com/PaddlePaddle/Paddle/pull/41973))

#### 分布式训练

- 分布式训练基础功能

  - 修复分布式多机训练时，端口报错的问题。([#37274](https://github.com/PaddlePaddle/Paddle/pull/37274))

  - 修复 brpc 编译依赖问题。([#37064](https://github.com/PaddlePaddle/Paddle/pull/37064))

  - 修复 Fleet 启动时，由于 tcp 自连接产生的端口被占用的问题。([#38174](https://github.com/PaddlePaddle/Paddle/pull/38174))

  - 修复数据并行下，由于 FP16 参数在多卡下初始化不一致，导致精度下降的问题。([#38838](https://github.com/PaddlePaddle/Paddle/pull/38838), [#38563](https://github.com/PaddlePaddle/Paddle/pull/38563), [#38405](https://github.com/PaddlePaddle/Paddle/pull/38405))

  - 修复数据并行下，由于 FP16 梯度同步时，没有除以卡数，导致精度下降的问题。([#38378](https://github.com/PaddlePaddle/Paddle/pull/38378))

- 动态图混合并行

  - 修复在混合并行下，通过使用新 update 接口，FP16 模式不更新参数的问题。([#36017](https://github.com/PaddlePaddle/Paddle/pull/36017))

- 静态图混合并行

  - 修复分布式 dp 模式下 grad merge 与 ClipGradientByGlobalNorm 不兼容的问题。([#36334](https://github.com/PaddlePaddle/Paddle/pull/36334))

  - 修复混合并行下，张量模型并行的非分布式参数在初始化阶段未被广播，导致各卡非分布式参数不一致的问题。([#36186](https://github.com/PaddlePaddle/Paddle/pull/36186))

  - 修复 sharding 开启 offload 时，sharding 的 save_persistables 接口未保存 FP16 参数和 offload 持久化变量的问题。([#40477](https://github.com/PaddlePaddle/Paddle/pull/40477))

  - 修复开启 sharding 训练时，ema 参数在非 0 号卡上无法保存的问题。([#39860](https://github.com/PaddlePaddle/Paddle/pull/39860))

  - 修复 FC 按照列切分梯度计算错误的问题。([#38724](https://github.com/PaddlePaddle/Paddle/pull/38724))

  - 修复 DistributedStrategy 设置为 without_graph_optimizer 时和 rnn 一起使用报错的问题。([#36176](https://github.com/PaddlePaddle/Paddle/pull/36176))

- GPUPS 参数服务器训练

  - 修复 GPUPS 宏定义触发 CPU 分支编译问题。([#37248](https://github.com/PaddlePaddle/Paddle/pull/37248))

  - 修复 GPUPS 流水线训练时在保存 delta 和 pullsparse 并发时引发的偶发报错问题。([#37233](https://github.com/PaddlePaddle/Paddle/pull/37233))

  - 修复 HDFSClient 查询目录未返回全路径，引发下载报错问题。([#36590](https://github.com/PaddlePaddle/Paddle/pull/36590))

  - 修复 GPUPS 流水线训练时拉取老参数问题。([#36512](https://github.com/PaddlePaddle/Paddle/pull/36512))

  - 修复 GPUPS 多流 allocation 问题。([#37476](https://github.com/PaddlePaddle/Paddle/pull/37476))

  - 修复 GPUPS pybind 出 core 的问题。([#37287](https://github.com/PaddlePaddle/Paddle/pull/37287))

#### 其他

- 修复动态图量化训练保存模型时 clip_extra 的问题。([#38323](https://github.com/PaddlePaddle/Paddle/pull/38323))

- 修复动态图量化训练 abs_max scale 初始化的问题。([#39307](https://github.com/PaddlePaddle/Paddle/pull/39307))

- 修复动态图量化训练保存模型节点异常的问题。([#38102](https://github.com/PaddlePaddle/Paddle/pull/38102), [#38012](https://github.com/PaddlePaddle/Paddle/pull/38012))

- 修复离线量化 flatten op 输出错误问题。([#37722](https://github.com/PaddlePaddle/Paddle/pull/37722))

- 修复了反量化 matmul op 时，维度对不上的问题。([#36982](https://github.com/PaddlePaddle/Paddle/pull/36982))

- 修复了量化无权重的 matmul_v2 时，错误添加量化 op 的问题。([#36593](https://github.com/PaddlePaddle/Paddle/pull/36593))

- 修复 conv op channel wise 量化在保存模型时 quant_axis 属性保存错误。([#39054](https://github.com/PaddlePaddle/Paddle/pull/39054))

- 修复 ChannelWise 量化训练速度慢的问题。([#40772](https://github.com/PaddlePaddle/Paddle/pull/40772))

- 修复量化训练初始化为 0 的 Tensor 出 NAN 的问题。([#36762](https://github.com/PaddlePaddle/Paddle/pull/36762))

- 修复多线程场景下混合精度 amp_level 设置错误问题。([#39198](https://github.com/PaddlePaddle/Paddle/pull/39198))

- 修复混合精度训练与 PyLayer，Recompute 等一起使用时，PyLayer 和 Recompute 中未正确设置混合精度的问题。([#39950](https://github.com/PaddlePaddle/Paddle/pull/39950), [#40042](https://github.com/PaddlePaddle/Paddle/pull/40042))

- 修复了 Mac 下编译自定义算子时 `D_GLIBCXX_USE_CXX11_ABI` 未生效的问题。([#37878](https://github.com/PaddlePaddle/Paddle/pull/37878))

- 修复 initializer 相关 API 在 block=None 时动静行为不统一的问题。([#37827](https://github.com/PaddlePaddle/Paddle/pull/37827))

- 修复 python3.6 环境下没有 fluid 模块的 bug。([#35862](https://github.com/PaddlePaddle/Paddle/pull/35862))

- 修复优化器 `paddle.optimizer.Adamw` 错误调用 adam op 的 bug。([#36028](https://github.com/PaddlePaddle/Paddle/pull/36028))

- 修复 multi tensor 策略下 `paddle.optimizer.Momentum` 优化器参数 `regularizer` 属性为 None 时的逻辑错误。([#38344](https://github.com/PaddlePaddle/Paddle/pull/38344))

- 修复 multi tensor 策略下 `paddle.optimizer.Momentum`、`paddle.optimizer.Adam` 优化器会对 `multi_precision` 属性进行修改的错误。([#38991](https://github.com/PaddlePaddle/Paddle/pull/38991))

- 修复最终态 API amp 与 optional 类型 Tensor 组合使用的代码编译错误。([#40980](https://github.com/PaddlePaddle/Paddle/pull/40980))

- 修复 paddle+lite+xpu 预测库调用 lite CPU 预测时会报错的 bug，修复 paddle+lite(without NNAdapter) 编译时会报错的 bug。([#37449](https://github.com/PaddlePaddle/Paddle/pull/37449))

- 修复 Debug 编译模式下 LoDTensorArray 因 Pybind11 绑定不一致导致 crash 的 bug。([#37954](https://github.com/PaddlePaddle/Paddle/pull/37954))

- 修复 shape 参数为 Tensor 和 int 构成列表的极端情况下，无法正确构建 Tensor 的 bug。([#38284](https://github.com/PaddlePaddle/Paddle/pull/38284))

- 修复 `paddle.optimizer.AdamW` API 兼容性问题。([#37905](https://github.com/PaddlePaddle/Paddle/pull/37905))

- 修复 _InstanceNormBase 中 extra_repr 的返回错误。([#38537](https://github.com/PaddlePaddle/Paddle/pull/38537))

- 修复联编开启 -DWITH_DISTRIBUTED 生成 Paddle Inference 缺少符号 `paddle::distributed::TensorTable` 的问题。([#41128](https://github.com/PaddlePaddle/Paddle/pull/41128))

- matmul_v2 op 新增 shape check，在 shape 中存在 0 值进行信息报错。([#35791](https://github.com/PaddlePaddle/Paddle/pull/35791))

- 修复动态图 recompute 对于没有梯度输入提示信息反复打印，改成用 warning 只打印一次的方式。([#38293](https://github.com/PaddlePaddle/Paddle/pull/38293))

- 修复 gelu op 在视觉模型中训练后期在验证集上精度低的问题。([#38450](https://github.com/PaddlePaddle/Paddle/pull/38450))

- 修复 adamw op 在数值计算上误差问题。([#37746](https://github.com/PaddlePaddle/Paddle/pull/37746))

- 补充 sparse_momentum `_C_ops` 接口 MasterParam 和 MasterParamOut 参数。([#39969](https://github.com/PaddlePaddle/Paddle/pull/39969))

- 修复 python3.6 环境下没有 `distributed` 模块的 bug。([#35848](https://github.com/PaddlePaddle/Paddle/pull/35848))

- 修复 eigh 单元测试数据初始化问题。([#39568](https://github.com/PaddlePaddle/Paddle/pull/39568))

- 修复 eigvalsh 单元测试数据初始化问题。([#39841](https://github.com/PaddlePaddle/Paddle/pull/39841))

- 修复 segment op 在 V100 上寄存器使用过多导致不能正常运行的问题。([#38113](https://github.com/PaddlePaddle/Paddle/pull/38113))

- 修复 conv 相关算子稀疏化维度错误的问题。([#36054](https://github.com/PaddlePaddle/Paddle/pull/36054))

- 提供自动稀疏训练（Automatic SParsity）静态图相关功能 Alias 至 `Paddle.static.sparsity`。([#36525](https://github.com/PaddlePaddle/Paddle/pull/36525))

- 修复 divide op 整数除法还是整数的 bug。([#40890](https://github.com/PaddlePaddle/Paddle/pull/40890))

- 修复 `paddle.multiplex` 候选 Tensor 大小为 0 崩溃问题。([#34972](https://github.com/PaddlePaddle/Paddle/pull/34972))

- 修复 `paddle.kl_div` 参数 `reduction` 给定情况下速度异常的问题。([#37283](https://github.com/PaddlePaddle/Paddle/pull/37283))

- 修复 Cifar 数据集加载 data source 无序的问题。([#37272](https://github.com/PaddlePaddle/Paddle/pull/37272))

- 修复 ProgressBar 类中 loss 从 uint16 到 float 的转换。([#39231](https://github.com/PaddlePaddle/Paddle/pull/39231))

- 修复 ShareBufferWith 共享数据类型的问题。([#37464](https://github.com/PaddlePaddle/Paddle/pull/37464), [#37247](https://github.com/PaddlePaddle/Paddle/pull/37247))

- 修复 `paddle.io.DataLoader` 使用 IterableDataset 并且 num_workers>0 时的性能问题。([#40541](https://github.com/PaddlePaddle/Paddle/pull/40541))

- 修复 `paddle.vision.ops.yolo_loss` 动态图返回值不全的问题。([#40185](https://github.com/PaddlePaddle/Paddle/pull/40185))

- 移出 `paddle.io.BatchSampler` 对输入参数 dataset 需要是 `paddle.io.Dataset` 类型的限制，扩大对用户自定义数据集的支持。([#40184](https://github.com/PaddlePaddle/Paddle/pull/40184))

- 修复 `paddle.summary` 报错 op_flops 不存在的问题。([#36489](https://github.com/PaddlePaddle/Paddle/pull/36489))

- 修复 lars_momentum op 在 lars_weight_decay=0 时公式错误的问题。([#40892](https://github.com/PaddlePaddle/Paddle/pull/40892))

- 修复 optimize-offload 无法保存 presistable var 的问题。([#36433](https://github.com/PaddlePaddle/Paddle/pull/36433))

- 修复 optimizer-offload 不支持 adamw op type 的问题。([#36432](https://github.com/PaddlePaddle/Paddle/pull/36432))

- 修复多线程场景下，Tracer 中 enable_program_desc_tracing_数据不安全的问题。([#39776](https://github.com/PaddlePaddle/Paddle/pull/39776))

- 修复模型读取时模型档案大小未初始化的问题。([#40518](https://github.com/PaddlePaddle/Paddle/pull/40518))

- 修复 Expand op 逻辑 bug，当输入 Tensor X 的维度，小于要拓展的 shape 时，可能导致取得 Out.Shape 是错误的。([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))

- 修复 Expand_As op 只取 y.shape，而没有 Y 变量输入时，导致的动转静报错。([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))

- 修复 Expand_As op 计算输出 shape 时逻辑的错误。([#38677](https://github.com/PaddlePaddle/Paddle/pull/38677))


- 修复 `core.VarDesc.VarType.STRINGS` 类型的变量获取 `lod_level` 属性报错的问题，并且设置其 `lod_level` 为 None。([#39077](https://github.com/PaddlePaddle/Paddle/pull/39077))

- 修复框架功能 `PyLayer` 不支持不同 dtype 的问题。([#37974](https://github.com/PaddlePaddle/Paddle/pull/37974))

- 修复了学习率衰减 API `paddle.optimizer.lr.PolynomialDecay` 的零除问题。([#38782](https://github.com/PaddlePaddle/Paddle/pull/38782))

- 修复调用 DisableGlogInfo() 接口后依旧残留部分日志的问题。([#36356](https://github.com/PaddlePaddle/Paddle/pull/36356))

- 修复 SimpleRNN、GRU 和 LSTM API CPU 训练时多层 RNN（dropout 设置为 0 时）反向计算出错的问题。([#37080](https://github.com/PaddlePaddle/Paddle/pull/37080))

- 为 cufft 和 hipfft 后端的 fft 添加了 cache。([#36646](https://github.com/PaddlePaddle/Paddle/pull/36646))

- 使 `paddle.roll` 的 shifts 参数支持传入 Tensor。([#36727](https://github.com/PaddlePaddle/Paddle/pull/36727))

- 为 fft 添加 onemkl 作为可选的计算后端。([#36414](https://github.com/PaddlePaddle/Paddle/pull/36414))

- 修复 mamtul_v2 和 elementwise_div 两个 op 在 bfloat16 类型下的精度问题。([#42479](https://github.com/PaddlePaddle/Paddle/pull/42479))

- 修复显存回收时 LoDTensorArray 只清理内部 Tensor 而未清空 Array 导致的下个 step 可能出错的问题。([#42398](https://github.com/PaddlePaddle/Paddle/pull/42398))

## 4. 部署方向（Paddle Inference）

### （1）新增特性

#### 新增 API

- 增加 Java API，Java 开发者可以通过简单灵活的接口实现在服务端和云端的高性能推理。([#37162](https://github.com/PaddlePaddle/Paddle/pull/37162))

- 增加 `GetTrtCompileVersion` 和 `GetTrtRuntimeVersion` 接口，用于获取 TensorRT 版本信息。([#36429](https://github.com/PaddlePaddle/Paddle/pull/36429))

- 增加 `ShareExternalData` 接口，避免推理时对输入数据进行内存拷贝。([#39809](https://github.com/PaddlePaddle/Paddle/pull/39809))

#### 新增功能

- 新增 ONNX Runtime 后端支持，当前集成版本只支持 CPU。([#39988](https://github.com/PaddlePaddle/Paddle/pull/39988), [#40561](https://github.com/PaddlePaddle/Paddle/pull/40561))

- 基于 Paddle Lite 子图方式，新增昇腾 310 推理支持。([#35226](https://github.com/PaddlePaddle/Paddle/pull/35226))

- 新增原生 GPU FP16 推理功能。([#40531](https://github.com/PaddlePaddle/Paddle/pull/40531))

- switch_ir_debug 接口增加 dump 模型的功能。([#36581](https://github.com/PaddlePaddle/Paddle/pull/36581))

- 新增 TensorRT config 的配置接口：`void UpdateConfigInterleaved(paddle_infer::Config* c, bool with_interleaved)`，用于 int8 量化推理中特殊的数据排布。([#38884](https://github.com/PaddlePaddle/Paddle/pull/38884))

- log 中增加 TensorRT inspector 输出信息，仅在 TensorRT 8.2 及以上版本有效。([#38362](https://github.com/PaddlePaddle/Paddle/pull/38362)，[#38200](https://github.com/PaddlePaddle/Paddle/pull/38200)))

- 增加 TensorRT ASP 稀疏推理支持。([#36413](https://github.com/PaddlePaddle/Paddle/pull/36413))

### （2）底层优化

#### CPU 性能优化

- 优化 MKLDNN 的缓存机制。([#38336](https://github.com/PaddlePaddle/Paddle/pull/38336), [#36980](https://github.com/PaddlePaddle/Paddle/pull/36980), [#36695](https://github.com/PaddlePaddle/Paddle/pull/36695))

- 新增 matmul_scale_fuse pass。([#37962](https://github.com/PaddlePaddle/Paddle/pull/37962))

- 新增 MKLDNN reshape_transpose_matmul_v2_mkldnn_fuse_pass。([#37847](https://github.com/PaddlePaddle/Paddle/pull/37847), [#40948](https://github.com/PaddlePaddle/Paddle/pull/40948))

- 新增 MKLDNN conv_hard_sigmoid_mkldnn_fuse_pass。([#36869](https://github.com/PaddlePaddle/Paddle/pull/36869))

- 新增 MKLDNN matmul_v2_transpose_reshape_fuse_pass。([#36481](https://github.com/PaddlePaddle/Paddle/pull/36481))

- 新增 MKLDNN softplus_activation_mkldnn_fuse_pass。([#36657](https://github.com/PaddlePaddle/Paddle/pull/36657))

- 新增 MKLDNN elt_act_mkldnn_fuse_pass。([#36541](https://github.com/PaddlePaddle/Paddle/pull/36541))

- 新增 MKLDNN mish 算子及 conv_mish_mkldnn_fuse_pass。([#38623](https://github.com/PaddlePaddle/Paddle/pull/38623))

#### GPU 性能优化

- 将推理默认的显存分配策略由 `naive_best_fit` 变更为 `auto_growth`，解决部分模型占满 GPU 显存问题。([#41491](https://github.com/PaddlePaddle/Paddle/pull/41491))

- 支持 gelu、FC+gelu ops 使用 TensorRT 推理。([#38399](https://github.com/PaddlePaddle/Paddle/pull/38399))合作团队

- 支持 `deformable_conv` 在静态 shape 下使用 TensorRT 推理。([#36612](https://github.com/PaddlePaddle/Paddle/pull/36612) [#36850](https://github.com/PaddlePaddle/Paddle/pull/36850) [#37345](https://github.com/PaddlePaddle/Paddle/pull/37345))

- 支持 nearest_interp_v2 op 使用 TensorRT 推理。([#34126](https://github.com/PaddlePaddle/Paddle/pull/34126))

- 增加 `yolo_box`TensorRT plugin，支持输入参数 `iou_aware` 和 `iou_aware_factor`，使推理计算得到的 IoU 作为置信度的因子。([#34128](https://github.com/PaddlePaddle/Paddle/pull/34128))

- 支持 `elementwise_sub` 和 `elementwise_div` 调用 TensorRT 推理。([#40806](https://github.com/PaddlePaddle/Paddle/pull/40806) [#41253](https://github.com/PaddlePaddle/Paddle/pull/41253))

- 支持 `multiclass_nms3` 使用 TensorRT 推理。([#41181](https://github.com/PaddlePaddle/Paddle/pull/41181) [#41344](https://github.com/PaddlePaddle/Paddle/pull/41344))

- 支持 flatten_contiguous_rang op 使用 TensorRT 推理。([#38922](https://github.com/PaddlePaddle/Paddle/pull/38922))

- 支持 `pool2d` 属性 `padding` 的维度为 4、`global_pooling` 和 `ceil_mode` 为 True 情况下使用 TensorRT 推理。([#39545](https://github.com/PaddlePaddle/Paddle/pull/39545))

- 支持 batch_norm 和 elementwise_add 为 5 维时使用 TensorRT 推理。([#36446](https://github.com/PaddlePaddle/Paddle/pull/36446))

- 新增 pool3d 使用 TensorRT 推理。([#36545](https://github.com/PaddlePaddle/Paddle/pull/36545), [#36783](https://github.com/PaddlePaddle/Paddle/pull/36783))

- 增加 `reduce` int32 和 float 类型使用 TensorRT 推理，增加 `reduce_mean` GPU 算子 int32、int64 注册。([#39088](https://github.com/PaddlePaddle/Paddle/pull/39088))

- 修改 MatmulV2ToMul pass，修改限定条件（不支持广播）和 op_teller 映射条件。([#36652](https://github.com/PaddlePaddle/Paddle/pull/36652))

- 增加 TenorRT plugin 接口 AddPluginV2IOExt 的支持。([#36493](https://github.com/PaddlePaddle/Paddle/pull/36493))

- 增加 roi_align op 中 aligned 属性并支持 TensorRT 推理。([#38905](https://github.com/PaddlePaddle/Paddle/pull/38905))

- 增加 concat 属性 `axis = -1` 时支持 TensorRT 推理。([#39096](https://github.com/PaddlePaddle/Paddle/pull/39096))

- 新增 TensorRT plugin ：preln_emb_eltwise_layernorm、 preln_skip_la、rnorm ops， 用于 ERNIE 类模型性能优化。([#39570](https://github.com/PaddlePaddle/Paddle/pull/39570))

- 新增 TensorRT fuse pass：preln_embedding_eltwise_layernorm_fuse_pass, preln_skip_layernorm_fuse_pass，用于 ERNIE 类模型性能优化。([#39508](https://github.com/PaddlePaddle/Paddle/pull/39508))

- 将 matmul 融合相关的 pass 基于不同的后端（GPU、CPU、TensorRT）拆开，支持 FC 权重的转置功能。([#39369](https://github.com/PaddlePaddle/Paddle/pull/39369))

- 新增 roll、strided_slice、slice op 在动态 shape 的情况下对 TensorRT 的支持。([#41913](https://github.com/PaddlePaddle/Paddle/pull/41913), [#41573](https://github.com/PaddlePaddle/Paddle/pull/41573), [#41467](https://github.com/PaddlePaddle/Paddle/pull/41467))

- 新增 div op 对 TensorRT 的支持。([#41243](https://github.com/PaddlePaddle/Paddle/pull/41243))

- 量化支持

  - `PostTrainingQuantization` API 新增支持`paddle.io.DataLoader` 对象或者 `Python Generator`的输入。([#38686](https://github.com/PaddlePaddle/Paddle/pull/38686))

  - ERNIE 全量化模型推理支持 interleaved 数据排布。([#39424](https://github.com/PaddlePaddle/Paddle/pull/39424))

  - 支持 PaddleSlim 新量化模型格式推理。([#41049](https://github.com/PaddlePaddle/Paddle/pull/41049))

  - 新增 matmul int8 量化的推理 op converter 和 plugin。([#37285](https://github.com/PaddlePaddle/Paddle/pull/37285))

  - 新增判断模型所有 op 能否支持 int8 量化的 pass。([#36042](https://github.com/PaddlePaddle/Paddle/pull/36042))

  - 支持 multihead attention 非变长分支中 FC 部分的量化推理。([#39660](https://github.com/PaddlePaddle/Paddle/pull/39660))

#### 昇腾 NPU 相关功能

- - 重构 shape 算子前向计算逻辑，支持在 NPU 上执行。([#39613](https://github.com/PaddlePaddle/Paddle/pull/39613))

  - 重构 reshape 算子前向计算逻辑，支持 ShapeTensor 输入。([#38748](https://github.com/PaddlePaddle/Paddle/pull/38748))

  - 模型权重加载时精度类型统一。([#39160](https://github.com/PaddlePaddle/Paddle/pull/39160))

### （3）问题修复

#### 框架及 API 修复

- 修复保存静态图时模型剪裁的问题。([#37579](https://github.com/PaddlePaddle/Paddle/pull/37579))

- C API 增加对的字符串的封装 PD_Cstr，并提供构造和析构的方式，避免用户直接使用 C 运行时库来析构字符串。([#38667](https://github.com/PaddlePaddle/Paddle/pull/38667))

- 修复预测时内存复用的逻辑问题。([#37324](https://github.com/PaddlePaddle/Paddle/pull/37324))

- 修复多线程下内存复用报错问题。([#37894](https://github.com/PaddlePaddle/Paddle/pull/37894))

- 在没有权重文件时，允许传递空字符串进行推理。([#38579](https://github.com/PaddlePaddle/Paddle/pull/38579))

- 修复开启 TensorRT dynamic shape 后不支持 clone 问题。([#38520](https://github.com/PaddlePaddle/Paddle/pull/38520))

- 修复开启 TensorRT dynamic shape 后多线程 clone 报错问题。([#40067](https://github.com/PaddlePaddle/Paddle/pull/40067))

- 修复 TensorRT engine 析构问题。([#35842](https://github.com/PaddlePaddle/Paddle/pull/35842), [#35938](https://github.com/PaddlePaddle/Paddle/pull/35938))

- lite xpu 接口修复无法选择 xpu 卡的问题。([#36610](https://github.com/PaddlePaddle/Paddle/pull/36610))

- TensorRT 动态 shape 参数自动生成接口增加文件存在性检查。([#36628](https://github.com/PaddlePaddle/Paddle/pull/36628))

- 修复 MKLDNN 不支持 conv3d 的问题。([#42055](https://github.com/PaddlePaddle/Paddle/pull/42055))

#### 后端能力修复

- 修复预测时 cuDNN 默认算法选择配置，使用非 deterministic 策略。([#41491](https://github.com/PaddlePaddle/Paddle/pull/41491))

- 修复 deformable_conv op 在 TensorRT plugin 资源回收处理错误的问题。([#38374](https://github.com/PaddlePaddle/Paddle/pull/38374))

- 修复 deformable_conv op 在 TensorRT plugin 序列化错误问题。([#38057](https://github.com/PaddlePaddle/Paddle/pull/38057))

- 适配 TensorRT 8.0 新的构建引擎和系列化 API。([#36769](https://github.com/PaddlePaddle/Paddle/pull/36769))

- 修复 Flatten2MatmulFusePass、Squeeze2MatmulFusePass、Reshape2MatmulFusePass 没有生效问题。([#37644](https://github.com/PaddlePaddle/Paddle/pull/37644))

- 修复 TensorRT 输入数据在上时报错的问题。([#37427](https://github.com/PaddlePaddle/Paddle/pull/37427))

- 增加输入维度错误时的报错信息。([#38962](https://github.com/PaddlePaddle/Paddle/pull/38962))

- 修复 EmbEltwiseLayernorm 输出类型错误的问题。([#40015](https://github.com/PaddlePaddle/Paddle/pull/40015))

- 删除 conv_affine_channel_fuse_pass 以及对应的单元测试。([#39817](https://github.com/PaddlePaddle/Paddle/pull/39817))

- 修复 adaptive_pool2d pass 错误替换 pool 属性的问题。([#39600](https://github.com/PaddlePaddle/Paddle/pull/39600))

- 修复 shuffle_channel_detect_pass 错误生成 shuffle_channel op 的问题。([#39242](https://github.com/PaddlePaddle/Paddle/pull/39242))

- 修复 transpose 参数错误。([#39006](https://github.com/PaddlePaddle/Paddle/pull/39006))

- 修复 nearest_interp_v2 输入 scale 维度小于 1 时崩溃的问题。([#38725](https://github.com/PaddlePaddle/Paddle/pull/38725))

- 修复 prelu 在 dynamic shape 时不支持一维输入的问题。([#39389](https://github.com/PaddlePaddle/Paddle/pull/39389))

- 修复 slice 的 special_slice_plugin 的核函数计算错误的问题。([#39875](https://github.com/PaddlePaddle/Paddle/pull/39875))

- 暂时禁用 skip_layernorm 变长下的 int8 分支，防止精度下降。([#39991](https://github.com/PaddlePaddle/Paddle/pull/39991))

- 修复关于支持 preln_ernie 模型的一些 bug。([#39733](https://github.com/PaddlePaddle/Paddle/pull/39733))

- 修复 slice 在 ERNIE 中 threads 可能超过限制的 bug，修复 spacial_slice 误触的 bug。([#39096](https://github.com/PaddlePaddle/Paddle/pull/39096))

- 修复 elementwise 在维度相同时不支持广播的问题。([#37908](https://github.com/PaddlePaddle/Paddle/pull/37908))

- 修复 nearest_interp op 当 align_corners 为 True 时，TensorRT layer 的结果和原生 op 的结果有 diff，底层实现不一样。([#37525](https://github.com/PaddlePaddle/Paddle/pull/37525))

- 修复 qkv_plugin：核函数计算错误。([#37096](https://github.com/PaddlePaddle/Paddle/pull/37096))

- 修复动态量化的推理 pass 的问题。([#35879](https://github.com/PaddlePaddle/Paddle/pull/35879))

- 当 Tensor 请求的内存容量低于已分配的 size 时直接复用。([#37880](https://github.com/PaddlePaddle/Paddle/pull/37880))

- 修复 ERNIE 定长模型开启 TensorRT 出现的 hang 问题。([#37839](https://github.com/PaddlePaddle/Paddle/pull/37839))

- 修复 TensorRT int8 时缺失 dynamic range 信息崩溃问题。([#36900](https://github.com/PaddlePaddle/Paddle/pull/36900))

- 修复 slice 反序列化代码问题。([#36588](https://github.com/PaddlePaddle/Paddle/pull/36588))

- 修复 yolo box 计算公式错误问题。([#36240](https://github.com/PaddlePaddle/Paddle/pull/36240))

- 修复老版本模型在使用新版本 roi_align 时崩溃问题。([#38788](https://github.com/PaddlePaddle/Paddle/pull/38788)) 外部开发者

- 修复 softmax 在 python 和 C++上性能差异较大的问题。([#37130](https://github.com/PaddlePaddle/Paddle/pull/37130))

- 修复 matmul 在静态 shape 2 维输入和动态 shape 3 维输入情况下推理失败问题。([#36849](https://github.com/PaddlePaddle/Paddle/pull/36849))

- 修复 reshape_transpose_matmul_mkldnn_fuse_pass 对 shape 处理不当问题。([#36731](https://github.com/PaddlePaddle/Paddle/pull/36731))

- 修复输入为 2 维，但 TensorRT 获取到 4 维的问题。([#36614](https://github.com/PaddlePaddle/Paddle/pull/36614))

- 修复 interpolate_v2 MKLDNN 算子在 scale 属性为空时报错问题。([#36623](https://github.com/PaddlePaddle/Paddle/pull/36623))

- 修复 recurrent 算子在多线程场景性能差问题。([#36052](https://github.com/PaddlePaddle/Paddle/pull/36052))

- 移除 relu、sigmoid、tanh、relu6、batch_norm、clip、concat、gelu、hard_sigmoid、prelu、softmax、split、swish 对 TensorRT 2 维输入的限制。([#37097](https://github.com/PaddlePaddle/Paddle/pull/37097))

- 修复 reshape op 使用 TensorRT 推理。([#41090](https://github.com/PaddlePaddle/Paddle/pull/41090))

- 修复 matmul 相关 pass，兼容 matmul_v2。([#36424](https://github.com/PaddlePaddle/Paddle/pull/36424))

- 开启 TensorRT 时，conv2d 算子中 padding 方式支持 VALID 及 SAME 属性。([#38999](https://github.com/PaddlePaddle/Paddle/pull/38999))

- 修复 MKLDNN 多输入算子量化问题。([#39593](https://github.com/PaddlePaddle/Paddle/pull/39593), [#39346](https://github.com/PaddlePaddle/Paddle/pull/39346), [#40717](https://github.com/PaddlePaddle/Paddle/pull/40717))

- 修复 MKLDNN 量化场景下 conv+activation 的 scale 错误问题。([#38331](https://github.com/PaddlePaddle/Paddle/pull/38331))

- 修复 MKLDNN 无参数算子量化中，根据后续算子量化情况不同需做不同处理的问题。([#39342](https://github.com/PaddlePaddle/Paddle/pull/39342))

- 修复 MKLDNN cpu_bfloat16_placement_pass 中的数据类型相关问题。([#38702](https://github.com/PaddlePaddle/Paddle/pull/38702))

- 修复 MKLDNN bfloat16 推理中 split 算子执行问题。([#39548](https://github.com/PaddlePaddle/Paddle/pull/39548))

- 修复 MKLDNN matmul_v2 算子不支持 6 维问题。([#36342](https://github.com/PaddlePaddle/Paddle/pull/36342), [#38665](https://github.com/PaddlePaddle/Paddle/pull/38665))

- 修复 MKLDNN matmul_v2_transpose_reshape 中的 MKLDNN DeviceContext 错误问题。([#38554](https://github.com/PaddlePaddle/Paddle/pull/38554))

- 修复分割模型在 MKLDNN 推理场景计算结果错误问题。([#37310](https://github.com/PaddlePaddle/Paddle/pull/37310))

- 修复 MKLDNN bfloat16 placement 算子列表并添加缺失算子。([#36291](https://github.com/PaddlePaddle/Paddle/pull/36291))

- 修复 MKLDNN 算子的格式问题，包括：FC、conv_transpose、6 维 Tensor 报错问题、conv 对 `NHWC` 输入的输出 format 错误问题。([#38890](https://github.com/PaddlePaddle/Paddle/pull/38890), [#37344](https://github.com/PaddlePaddle/Paddle/pull/37344), [#37175](https://github.com/PaddlePaddle/Paddle/pull/37175), [#38553](https://github.com/PaddlePaddle/Paddle/pull/38553), [#40049](https://github.com/PaddlePaddle/Paddle/pull/40049), [#39097](https://github.com/PaddlePaddle/Paddle/pull/39097))

- 修复 MKLDNN 多线程推理场景因 cache 机制报错问题。([#36290](https://github.com/PaddlePaddle/Paddle/pull/36290), [#35884](https://github.com/PaddlePaddle/Paddle/pull/35884))

- 修复 MKLDNN 因 matmul 及 FC 引起的量化模型精度异常问题。([#38023](https://github.com/PaddlePaddle/Paddle/pull/38023), [#37618](https://github.com/PaddlePaddle/Paddle/pull/37618))

- 修复 MKLDNN 量化转换脚本因 pass 缺少引起的量化模型精度异常问题。([#37619](https://github.com/PaddlePaddle/Paddle/pull/37619), [#40542](https://github.com/PaddlePaddle/Paddle/pull/40542),
  [#38912](https://github.com/PaddlePaddle/Paddle/pull/38912))

- 修复 MKLDNN 开启量 op 因为数据类型不匹配崩溃的问题。([#38133](https://github.com/PaddlePaddle/Paddle/pull/38133))

- 修复 MKLDNN 某些 op 修改 layout 后需要改回原 layout 的问题。([#39422](https://github.com/PaddlePaddle/Paddle/pull/39422))

- 修复针对昇腾 910 推理场景下，由于未释放 GIL 锁，导致与昇腾软件栈冲突，python API 下报错的问题。([#38605](https://github.com/PaddlePaddle/Paddle/pull/38605))

## 5. 环境适配

### 编译安装

- 从 2.3.0 版本开始，飞桨对框架支持的 GPU 架构种类进行了调整和升级。(更多请参考：[飞桨支持的 GPU 架构](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.3rc/install/Tables.html#gpu))

备注：

- PIP 源安装是指用 `pip install paddlepaddle` 或 `pip install paddlepaddle-gpu`从 PIP 官网下载安装包及依赖库的安装方式，支持架构种类少，安装包更轻量，下载源来自国外（相比 bos 源支持架构种类精简，安装包更轻量，只提供一种 CUDA 版本的安装包）。

  - 2.3 版本之前，飞桨 PIP 源安装包（CUDA10.2）支持的 GPU 架构为：3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 7.5。

  - 2.3 版本之后，飞桨 PIP 源安装包（CUDA11.0）支持的 GPU 架构为：6.0, 6.1, 7.0, 7.5, 8.0

- 飞桨官网 bos 源是指从飞桨官网下载安装包及依赖库的安装方式，支持的 GPU 架构更多，下载源来自国内，速度较快。（相比 PIP 源支持架构种类多，提供多个 CUDA 版本的安装包）：

  - 2.3 版本之前，飞桨官网 bos 源安装包支持的 GPU 架构：

    - CUDA10：3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 7.5；

    - CUDA11：5.2，6.0，6.1，7.0，7.5，8.0。

  - 2.3 版本之后，飞桨官网 bos 源安装包支持的 GPU 架构

    - CUDA10：3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 7.5；

    - CUDA11：3.5, 5.0, 6.0, 6.1, 7.0, 7.5, 8.0。

- 支持 Python 3.10，修复 Windows 下某些 PythonC API 变化导致的编译 bug。([#41180](https://github.com/PaddlePaddle/Paddle/pull/42180))

- Windows 平台支持 Visual Studio 2019 编译。([#38719](https://github.com/PaddlePaddle/Paddle/pull/38719))

- 消除 Windows 平台编译时出现的各种 warning。([#38034](https://github.com/PaddlePaddle/Paddle/pull/38034), [#37890](https://github.com/PaddlePaddle/Paddle/pull/37890), [#37442](https://github.com/PaddlePaddle/Paddle/pull/37442), [#37439](https://github.com/PaddlePaddle/Paddle/pull/37439), [#36857](https://github.com/PaddlePaddle/Paddle/pull/36857))

- 修复底层数据结构升级引入的 jetson 编译问题。([#39669](https://github.com/PaddlePaddle/Paddle/pull/39669), [#39441](https://github.com/PaddlePaddle/Paddle/pull/39441))


### 新硬件适配

- 自定义新硬件接入：提供一种插件式扩展 PaddlePaddle 硬件后端的方式。通过该功能，开发者无需为特定硬件修改 PaddlePaddle 代码，只需实现标准接口，并编译成动态链接库，则可作为插件供 PaddlePaddle 调用。降低为 PaddlePaddle 添加新硬件后端的开发难度。当前支持自定义 Runtime 接入和自定义 Kernel 接入。

- 华为 NPU 芯片（Ascend910）训练/推理支持，支持 ResNet50、YoloV3、BERT、Transformer 等多个模型，支持静态图与混合精度训练，支持单卡、单机、多机分布式训练。

- Graphcore IPU 芯片（包括 IPU Mk2 GC200 和 Bow IPU）训练/推理支持，支持 ResNet50、BERT 等模型，支持静态图训练，支持单芯片、单机、多机分布式训练。

- 寒武纪 MLU 芯片（MLU370x4）训练/推理支持，支持 ResNet50 等模型，支持静态图+动态图训练，支持混合精度训练，支持单卡、单机、多机分布式训练。

- 昆仑芯 2 代芯片（昆仑芯 AI 加速卡 R200、R300）训练/推理支持，支持 ResNet50、YoloV3、OCR-DB、SSD、MobilnetV3、UNet、BERT、Transformer、GPT-2、Wide&Deep、DeepFM，支持静态图+动态图训练，支持混合精度训练，支持单机单卡、单机多卡训练。

## Thanks to our Contributors

This release contains contributions from the project core team as well as:

Adam Osewski, Allen Guo, arlesniak, chenenquan, chenyanlann, fengkuangxiaxia, fuqianya, fwenguang, guguguzi, helen88, houj04, Jacek Czaja, jakpiase, jianghaicheng, joanna.wozna.intel, joeqiao12, Leo Chen, Leo Guo, Li-fAngyU, lidanqing, Liyulingyue, Matsumoto GAO, maxhuiy, Ming-Xu Huang, Nyakku Shigure, piotrekobi, piotrekobiIntel, QingshuChen, qipengh, Skr Bang, Sylwester Fraczek, Sławomir Siwek, taixiurong, tanzhipeng, Tomasz Socha, TTerror, Webbley, yaozhixin, ykkk2333, yujun, Zhangjingyu06, zhangxiaoci, zhangyikun02, zhangyk0314, zlsh80826, zn, Zuza.
