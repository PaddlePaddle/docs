# 版本说明

## 目录

* 重要更新
* 基础框架
    * 安装
    * 中间表达IR和Pass方面的优化
    * IO优化
    * 执行优化
    * 显存优化
    * 完善CPU JITKernel
    * Intel CPU底层计算优化
    * 集成Intel nGraph图编译引擎
    * 框架基础功能增强
    * 动态图preview版基础功能完善
* 预测引擎
    * 服务器预测引擎
    * 移动端预测引擎
    * 部署工具
* 分布式训练
* 模型建设
    * PaddleCV 智能视觉
    * PaddleNLP智能文本处理
    * PaddleRec智能推荐
* 工具组件
* BUG修复

## 重要更新

* 基础框架对训练速度和显存占用进行了全面优化，完整支持量化训练。初步集成了Intel nGraph，动态图preview版单机单卡基本功能完善。
* 正式发布模型压缩工具包[PaddleSlim](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim)和模型预测服务[Paddle Serving](https://github.com/PaddlePaddle/Serving)，全面提升PaddlePaddle部署能力。
* 优化分布式IO，增加远程文件系统流式读取能力。GPU多机多卡同步训练通过增加稀疏通信能力提升带宽不敏感训练能力，在低配网络带宽网络环境下，例如10G网络下，同步训练可提速10倍。
* 更好支持K8S生态，提供工业生产环境下的Paddle-K8S-Operator支持；Kubeflow支持paddle-job。
* 正式发布[视频识别工具集](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video)，覆盖主流视频分类模型，包括Nonlocal、TSM 、Attention Cluster、NeXtVLAD、LSTM、StNet、TSN。
* 新增中文语义表示模型[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，在多项中文任务上相对 BERT精度绝对提升1-2个百分点。新增对话通用理解相关模型 DGU，支持5类对话任务，在3个公开数据集达到 SOTA 的效果。
* 新增基于图神经网络的推荐模型[（Graph Neural Network）](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn)，并提供公开数据集下的benchmark效果。
* 正式发布[PaddleHub](https://github.com/PaddlePaddle/PaddleHub)预训练模型管理工具，提供包括预训练模型管理、命令行一键式使用和迁移学习三大功能。旨在帮助用户更高效地管理模型并开展迁移学习的工作。
* 正式发布[X2Paddle模型转换工具](https://github.com/PaddlePaddle/X2Paddle)，用户可以无损地将其他深度学习框架预测模型迁移至PaddlePaddle。

## 基础框架

* 安装
    * 增加install\_check.run\_check()接口，对安装是否成功提供更完善的检查。
* 中间表达IR和Pass方面的优化
    * 完成IrGraph、IrNode、IrVarNode以及IrOpNode的封装，支持使用Python编写IR Pass。
* IO优化
    * PyReader接口优化：可通过新接口reader = fluid.io.PyReader (..., iterable=True, ...)创建for循环可迭代的reader，并通过feed方式将数据送入网络训练。
* 执行优化
    * 用户可设置with\_data\_parallel的places参数，指定在某些GPU卡上运行，从而支持单进程多训练任务执行。
    * 优化了多卡执行器调度策略，在ResNet50和Transformer模型上验证速度提升8%~19%。
    * 多卡情况下支持对AllReduce进行按分组Fuse，ResNet模型的多卡速度提升8%~30%（不同卡数提速有差异），Transformer模型的多卡速度提升4%左右。
* 显存优化
    * GC策略优化：Eager Deletion策略支持while\_op内部变量的及时删除；支持非全量Eager Deletion策略，用户可设置FLAGS\_memory\_fraction\_of\_eager\_deletion=0.xx控制即时删除内存/显存空间的百分比。
    * Op优化：优化cross entropy、expand、layer\_norm、dropout等p的反向注册机制，去除无关变量依赖，提高框架显存性能。
    * 新增两个FLAGS（FLAGS\_initial\_gpu\_memory\_in\_mb和FLAGS\_reallocate\_gpu\_memory\_in\_mb）来让用户指定初始显存池容量和再分配显存池容量。
    * 调整inplace\_op\_pass策略，提高inplace的策略的覆盖率。
    * 取消了在python端做activation op inplace优化的逻辑，统一到inplace\_op\_pass。
    * 新增Memory Profile功能。
* 完善CPU JITKernel
    * 优化JITKernel的调用方式，添加Cache机制和获取所有相同类型函数的接口，方便开发者根据不同情况有选择的调用。
    * 使用JITKernel优化SGD算法，在PyramidDNN模型下对应的OP部分速度提升44%，整体训练速度提升12%；使用JITKernel优化fused\_embedding\_seq\_pool，在PyramidDNN模型下对应op的反向算子速度提升18%， 整体训练速度提升6%。
* Intel CPU底层计算优化
    * MKLDNN升级至18，包含若干性能增强（如基于GEMM的卷积运算 / INT8卷积运算等）。
    * 使用MKL优化GELU OP，OP性能提升至原来的3倍。
    * 增强MKLDNN相关Kernel的单元测试。
* 集成了Intel nGraph图编译引擎，为PaddlePaddle支持更多硬件后端提供了便利
    * 通过ngraph\_engine OP将子图交由nGraph核心，经图优化后调度在CPU上执行。用环境变量FLAGS\_use\_ngraph=true即可在运行时调用nGraph。
    * 支持ResNet50模型在CPU上的训练和预测。ResNet50在CPU上的性能，和基于MKLDNN的直接优化相比，预测和训练性能均有显著提升。
* 框架基础功能增强
    * 支持同步的Batch Norm操作；支持softmax设置axis; 新增spectral norm, rang, acos, asin, atanh操作；新增Npair Loss，用于特征学习。
    * 框架中添加cosine\_decay学习率调整策略。
    * 新增sampled\_softmax\_with\_cross\_entropy, 用于提升大词典下的训练效率。
    * 支持SGD和Adam优化算法的fuse，在Transformer模型上，速度能够提升2%，在Cycle GAN模型上，速度能够提升6%。
    * 加强lsmtp，支持cell内部裁剪、初始化cell state和hidden state。
    * 加强adagrad，支持初始化累积动量。
    * 支持Tensor使用\_\_getitem\_\_ 方式操作。
    * 新增QuantizationFreezePass、ConvertToInt8Pass以及TransformForMobilePass。完整支持动态和静态两种量化训练方式及对应模型保存。
* 动态图preview版基础功能完善：
    * 基础功能：支持LRDecay，整体支持GPU单卡及CPU单机的模型训练和评估。
    * API：公开动态图对应基础接口，重构现有的 Layers，增加对 GRU、LayerNorm、NCE、PRelu 等 Layers 的支持。
    * 性能：在Resnet，Mnist模型上验证与静态图基本持平。
    * 增加Transformer、MNIST、Se-Resnext 等模型的动态图实现。

## 预测引擎

### 服务器预测

* 预测库整合PaddlePaddle/Anakin，统一接口提供高效预测能力。
    * 支持Anakin GPU子图和CPU子图。
    * Python预测接口支持Anakin子图。
    * Resnet、VGG、Googlenet、Mobilenet、ShuffleNet、Faster RCNN、Yolo、SSD等模型实现显著预测加速。
* 预测框架优化，小模型预测速度提升明显
    * 增加runtime\_context\_cache\_pass，重点模型提升17%。
    * 优化5个OP的infershape，重点模型提升13%。
    * 完善ZeroCopy接口，避免使用AnalysisPredictor 时存在多余CPU拷贝。
* INT8 量化预测持续加强
    * 进一步完善通过TensorRT 支持INT8 量化，支持Alexnet、Googlenet、Vgg、Mobilenet、ShuffleNet等模型。优化调用TensorRT下的信息序列化反序列化，加快模型初始化速度。
    * 实现基于C++ Pass的INT8量化框架。增加若干INT8 OP Kernel : Transpose, Contact, Requantize。通过微调MkldnnQuantizerConfig中的量化策略，用户可快速得到符合精度要求的INT8量化模型。INT8量化后的ResNet-50 / MobileNet v1模型，相比原始FP32模型，性能分别提升至7倍 / 3.0倍 （在支持AVX512-DL Boost指令集的至强 6271服务器上)。

### 移动端预测

* ARM CPU
    * Paddle-mobile完成矩阵运算库sgemm和sgemv的重构和效率优化，在大部分模型上能获得10%〜100%以上的性能加速。
    * 新增while、sequence\_expand、sequence\_pool、sequence\_softmax、gru\_unit、beam\_search和beam\_search\_decode等19个算子，以及对应大量的优化工作，支持attention-based端到端模型的预测。
    * 新增winograd 的arm v8实现，在IOS上的v8的硬件上能取得更高的预测性能；winograd支持算子融合 ，保证算子融合后的效率更高。
    * 新增kernel为3x3的滑窗直接卷积实现，在channel数较少时会比winograd和gemm效率更高。
    * 完成kernel为3x3的depthwise convolution重构和优化，相比之前版本支持任意的padding、性能更优且计算结果更可靠。
    * 完成kernel为5x5的depthwise convolution armv8版本的实现，NAS模型的预测效率提升30%以上。
    * 完成反卷积conv2d\_transpose的效率优化。
    * 新增基于图优化的精简内存复用策略，大部分模型能降低近50%的内存占用。对于ARM CPU已自动开启（FPGA和GPU暂不支持）。
* ARM GPU
    * Paddle-mobile完成kernel为1x1的卷积优化，MobileNet v1在高通Adreno GPU上平均预测性能提升35%。
* Paddle Inference初步完成和Paddle-mobile、Anakin的接口统一，待进一步深度融合。

### 部署工具

* 模型压缩工具包PaddleSlim
    * 剪切模型压缩策略：支持敏感度和uniform两种方式，支持vgg、resnet、mobilenet等多种类型的网络，支持用户自定义剪切范围。
    * 量化训练模型压缩策略：支持动态和静态两种量化训练方式，支持对参数进行分channel量化或整体量化，支持以float类型模拟int8值域保存模型，支持以int8类型保存模型，支持以兼容paddle-mobile的格式保存模型。
    * 蒸馏模型压缩策略：支持在teacher网络和student网络任意层添加组合loss，支持FSP Loss, L2 Loss, Softmax with Cross-entropy Loss。
    * 其它功能：支持配置文件管理压缩任务超参数，支持多种压缩策略组合使用，蒸馏和剪切压缩过程支持checkpoints功能。

* Paddle Serving

    * 支持paddle inference远程部署。
    * 服务端支持用户新增数据处理Operator，支持用户自定义预估逻辑，支持模型热加载功能。
    * 客户端提供C++ SDK，供业务逻辑进行调用，支持自定义protobuf定制网络数据传输协议，A/B测试能力。
    * 提供经典任务使用paddle serving的示例模板，包括文本分类，图像分类任务。
    * 针对文本分类任务，给出延迟和吞吐的benchmark。

## 分布式训练

* 分布式IO优化
    * Pipe Reader接口优化：在保持数据预处理灵活性的前提下，提供高效IO的方法。支持企业级Linux系统用户定制化，实现高性能IO组件，在离线数据预处理处进行统一维护。增强远程文件系统流式读取能力，支持数据载入内存模式、分布式打乱功能。
* Executor与分布式IO的整合
    * AsyncExecutor整合进入Executor，增加train\_from\_dataset/infer\_from\_dataset接口，支持基于Pipe Reader的训练，在保持多队列IO功能的前提下，支持用户自定义PipeLine程序，提供python端灵活处理数据的能力。
* GPU多机多卡同步训练增加带宽不敏感训练能力
    * GPU同步训练增加稀疏通信能力，支持sparse all reduce。
    * 通过通信稀疏度的控制，在算法层面保障模型收敛，并增加DGCOptimizer。
    * 通过在resnet50 on imagenet上进行实验证明：模型收敛性方面，resnet50 90轮收敛效果不变；在高速互联网络环境下，稀疏通信不会降低训练速度；低配网络带宽网络环境下（例如10G网络），稀疏通信在训练速度上有明显优势，相比稠密通信的同步训练提速10倍。
* Collective Operator模式
    * Collective Operator模式的支持，增加GPU下多个all reduce的操作。通过Python API向Program中增加collective op，使得分布式优化算法开发的灵活性显著提升。
* Resnet50 on Imagenet收敛速度优化
    * 支持动态BatchSize、动态ImageSize以及矩形crop等方法；FP32精度下,在v100单机8卡验证，收敛速度提升68%(acc1\&gt;=75.9%, acc5=93.0%)。
* K8S生态支持
    * Kubeflow支持paddle-job，并贡献到kubeflow社区。
    * 支持工业生产环境下的Paddle-K8S-Operator，可与kubeflow配合使用。
    * K8S环境适合新手提交任务的脚本，提供百度云可复现教程。

## 模型建设

* PaddleCV 智能视觉
    * 正式发布视频识别工具集，覆盖主流视频分类模型，包括Nonlocal、TSM 、Attention Cluster、NeXtVLAD、LSTM,、StNet、TSN，效果和主流实现打平。
    * 新增基于ImageNet的预训练模型: GoogleNet, ShuffleNetV2, ResNet18,ResNet34。
    * 新增支持目标检测YOLOv3模型，效果与最好公开实现打平（mAP比原作者提高7绝对百分点）。
    * 发布基于COCO和MPII数据的Simple Baselines人体姿态估计模型，效果和主流实现打平。
    * 特征学习模型新增npair loss， 在预训练模型（arcmargin loss）的基础上将recall@1提升至03%（+0.78%)。
* PaddleNLP智能文本处理
    * 新增支持中文语义表示ELMO模型，支持多卡训练，训练速度比主流实现快1倍。验证在中文词法分析任务上F1值绝对提升1.1%，在中文阅读理解任务上Rouge-L值提升1%。
    * 新增中文语义表示模型ERNIE，在自然语言推断、语义相似度、命名实体识别、情感分析、问答匹配等中文任务上相对 BERT 中文模型绝对提升了 1% ~ 2% 的精度。
    * 阅读理解模型升级，优化数据预处理和文档选取，在DuReader验证数据集上Rouge-L提升至65（baseline 39.29)。
    * 新增基于知识感知的对话模型，对比基线生成对话模型，在F1, BLEU1, BLEU2的指标上平均提升1个百分点。
    * 发布对话模型工具集，包含Deep Attention Matching Net, 新增对话自动评估工具和基于BERT的对话通用理解相关模型DGU(Dialogue General Understanding)，支持对话语义匹配、DA、DST、槽位解析和意图识别五种对话任务，3个公开数据集达到SOTA 的效果。
    * 发布PaddleNLP工具包，统一文本分类、文本匹配、序列标注、阅读理解、智能对话等NLP任务的建模，并开放对应的工业级预训练模型。
* PaddleRec智能推荐
    * Deep Interest Network（DIN）：新增DIN模型，并在公开数据复现效果，支持cpu和gpu模式下的单机单/多卡训练。DIN适用于推荐中的排序场景（如ctr预估），主要特点为对历史序列建模的过程中结合了预估目标的信息。
    * Graph Neural Network（GNN）：新增基于session的图神经网络推荐模型，并在公开数据复现效果，支持cpu和gpu模式下的单机单卡训练。该模型适用于推荐中的召回场景，使用GNN对用户的历史信息进行建模，可以捕捉到item序列之间蕴含的更复杂的转换关系。
    * Word2vec：word2vec采样策略调优，并在公开数据复现效果，添加多机训练支持。

## 工具组件

* 正式发布PaddleHub预训练模型管理工具，旨在帮助用户更高效的管理模型并开展迁移学习的工作
    * **预训练模型管理** ：通过hub命令行可完成PaddlePaddle生态的预训练模型下载、搜索、版本管理等功能。
    * **命令行一键使用：** 无需代码，通过命令行即可直接使用预训练模型进行预测，快速调研训练模型效果。目前版本支持以下模型：词法分析LAC；情感分析Senta；目标检测SSD；图像分类ResNet, MobileNet。
    *  **迁移学习：** 提供了基于预训练模型的Finetune API，用户通过少量代码即可完成迁移学习，包括BERT/ERNIE文本分类、序列标注、图像分类迁移等。
* 正式发布X2Paddle模型转换工具，可以无损地将其他深度学习框架预测模型迁移至PaddlePaddle。工具还附带TensorFlow, Caffe框架的API详细对比文档，旨在帮助用户更便捷的从其他框架迁移PaddlePaddle。

## BUG修复

* 修复backward时BFS带来的精度不一致的问题
* 修复ptimizer minimize创建多余反向输入
* 修复Paddle-TRT运行显存占用大的问题
* 修复AllReduceDepPass中的Bug
* 修复FastThreadedExecutor中的Bug
* 修复Reshape、cross\_entropy、arg\_min\_max、recurrent等Op中的bug
* 修复VarBase构造的问题
* 修复了若干memory\_optimizer\_pass中的问题与bug：将复用逻辑由\>= 调整为 =，减少了因Variable复用造成的碎片，去掉了memory\_opitmize\_pass对BlockDesc的依赖，修复了不同类型的Variable会相互复用的bug
* 修复python3下使用util.plot报错问题
* 提升Profiler的稳定性并新增Memory Profile功能
* 修复C++预测必须在线程内clone，才能使多线程生效的问题
* 修复一些op在InferShape时对变长shape检查的错误
* 增加一些op对长度为零的LoD序列输入的支持
* 修复用recurrentp实现StaticRNN的一些bug
* 修复动态图dygraph模型checkpoint存储和读取的bug
