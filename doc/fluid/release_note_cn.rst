==============
Release Notes
==============

目录
##########
* 重要更新
* 基础框架
    * 安装&环境
    * 动态图Preview版
    * 性能优化
    * 显存优化
    * 执行优化
    * 框架基础功能增强
    * OP完善 
* 预测部署
    * 服务端部署库
    * Paddle Serving
    * PaddleSlim
* 分布式训练
* 模型建设
    * 图像分类 
    * PaddleDetection
    * PaddleGAN 
    * PaddleVideo         
    * PaddleNLP   
* 工具组件
* BUG修复

重要更新
##########
* 训练性能在数据读取、执行调度优化、OP计算逻辑及底层cudnn、CUDAKernel、MKLDNN等方面进行了大量优化，训练性能大幅提升；进一步优化显存占用，整体具备领先优势。
* 新增基于Padding方式实现的LSTM、GRU，更方便用户学习和使用；并基于对应API新增语言模型、seq2seq翻译模型的示例模型；增强部分OP功能，更好地支持NLP中Tensor多个维度可变的任务。
* 正式发布动态图Preview版并提供相关的API文档，并提供 7个模型动态图版本官方实现。
* 官方模型库方面正式发布PaddleDetection物体检测统一框架，覆盖主流目标检测算法，易扩展和模块化组合使用；发布图像生成库，覆盖主流的GAN算法，可一键式运行；发布PaddleNLP-Research，包含百度在 NLP 领域最新研究工作。
* 模型压缩框架PaddleSlim新增基于模拟退火的自动剪切策略和轻量级模型结构自动搜索功能（Light-NAS）。
* 分布式训练发布HighLevel API Fleet，单机转分布式训练成本显著降低；GPU多机多卡性能显著提升，在ResNet50、BERT、ERNIE等模型中4x8 v100配置下相比此前发布的Benchmark提速超过50%。
* PaddleHub新增29个预训练模型，总计覆盖文本、图像、视频三大领域40个模型，并全面提升易用性，发布PaddleHub官网。
* 发布图学习框架PGL(Paddle Graph Learning) Preview版，提供基于游走以及消息传递两种计算范式去搭建最前沿的图学习算法。


基础框架
##########
* 安装&环境
    * 增加Linux下对CUDA 10的支持，增加Windows下对CUDA 9的支持，cuDnn版本统一为7.3+
    * 安装包不按照CPU处理器是否支持AVX指令集做区分，支持自动判断并选择使用AVX指令集或不使用AVX指令集
    * 针对Python2、Python3下可能版本不兼容的依赖包限制了版本范围，以支持Python相应环境下正确安装
    * 提供可全离线安装PaddlePaddle的Docker镜像
    * 增加安装后的GPU多卡运行检测
    * 解除GPU单卡训练时对NCCL的依赖
* 动态图Preview版
    * 发布动态图相关的API和文档
    * 基础功能完善，显存和速度优化，支持GPU单机多卡训练
    * 增加transformer、ocr recognition、resnet、language model等7个模型效果对齐的动态图版本实现
* 性能优化
    * 数据读取优化
        * 使用多进程优化数据读取、预处理部分，DeepLab V3+单GPU训练获得63%的性能提升。
    * Op计算逻辑优化
        * 优化concat/spilt op输入/输出个数<=4的实现，避免1次CPU->GPU的数据传输。
        * 优化recurrent op中执行器的调用方法，修改成在迭代前调用一次executor.Prepare，迭代中executor.RunPreparedContext执行计算，从而避免每次迭代反复创建op。该优化对PaddingRNN padding small和large模型分别带来23%和15%的性能提升。
        * 融合优化器Momentum op的计算，对Resnet50单GPU、4 GPU训练分别可带来1.6%、10.6%的性能提升。
    * cuDnn使用策略优化
        * 使用cuDnn v7中新增的算法选择API cudnnGetConvolutionForwardAlgorithm_v7优化conv_cudnn op算法选择策略，Mask-RCNN和YoloV3单GPU训练分别取得32%和11%的加速。
        * 一些op的cuDnn实现慢于cuda实现，比如conv2d_transpose、pool2d（global_pooling=True）时，设置use_cudnn=False后，Cycle GAN、SE-ResNeXt单GPU训练分别获得33%、34%的性能提升。
    * Op CUDAKernel优化
        * 使用精心优化的CUDA kernel优化sum op，对多个LoDTensor求和这种情况优化效果特别明显，GPU执行获得3.3x的加速。
        * 使用2D线程Block配置优化elementwise_mul grad op，加速其CUDA Kernel中的Broadcast操作。
    * Intel CPU底层计算优化
        * 增加新的OP融合Pass（conv+relu6，conv_transpose+elementwise_add）
        * 增加新的FP32 MKLDNN kernel (FC)，INT8 MKLDNN kernel (Concat)
        * 优化若干OP，包括sequence_reverse（前向）, sequence_padding（前向）, sequence_unpad（反向），bilinear interpolate（前向）
        * 优化MKLDNN集成（如对reorder原语进行重用以减少每次创建原语的时间）
* 显存优化
    * Op层显存优化（在Transformer、Mask-RCNN等模型上显存节省1G以上）
        * 提高了inplace策略的覆盖面，支持sum、softmax、softmax_with_cross_entropy等op的inplace计算
        * 修复了dropout、conv_transpose、activation op的反向注册，降低op的显存占用
    * 显存分配与显存复用策略重构
        * 重构Allocator底层架构，为后续扩展Allocator策略提供基础
        * 重构Inplace策略，使其代码便于维护，并排除之前策略中变量可能存在误inplace、graph存在环等bug
    * 配置优化
        * 用户可通过环境变量FLAGS_conv_workspace_size_limit设置conv层的最大workspace size，单位为MB
* 执行优化
    * 更新CPU_NUM的默认配置为1，之前为设备的逻辑总核数。
    * 对Operator中OpKernel进行cache，避免每次run都重复的选择kernel。
    * ParallelExecutor执行模式（CompiledProgram.with_data_parallel())下的优化：减少同步操作；优化在num_thread=1时的速度，对于小模型的速度提升较为明显。（对于PaddingRNN small model 速度提升16%）
* 框架基础功能增强
    * build_strategy新增mkldnn_enabled_op_types选项，用户可以灵活地控制哪些op需要使用mkldnn kernel以获得加速
    * 新增ParallelExecutor下的drop_local_exe_scopes接口，可以控制什么时候清理local scope中的数据，num_iteration_per_drop_scope的设置依然有效
    * 新增自动混合精度训练接口fluid.contrib.mixed_precision.decorate()，支持图像分类、BERT等模型的训练
    * 新增fluid.gradients接口，11个操作支持做二次反向，使用于图像生成的梯度惩罚功能
    * Intel nGraph图编译引擎支持加强，增加了Bert模型所需的op支持，可以通过Intel nGraph图编译引擎进行BERT模型训练，收敛效果对齐。
* OP完善
    * 增强fused_elewise_activation op的功能，添加对x+sigmoid(y)、x+tanh(y)计算模式的支持
    * 新增指数滑动平均(Exponential Moving Average), 使模型训练更加平滑稳定
    * 新增sigmoid_focal_loss损失函数
    * 新增deformable RoI pooling操作
    * 新增deformable convolution v2操作
    * 提供unfold操作(即im2col操作)
 
预测部署
########
* 服务端部署库
    * 优化显存优化功能。DAM模型显存占用从4G下降至940M; MobileNet 模型显存占用从1G下降至500M。
    * 将Paddle-TRT的优化过程迁移到模型初始化期间，解决Paddle-TRT初次预测时间过长的问题。例如使MobileNet初次预测时间从秒级别下降至毫秒级。
    * 解决使用AnalysisPredictor从内存载入模型时，模型参数多次内存分配的问题。
    * 增强Python预测API，并在官网文档预测部署下增加Python预测API的使用说明。
    * Intel INT8 量化预测持续加强
        * 持续优化INT8量化框架（训练后量化），新增五个模型（ GoogleNet, MobileNetV2, VGG16, VGG19, ResNet101)；与FP32模型相比，精度损失均在1%以内，性能提升2～3.7倍
        * 支持QAT（训练中量化）训练出来的模型运行在INT8 kernel上，通过Pass对QAT模型进行修改，使其能运行在INT8 kernel上（目前支持 量化/去量化/卷积），在7个模型上（GoogleNet, MobileNetV1, MobileNetV2, VGG16, VGG19, ResNet50, ResNet101），和在FP32 kernel上模拟运行相比，精度变化在0.1%以内
* Paddle Serving
    * 支持GPU设备；支持多卡并行预测
    * 提供SE_ResNeXt50_32x4d模型作为标准示例，给出图像分类任务上单卡多并发、多卡多并发等场景benchmark
    * 支持大规模稀疏参数任务：用于CTR预估等场景下超大规模embedding的存储和在线访问。一期发布单机版本，支持亿级别embedding访问
    * 易于使用的API接口，API demo示例
* PaddleSlim 
    * 集成INT8量化框架
    * 新增自动剪切策略，基于模拟退火算法搜索最优剪切率：对比MobileNet V1在ImageNet 1000类分类任务上FLOPS减少50%; Top1-Accuracy=69.7%
    * 新增轻量级模型结构自动搜索功能（Light-NAS）：对比MobileNet V1在ImageNet 1000类分类任务上精度无损情况下FLOPS 减少17%
 
 
分布式训练
############
* 分布式High-Level API Fleet
    * 分布式训练统一API，支持参数服务器（Parameter Server）和Collective模式训练，大幅度降低用户从单机切换到多机训练的新增代码量
    * 用户可以通过配置分布式策略调用不同的并行训练方法，对于不同的分布式环境支持多种内建RoleMaker，方便用户调用
* 参数服务器（Parameter Server）训练新增Communicator设计
    * 独立通信逻辑到Communicator，简化异步训练逻辑
    * 提供可控制通信开关，可针对不同模型针对性调优
* GPU多机多卡增加多个提升扩展性Feature，NLP/CV经典模型下多机多卡训练提速50%
    * 新增Fused All Reduce：通过对gradient tensor进行自动合并，降低参数同步次数
    * 新增Hierachical All Reduce：层次化all reduce操作
    * 新增All Reduce通信并发能力：增加多机训练下，训练对网络波动的容忍能力
    * 新增反向与优化算法之间的依赖分析：提升通信与计算overlap并发的能力
    * 以上新增能力融合可实现在Bert Large(batch 16 x 128)和Resnet50(batch 32)上多机(v100 8*4 卡)训练速度比PaddlePaddle1.4.1提速50%+。
* GPU多机多卡Benchmark更新
    * ResNet50、VGG16、Transformer和Bert上的速度对比，并提供可复现的benchmarks脚本。
* CPU-GPU异构设备流水线并行能力支持
    * 新增流水线并行能力，可支持用户自定义在异构硬件分配计算OP，通过流水线交换数据，从而实现异构计算设备的搭配和计算资源的自由配比，提升训练速度。
    * 在IO量大、计算量较小的场景例如CTR预估，Graph Neural Network下相比纯GPU训练有明显速度优势。
 
 
模型建设（PaddlePaddle/models）
##############################
* 图像分类
    * 发布9个ImageNet预训练模型，包含ResNet50_vc, ResNet50_vd,  ResNet101_vd, ResNet152_vd, ResNet 200_vd,  ResNeXt101_64x4d, ResNeXt101_vd_64x4d, SENet154_vd, InceptionV4
    * ResNet50_vd相比已发布的ResNet50效果提升2.62%，可以达到ResNet101精度。ResNet101_vd相比已发布ResNet101效果提升1.88%
* PaddleDetection
    * 发布PaddleDetection物体检测统一框架，包含Faster-RCNN (支持FPN), Mask-RCNN (支持FPN), Cascade-RCNN, RetinaNet, Yolo v3, SSD算法，其中FPN, CascadeRCNN, RetinaNet是本次新增算法。
    * 发布一系列预训练模型，其中RCNN系列模型支持ResNet, ResNet_vd, ResNeXt, ResNeXt_vd, SEResNeXt主干网络。Yolo v3持续增加更加轻量的ResNet34, MobileNet主干网络，并发布预训练模型
* PaddleGAN
    * 发布PaddleGAN图像生成库，包含CGAN、DCGAN、CycleGAN、Pix2Pix、StarGAN、AttGAN、STGAN，支持多种数据集，支持经典的GAN网络结构。其中STGAN是百度视觉技术部自研的任意图像属性编辑模型。
* PaddleVideo
    * 优化已经发布的分类模型，NeXtVLAD训练速度提升60%， TSM速度领先竟品39%
    * 增加已发布的模型骨干网络，Nonlocal模型增加ResNet101和I3d网络结构
    * 增加动作定位模型C-TCN，百度2018年ActivityNet比赛夺冠方案
* PaddleNLP
    * BERT on PaddlePaddle：支持动态混合精度训练，保证了预训练任务在混合精度训练模式下的精度；支持以多进程的方式进行多卡任务的训练，提高了多卡加速比；优化多机分布式训练的加速比，在 V100 GPU集群上将 6 机相对于单机的 FP32 训练加速效率提高至76%
    * 发布PaddleNLP-Research，开源MRQA2019阅读理解竞赛Paddle Fluid基线、 DuConv (ACL2019) 等近期百度在 NLP 学术领域的工作
 
 
工具组件
#########
* PaddleHub
    * 全新发布PaddleHub官网，易用性全面提升
        * 新增网站http://hub.paddlepaddle.org.cn，包含PaddlePaddle生态的预训练模型使用介绍
        * 迁移学习Demo接入AI Studio与AI Book,无需安装即可快速体验
        * 新增PaddleHub后端服务，支持模型检索、下载、私有化部署等功能
    * 新增29个预训练模型，覆盖文本、图像、视频三大领域；目前官方提供40个预训练模型
        * CV预训练模型
            * 新增图像分类预训练模型11个：SE_ResNeXt, GoogleNet, ShuffleNet等
            * 新增目标检测模型Faster-RCNN和YOLOv3
            * 新增图像生成模型CycleGAN
            * 新增人脸检测模型Pyramidbox
            * 新增视频分类模型4个: TSN, TSM, StNet, Non-Local
        * NLP预训练模型
            * 新增语义模型ELMo
            * 新增情感分析模型3个: Senta-BOW, Senta-CNN, Senta-GRNN
            * 新增中文情绪识别模型EmoTect
            * 新增中文语义相似度分析模型Simnet
            * 升级LAC词法分析模型，新增词典干预功能，支持用户自定义分词
    * Fine-tune API升级，灵活性与性能全面提升
        * 支持多卡并行、PyReader多线程IO，ERNIE文本分类Fine-tune速度提升60%
        * 简化finetune、evaluate、predict等使用逻辑，提升易用性
        * 增加事件回调功能，方便用户快速实现自定义迁移学习任务
        * 新增多标签分类Fine-tune任务
* 图学习框架 `PGL <https://github.com/PaddlePaddle/PGL>`_  (Paddle Graph Learning) 
    * 发布基于PaddlePaddle的图学习框架PGL Preview版，提供基于游走 (Walk Based) 以及消息传递（Message Passing）两种计算范式去搭建最前沿的图学习算法，如图表征学习、图神经网络等。PGL充分利用Paddle LoD Tensor特性大幅提升Message-Passing范式中信息聚合效率，兼顾了灵活性和高效性
        * 新增基于PGL实现的GCN、GAT，在多个数据集达到SOTA水平
        * 新增基于大规模子图采样模型Graphsage模型，单机可支持5千万节点、20亿条边的巨图
        * 新增node2vec，deepwalk等图表征学习方法，达到SOTA水平
        * 新增PGL文档、API、Tutorial等材料 

BUG修复
##########
* 修复softmax_with_cross_entropy操作CPU版本中ignore_label不支持在0到类别数之外label的问题
* 修复import paddle之后logging.basicConfig设置失效问题
* 修复python/paddle/fluid/layers/ops.py在python3下报错的问题
* 修复sequence unpad op在训练过程中不稳定的问题
* 修复Concat Op属性axis为负数时挂掉的问题
* 修复了enable_inplace和memory_optimize的潜在bug，保证某些op的输出变量不会被错误地复用
* 修复了Eager Deletion策略可能会提前误删变量存储空间的bug，提高Eager Deletion策略的稳定性
* 修复了模型图分析中拓扑排序存在bug导致的在相同模型输入情况下有不同的模型图生成的情况
* 修复了预测结束后其他服务线程OMP线程冲突的问题。修复为在CPU模式下，预测引擎会在预测结束后将全局的OMP线程数设回为1。
