
Release Notes
==============
##  重要更新

本版本对框架功能层面进行了重点增强，预测部署能力全面提升，分布式训练发布PLSC支持超大规模分类，并对参数服务器模式进行优化整合。对编译选项、编译依赖以及代码库进行了全面清理优化。模型库持续完善，优化了整体层次结构，增加了动态图模型实现。端到端开发套件和工具组件进一步完善。

**训练框架**：增加自动混合精度训练AMP接口和新控制流接口；优化Tensor使用方式和显存分配策略；新增支持Nvidia DALI GPU数据预处理库；持续优化基础OP的功能和性能；动态图的功能进一步完善，性能大幅提升，对data independent的动态图模型提供转为静态图可预测部署模型的功能；框架调试分析功能和易用性全面提升。

**预测部署**：服务器端预测库的Python API大幅优化，新增R语言、Go语言调用预测库的使用方法和示例，强化了量化支持能力；Paddle Lite支持无校准数据的训练后量化方法生成的模型，加强对OpenCL的支持，支持昆仑XPU的预测；模型压缩库PaddleSlim重构裁剪、量化、蒸馏、搜索接口，与模型库充分打通，新增大规模可扩展知识蒸馏框架 Pantheon。

**分布式训练**：参数服务器模式下针对transpiler半异步、全异步、GEO三种模式，后端实现上统一到communicator中，前端接口统一到fleet中，通过fleet strategy灵活选择不同模式；发布大规模分类库PLSC，通过模型并行支持超多类别的分类任务。

**基础模型库**：发布语音合成库Parakeet，包括多个前沿合成算法；PaddleCV新增14个图像分类预训练模型，3D和跟踪方向模型持续丰富；PaddleNLP的分词和词性标注模型支持jieba分词；PaddleRec增加多任务模型MMoE。模型库整体增加了广泛的动态图模型实现。模型库整体层次结构做了调整优化。

**端到端开发套件**：PaddleDetection和PaddleSeg新增大量模型实现及预训练模型，典型模型的训练速度和精度提升，模型压缩和部署能力大幅提升，使用体验全面优化。发布ElasticRec推荐排序系统，通过K8S进行部署，支持流式训练和在线预测服务。

**工具组件**：PaddleHub新增52个预训练模型，总数超过100，功能和体验持续优化；多任务学习框架PALM升级内核，开放API调用，支持更多的任务类型；联邦学习PaddleFL新增公开数据集。深度强化学习框架PARL和飞桨图学习框架PGL也对应版本升级，支持更多功能，开放更多算法和基线。



## 训练框架

- API
    - 增加自动混合精度训练AMP接口：能以通用的方式把一个网络转成混合精度训练，同时保证精度波动在正常范围内
    - 增加新的控制流接口并推荐使用：新增while_loop（循环控制功能）、cond（条件分支功能）、case和switch_case（分支控制功能）4个控制流OP，更加易用，且支持如下新增功能：
        - 支持使用python callable作为控制条件或执行体
        - 支持控制流中的不同分支使用不同loss或optimizer
        - 支持控制流中的condition部分使用CPU数据或GPU数据
    - 部分API参数支持使用变量列表：针对部分API的parameter_list或no_grad_set参数只支持使用字符串列表的情况，增加对变量列表的支持，使用如下API时不再需要提前获取相关变量的name属性：
        - fluid.backward.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)
        - fluid.backward.gradients(targets, inputs, target_gradients=None, no_grad_set=None)
        - 各种Optimizer的minimize方法，如Adam的minimize：minimize(loss, startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None)
- 基础功能优化
    - 支持使用numpy的float16类型设置Tensor数据，无需先转换为uint16类型。
    - 支持直接使用负号，得到Tensor的相反数。
    - 显存分配策略：
        - 默认策略变为AutoGrowth：在不影响训练速度的情况下，按需申请显存。规避之前的默认显存预分配策略下难以在同一张GPU卡上再起新任务的问题。
        - 多卡任务显存分配调整：将不同GPU卡上的显存分配器设置为Lazy初始化的方式。若用户不使用某张卡，则不会在该卡上申请显存。避免当其他GPU卡上有显存占用时，在空闲GPU卡上跑任务若不设置CUDA_VISIBLE_DEVICES导致显存OOM的问题。
    - OP功能升级
        - elu：该激活函数支持计算二阶梯度。
        - prroi_pool：rois参数可以接受Tensor或LoDTensor类型。
        - conv2d，pool2d，batch_norm，lrn：反向计算全部支持使用MKL-DNN高性能计算库。
        - argsort：支持降序排序（新增descending参数，默认值False）。
- 基础性能优化
    - DALI预处理加速
        - 增加对Nvidia DALI GPU数据预处理库的支持，可用于加速图片，视频，语音等数据预处理。
    - 自动混合精度训练优化
        - 实现如下优化策略，并配合DALI数据预处理，ResNet50模型训练吞吐大幅提升：V100单卡混合精度训练吞吐从600+ images/sec提升到1000+ images/sec；单机8卡吞吐达到7840 image/sec，4机32卡吞吐达到28594 images/sec。
            - 增加batch_norm和conv2d等op对NHWC数据布局输入的支持，以使用Tensor Core加速fp16计算或减少访存耗时。
            - 基于IR Pass机制对模型中的部分op pattern进行融合，如batch_norm和relu等。
            - 优化elementwise(add,mul)等op的计算kernel。
    - 优化RecomputeOptimizer提升batchsize, 在Bert-large模型上最大batchsize比不使用RecomputeOptimizer增大533.62%，比上一版本提升一倍。
    - OP性能优化
        - 实现embedding和sequence_pool的融合算子fuse_emb_seq_pool，优化bloom_filter中的murmurhash3_x64_128，有效提升部分NLP模型的训练速度。
        - 优化了mean op的GPU性能，输入数据为32\*32\*8\*8的Tensor时，前向计算速度提升2.7倍。
        - 优化assign、lod_reset op，避免不需要的显存拷贝和data transform。
        - 优化了stack OP的kernel实现，XLnet/Ernie模型GPU单卡性能提升4.1%。
- 动态图
    - 功能优化
        - 移除了动态图Layers 中的 name_scope 参数，使得用户更方便继承和调用。
        - 移除to_variable接口中的block参数，简化了API的使用。
        - 针对模型参数依赖数据的问题，移除了 build_once设计，使得Layers在 **init** 执行完成之后就可以获取到所有的参数表，方便save load、参数初始化、参数debug、参数优化等。
        - 完善自动剪枝，方便用户组网并减少反向计算量。
        - 支持 SelectedRows 操作，使 Embedding 层支持单卡的稀疏更新。
        - 针对框架缺少容器类的问题，新增ParameterList、LayerList、Sequencial功能，方便用户组网。
        - 支持named_sublayers、named_parameters功能，方便用户编程。
        - 支持Linear lr warmup decay策略。
    - 性能优化
        - 优化了python 与c++ 交互，GradMaker、OperatorBase、allocator等。基于LSTM的语言模型任务p在P40机器上性能提升提升270%。
        - 针对optimize中多次调用optimized_guard无用代码导致的性能问题，移除了冗余代码。Transformer模型（batch_size=64）在P40机器上，SGD、Adam等优化器有5%~8%%的性能提升。
        - 针对AdamOptimizer中额外添加scale_op更新beta参数对性能的影响，将beta更新逻辑融合到adam_op中，减少op kernel调用开销。Dialogue-PLATO模型P40机器上性能提升9.67%。
        - 优化动态图异步DataLoader，对于Mnist、ResNet等CV模型任务在P40机器上单卡训练速度提升超过40%。
        - 新增numpy bridge功能，支持在cpu模式下Tensor和ndarray之间共享底层数据，避免创建Variable时numpy输入需要拷贝的问题，提升效率。
        - 显存优化：提前删除反向不需要Tensor Buffer的前向变量空间的优化策略，在ResNet等模型上最大batch size提升20%-30%以上。
    - 动态图部署
        - 支持TracedLayer接口，实现 data independent的动态图模型转为静态图可预测部署的模型。
- 调试分析
    - 报错信息优化 ：对框架报错信息整体归类，实现报错信息的体系化，同时完成文案优化，帮助用户更快速、准确的定位和解决问题。
    - 优化性能分析profile 功能
        - 增强profiler的功能和准确性，支持不同级别的profile选项，能够在profile数据中记录事件的调用关系并打印出来。
    - 优化nan inf检查调试（通过FLAGS_check_nan_inf生效），性能、功能及输出信息均有较大提升：
        - 速度上，v100测试ResNet50模型相比原工具组件约有1000倍性能提升，保持正常训练80%以上的效率。
        - 功能上，增加fp16的支持，可设置环境变量跳过op、op_role、op_var的检查，方便fp16模型的调试。
        - 输出信息更加翔实，除出错的op及tensor名称外，还会打印出错的nan、inf及正常数值的数量以便于调试。
- 发布cpu训练和预测的轻量级安装包paddlepaddle-tiny，支持window/linux/Mac操作系统以及python27/python35/python36/python37：
    - 编译选项：no avx, no ml, no gpu, no unittest
    - 裁剪掉slim和部分dataset。
    - linux包体积从90M减小到37M；windows包体积从50.8M减小到9.6M；mac包体积从59M减小到19.8M。
    - 安装requirements依赖从15个减小到7个。

## 预测部署

- 服务器端预测库
    - Python API
        - 支持从内存读写模型，以满足模型加密的需求。
        - 不再在预测模型最后添加 Scale 算子。
        - 新增ZeroCopy API，与C++接口基本一致，支持以numpy.ndarray作为输入和输出，在Python端使用更加方便。
        - 在AnalysisConfig中增加多个接口，完整覆盖C++预测的功能，包括删除pass、禁用预测glog等。
    - 其他编程语言的支持
        - 新增R语言、Go语言调用预测库的使用方法和示例
    - 对外提供 ProtoBuf 对应的头文件，方便用户解析模型结构的需求。
    - 带TRT编译的预测库不再从thrid_party中提供TensorRT库，需要用户自行到https://developer.nvidia.com/tensorrt 下载
    - 功能增强：
        - 打通Paddle Lite以子图方式接入，已验证 ResNet50。
        - 新增MKL-DNN FC INT8 kernel的支持
        - Paddle-TensorRT支持Ernie模型，Ernie模型（seq length=128） 在T4卡上fp16预测速度为3.6ms, 比fp32加速37%。
        - 量化：ERNIE INT8精度相比于FP32 精度略有下降，但其在第二代至强可扩展平台6271上单线程性能优化提升2.70倍，多线程性能提升1.79倍
- 移动/嵌入式端Paddle Lite（https://github.com/PaddlePaddle/Paddle-Lite）
    - 对应发布v2.3版本。
    - model_optimize_tool多项功能升级。
    - 支持“无校准数据的训练后量化方法”，模型存储空间可减少2~4倍。
    - OpenCL：完成30个Image2D Kernel迁移，涵盖14个OP。
    - 对FPGA、NPU的支持进一步加强；支持昆仑XPU的预测。
    - 发布全新官网文档；新增“无校准数据的训练后量化方法”使用文档。
- Paddle Serving（https://github.com/PaddlePaddle/Serving）：
    - 发布bert类语义理解模型的远程文本向量表示预测服务。
    - 发布了paddle-gpu-serving whl包，通过pip安装和Python代码即可部署和使用预测服务;
    - 支持Paddlehub中的13种语义理解模型，支持单机多卡，使用Ernie_tiny模型在单张P4 GPU下平均样本长度为7时预测速度为869.56样本每秒。
- PaddleSlim（https://github.com/PaddlePaddle/PaddleSlim）：
    - 拆分PaddleSlim为独立repo。
    - 重构裁剪、量化、蒸馏、搜索接口，对用户开放底层接口。
        - 量化:
            - 新增基于KL散度的离线量化功能，支持对Embedding层量化。
            - 新增对FC的QAT MKL-DNN量化策略支持
            - 新增PostTrainingQuantization，完整实现训练后量化功能：支持量化30种OP，支持灵活设置需要量化的OP。
            - 量化训练支持设定需要量化的OP类型。
        - 裁剪: 重构剪裁实现，方便扩展支持更多类型的网络。
        - 网络结构搜索:
            - 支持SA搜索，增加更多的搜索空间，支持用户自定义搜索空间。
            - 新增one-shot搜索算法，搜索速度比上个版本快20倍。
    - 新增大规模可扩展知识蒸馏框架 Pantheon
        - student 与 teacher 、teacher与 teacher 模型之间充分解耦，可分别独立运行在不同的物理设备上，便于充分利用计算资源；
        - 支持 teacher 模型的单节点多设备大规模预测，在 BERT 等模型上测试加速比达到线性；
        - 用 TCP/IP 协议实现在线蒸馏模式的通信，支持在同一网络环境下，运行在任意两个物理设备上的 teacher 模型和 student 模型之间进行知识传输；
        - 统一在线和离线两种蒸馏模式的 API 接口，不同的 teacher 模型可以工作在不同的模式下；
        - 在 student 端自动完成知识的归并与知识数据的 batch 重组，便于多 teacher 模型的知识融合。
    - 模型库:
        - 发布ResNet50、MobileNet模型的压缩benchmark
        - 打通检测库，并发布YOLOv3系列模型的压缩benchmark
        - 打通分割库，并发布Deepabv3+系列分割模型的压缩benchmark
    - 完善文档：
        - 补充API文档；新增入门教程和高级教程；增加ModelZoo文档，覆盖分类、检测、分割任务。所有文档包含中、英文。

## 分布式

- 参数服务器模式：
    - 大幅降低训练过程中的内存占用，在1亿规模embedding任务上，Trainer端内存可以降低90%
    - 大幅降低分布式保存模型、加载模型的内存占用， Pserver端内存峰值最大可降低为原先的$1/N，N$为Pserver节点个数。
    - 优化GEO模式 稠密参数通信
    - 支持分布式AUC指标计算
    - 新增分布式Barrier功能
    - 非Fleet的transpiler API加入过期警示， 该API计划在下一个版本中移除
    - Communicator加入半异步模式
    - TrainFromDataset训练接口支持半异步模式
    - Fleet加入DistributedStrategy， 进一步提升分布式易用性， 整合目前分布式相关FLAG
    - Fleet pslib模式支持一个program多loss训练，优化训练性能
    - 千亿稀疏模式支持k8s环境。
- 大规模分类库PLSC：支持受限于显存容量数据并行无法处理的大规模分类问题（https://github.com/PaddlePaddle/PLSC）
    - 内建ResNet50、ResNet101和ResNet152三种模型，并支持自定义模型；单机8张V100 GPU配置下，ResNet50模型百万类别训练速度2,122.56 images/s，相比标准ResNet50模型加速倍1.3倍；
    - 发布模型在线预测服务plsc-serving whl包，预测人脸识别模型的图片语义向量表示，支持使用用户训练的模型进行预测。ResNet50模型（batch size=256）在单张V100 GPU下预测速度为523.47 images/s；
    - 发布基于ResNet50网络和MS1M-ArcFace数据集的预训练模型：https://plsc.bj.bcebos.com/pretrained_model/resnet50_distarcface_ms1mv2.tar.gz。
- 发布ResNet50混合精度训练benchmark（单卡、多卡、多机）。

## 基础模型库
（https://github.com/PaddlePaddle/models）

- PaddleNLP
    - seq2seq支持RL和GAN等训练模式
    - 发布分词和词性标注训练模型，利用知识蒸馏框架 Pantheon，在自有数据集上比PaddleNLP上LAC上F1值提升1%；合入jieba分词，通过加入use_paddle标签来开启深度学习模型模式；并在在jieba加入paddle版本检测和回退机制，保障用户体验。
    - 增加动态图模型实现：word2vec、senta、transformer、bert、seq2seq、LAC。
- PaddleSpeech
    - 发布语音合成库Parakeet (Paddle PARAllel text-to-speech toolkit)
        - 实现语音合成模型数据预处理、训练和合成等的标准工作流
        - 提供对常见数据集的开箱即用的预处理实现
        - 提供语音合成领域常用模型组件，为实现模型提供支持
        - 发布语音合成模型 DeepVoice3、ClarinNet 、TransformerTTS、FastSpeech、WaveNet、WaveFlow

- PaddleCV
    - 图像分类:
        - 新增预训练模型SENet-vd、Res2Net、HRNet系列模型总共14个：
            - SE_ResNet18_vd，SE_ResNet34_vd，SE_ResNeXt50_vd_32x4d，ResNeXt152_vd_32x4d
            - Res2Net50_26w_4s，Res2Net50_14w_8s，Res2Net50_vd_26w_4s
            - HRNet_W18_C，HRNet_W30_C，HRNet_W32_C，HRNet_W40_C，HRNet_W44_C，HRNet_W48_C，HRNet_W64_C  
        - 支持使用DALI加速数据预处理，在ImageNet训练上获得1.5倍(ResNet50) 至3倍以上(ShuffleNet)加速，并大幅提升GPU利用率。
    - 3D方向:
        - 发布模型PointNet++、PointRCNN。
    - 跟踪模型库 :
         - 发布模型SiamFC、ATOM。
    - 增加动态图模型实现: MobileNet-v1/v2、YOLOv3、FasterRCNN、MaskRCNN、视频分类TSM模型、视频动作定位BMN模型。

- PaddleRec
    - 发布推荐领域多任务模型MMoE, 适用于工业界大规模多任务联合训练。
    - 增加动态图模型实现：gru4rec、deepfm。

## 端到端开发套件

- PaddleDetection（https://github.com/PaddlePaddle/PaddleDetection）
    - 进一步提升YOLOv3模型精度，COCO数据上精度达到43.2%，相比上个版本绝对提升1.4%。
    - 新增模型实现及预训练模型:
        - 新增Google AI Open Images 2019-Object Detction比赛中的最佳单模型CascadeCARCNN-FPN-Dcnv2-Nonlocal ResNet200-vd，同时也发布此算法基于Objects365数据的预训练模型。
        - 新增backbone为CBResNet、Res2Net、HRNet的系列预训练模型。
        - 新增LibraRCNN算法及预训练模型。
        - FasterRCNN R50 FPN模型新增基于GIoU、DIoU、CIoU loss的预训练模型，不降低预测速度的情况下，在COCO数据上精度分别提升1.1%，0.9%，1.3%。
    - 新增模块:
        - 主干网络: 新增CBResNet、Res2Net、HRNet。
        - Loss模块: 新增GIoU loss、 DIoU loss、CIoU loss，以及Libra loss，YOLOv3的loss支持细粒度op组合。
        - 后处理模块: 新增softnms，DIOU nms模块。
        - 正则模块: 新增DropBlock模块。  
    - 功能优化和改进:
        - 加速YOLOv3数据预处理，整体训练提速40%。
        - 优化数据预处理逻辑。
        - 增加人脸检测预测benchmark数据。
        - 增加Paddle预测库Python API下的预测示例。
    - 检测模型压缩 :
        - 裁剪: 发布MobileNet-YOLOv3裁剪方案和模型，在VOC数据集上FLOPs - 69.6%, mAP + 1.4%，在COCO数据集上FLOPS-28.8%, mAP + 0.9%; 发布ResNet50vd-dcn-YOLOv3裁剪方案和模型，在COCO数据集上FLOPS - 18.4%, mAP + 0.8%。
        - 蒸馏: 发布MobileNet-YOLOv3蒸馏方案和模型，在VOC数据上mAP + 2.8%，在COCO数据上mAP + 2.1%。
        - 量化: 发布YOLOv3和BlazeFace的量化模型。
        - 裁剪+蒸馏: 发布MobileNet-YOLOv3裁剪+蒸馏方案和模型，在COCO数据集上FLOPS - 69.6%，GPU下预测加速64.5%，mAP - 0.3 %; 发布ResNet50vd-dcn-YOLOv3裁剪+蒸馏方案和模型，基于COCO数据FLOPS - 43.7%，GPU下预测加速24.0%，mAP + 0.6 %。
        - 搜索: 开源BlazeFace-Nas的完整搜索方案。
    - 预测部署:
        - 适配Paddle预测库对TensorRT的支持、对FP16精度的支持。
    - 文档:
        - 新增数据预处理模块介绍文档、实现自定义数据Reader的文档。
        - 新增如何新增算法模型的文档。
        - 文档部署到网站: https://paddledetection.readthedocs.io/zh/latest/  

- PaddleSeg（https://github.com/PaddlePaddle/PaddleSeg）
    - 新增模型
        - 适用于车道线分割场景的LaneNet模型。  
        - 适用于轻量级Fast-SCNN模型。  
        - 适用于高精度场景的HRNet语义分割模型 。
    - 发布基于PaddleSlim的多种模型压缩方案:
        - 基于Cityscape的Fast-SCNN裁剪方案和模型。
        - 基于Cityscape的Deeplabv3p-Xception和Deeplabv3p-MobilenetV2蒸馏方案。
        - 基于Cityscape的Deeplabv3p-MobilenetV2搜索方案。
        - 基于Cityscape的Deeplabv3p-Mobilenet量化方案和模型。  
    - 预测部署能力提升
        - 新增Python轻量级部署。
        - 新增对 FP16、Int8量化模型的TensorRT预测加速支持。
        - 新增DeepLabv3p-MobileNetV2的人像分割Paddle-Lite移动端部署教程和案例。
        - 优化模型导出环节，支持图像预处理和后处理的GPU化，性能提升10%~20%。
        - 提供U-Net, ICNet, PSPNet, DeepLabv3+等模型的在不同尺寸图像的预测性能Benchmark，便于用户根据性能进行模型选型。  
    - 体验优化  
        - 新增学习率warmup功能，支持与不同的学习率Decay策略配合使用，提升Fine-tuning的稳定性。
        - 支持对标注图使用伪彩色图像格式的保存，提升标注图片的预览体验。  
        - 新增自动保存mIoU最优模型的功能。  
        - 全面优化文档逻辑，提供如工业质检、眼底筛查等工业场景的AIStudio实战教程。

- ElasticRec（https://github.com/PaddlePaddle/ElasticRec）
    - 发布ElasticRec推荐排序系统，通过K8S进行部署，支持流式训练和在线预测服务。

## 工具组件

- PaddleHub（https://github.com/PaddlePaddle/PaddleHub）
    - 预训练模型丰富，新增52个预训练模型，目前预训练模型总数100+：
        - 语义模型：新增RoBERTa_wwm、BERT_wwm、ERNIE-Tiny等5个语义模型
        - 文本分类：新增黄反鉴别模型3个。
        - 图像分类：新增ResNext-WSL、EfficientNet等共36个图像分类模型。
        - 目标检测：新增行人检测，车辆检测等共5个检测模型。
        - 关键点检测：新增人脸关键点检测和人体姿态关键点检测模型2个。
        - 人脸口罩检测：新增基于PyramidBox-Lite的人脸口罩检测模型2个。
        - 通用人脸检测：新增Ultra Light Fast Generic Face Detector、PyramidBox-Lite等通用人脸检测模型4个。
    - 功能:
        - 新增基于Paddle Serving的Bert Service文本向量表示服务。
        - Task灵活性增强，新增Hook机制可以支持用户自定义代码加载。
        - 新增彩色Colorlog，修复日志重复打印问题。
        - 优化代码结果，命令行执行速度提升50% 。
        - 重构Dataset、Reader，适配自定义数据集代码量降低60%。
        - 优化AutoFinetune接口，支持多实验的可视化效果显示。
    - 体验优化
        - 逻辑全面优化，新增丰富的AIStudio教程内容。
        - 官网落地页全新升级，提供在线快速体验和教程指导的功能。

- 多任务学习框架PALM（https://github.com/PaddlePaddle/PALM）
    - 支持python3和windows
    - 升级框架内核和多任务底层机制，开放API调用
        - 灵活的模型保存机制，支持单任务保存和全图保存
        - 支持连续训练和连续预测，单次执行下可自由切换数据集文件
        - 新增模型定制化/自定义功能
        - 重构多任务底层kernel，修复若干影响通用性和稳定性的bugs
    - 强化多任务学习能力
        - 支持多任务场景下每个任务有不同的batch size和sequence length
        - 修复了多任务多卡训练时，各个显卡上任务不一致的问题
        - 优化了多任务学习调度和终止策略，普遍提升模型泛化能力
    - 强化支持的任务的功能和类型
        - 匹配任务支持增强，支持pairwise learning和多类别（如NLI句子关系判断）。
        - 机器阅读理解任务支持增强，新增用户可控的预处理超参数。
        - 新增支持序列标注任务。
    - 强化大规模训练/推理能力
        - 新增自动多卡预测能力
        - 重构异步reader，多卡场景下支持变长padding
    - 新增预训练模型管理和下载模块
        - 支持BERT、ERNIE、RoBERTa等各预训练模型的管理和下载
        - 新增RoBERTa中文预训练模型

- 联邦学习PaddleFL（https://github.com/PaddlePaddle/PaddleFL）：
    - 新增scheduler与submitter功能：scheduler可用于在训练过程中控制trainer是否参加更新 。submitter可用于完成在MPI集群提交paddleFL任务的功能
    - 新增LEAF dataset联邦学习公开数据集，并添加api，用于设置benchmark。支持图像分类，情感分析，字符预测等领域的经典数据集，如MNIST，Sentiment140
    - 根据新增组件，在example中修改了原有的样例，并添加了femnist_demo, submitter_demo样例
    - 优化fl_distribute_transpiler，使FedAvg strategy新增对adam optimizer支持；
    - 新增SecAgg strategy（Secure Aggregation），用于实现安全的参数聚合；

- 深度强化学习框架PARL（https://github.com/PaddlePaddle/PARL）
    - 发布v1.3版。
    - 新增对Multi-Agent RL算法支持，包括MADDPG。
    - 新增对多卡训练的支持，发布多卡DQN算法示例。
    - 开源连续控制领域的SOTA算法TD3和SAC。
    - 开源NeurIPS2019强化学习挑战赛事冠军模型实现和训练方案，开放训练好的模型（可考虑公开课）
- 飞桨图学习框架PGL（https://github.com/PaddlePaddle/PGL）
    - 发布v1.1版：
    - 新增对权威图学习数据集OGB的支持，全面支持nodepropered、linkpred、graphpropered三大类型任务，并发布SOTA基线。
    - 发布图推荐解决方案PGL-Rec和知识图嵌入算法集PGL-KE。
    - 易用化改进，发布PGL高阶API。
    - 其他升级点：多进程图采样优化，加速GraphSAGE类模型3倍；新增基于Lod Tensor的Graph Batch算子，Graph Pooling算子；Model Zoo新增模型，包括分布式异构图算法、GraphZoom、PinSage等。

## 代码重构和升级

- 编译
    - 增加WITH_NCCL编译选项，单卡用户可显示指定WITH_NCCL=OFF加速编译。
    - 新增编译选项WITH_TP_CACHE，缓存第三方源码，避免重复下载，Windows用户可将其设置为ON，加快编译速度并提高编译稳定性。
    - `CUDA_ARCH_NAME`默认值设成`Auto`(`All`表示编译所有gpu架构，`Auto`表示只编译当前机器gpu架构），对开发者来说，使用`Auto`比`All`节省非常多的编译时间，提高开发效率。
    - 减少了冗余的link环节与产物、多余的文件拷贝，加快了Windows下的编译速度。
- 外部依赖库
    - 升级MKL-DNN到最新1.1版本。
    - 将预测库与`third_party` 解耦，重构了28个第三方依赖的编译代码，便于统一管理外部依赖。
    - 移除了第三方依赖的私人仓库2个、无用依赖1个、无用的patch下代码2000+行，提高仓库质量。
- 代码清理、重构和优化
    - 去掉无用的`contrib/float16`目录，删除BRPC下无用的snappy/snappystream依赖。
    - 从 `python/paddle/fluid/layers/nn.py`中，根据API功能拆出`loss.py`和`sequence_lod.py`，减少`nn.py`的代码量，便于阅读。
    - 修复`-Wno-error=sign-compare`的warning对应的代码（共100多处），后续所有该类warning会在编译时报错，提高代码质量
    - 去掉WindowsMSVC编译的`WarningLnk4006/WarningLnk4221`（共约300处），提高仓库质量。
    - 减少reduce_op, expand_op, expand_as_op模版类数量，加速GPU编译和减少whl包70M的空间。
    - 动态图下通过代码自动生成每个OP的pybind函数，用于在layers中直接调用，提高动态图性能并减少与静态图的耦合度。

## BUG修复

- 修复基于PaddleDetection的 Faster-RCNN使用Python API预测时MKL-DNN报错问题。
- 修复sum op的GPU实现中，由于部分Tensor没有初始化引起训练挂掉的问题。
- 修复fill_constant中，value设置为大整数时精度损失的问题。
- 修复softmax_with_cross_entropy_op在CUDA上的精度不一致问题。
- 修复clone program时program中的stop_gradient属性不能拷贝到新program的问题。
- 修复elementwise_pow op在整数上的精度损失问题。
- 修复一些 GFLAGS 不能在预测库外进行指定的问题。
- 修复 Analysistor 多线程下若干 Pass 导致预测随机 core 的问题。（fc_gru_fuse_pass，seqconv_eltadd_relu_fuse_pass，attention_lstm_fuse_pass，embedding_fc_lstm_fuse_pass，fc_lstm_fuse_pass，seq_concat_fc_fuse_pass）
- 修复了在使用 NativePredictor 指定使用 CPU 预测后，在同一进程内使用 AnalysisConfig 指定 GPU 不生效的错误。
- 修复Windows上-DWITH_MKL=OFF时编译报错的bug。
- 修复py_func OP无法输入tuple(Variable) 的bug，新增如何写PythonOP的代码示例。
- 修复sigmoid cudnn kernel错调用成tanh cudnn kernel的问题。
- 修复部分动态图模式下reshape、Conv2D相关的bug；修复网络中部分参数无梯度，导致程序crash 的bug。
- 修复GradientClip在参数服务器模式下运行错误的BUG。
- 修复参数服务器全异步模式下内存泄露的问题。
