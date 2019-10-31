==============
Release Notes
==============

目录
##########
* 重要更新
* 用户体验提升
    * 编程易用性提升
    * 默认配置项
    * 接口优化
    * 报错信息优化
    * 文档优化
    * 编译优化
    * Windows支持增强
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
* 用户体验和易用性专项提升，包含全面的文档优化、报错信息优化、配置项优化、接口优化、编译优化、多平台支持以及编程易用性提升等各方面。
* 训练框架进一步优化了速度，完善了显存优化机制，并支持在框架外部自定义C++/CUDA OP。新增了大量OP,并从多个维度优化了大量存量OP，包括兼容性、行为一致性、功能提升等方面。
* 分布式训练新增LocalSGD、GEO-SGD等策略，大规模同步训练、异步训练速度继续提升，并支持K8S + Volcano任务提交。
* 部署能力强化
    * 服务器端预测库增加C API，并支持版本兼容检查，实现了大量性能优化工作。
    * 发布PaddleLite，定位高性能、多平台、轻量化的端侧预测引擎，并可作为服务器端预测库的加速库。
    * PaddleServing新增超大规模分布式预估服务能力。
    * PaddleSlim强化了量化训练功能，增加了基于硬件的小模型搜索功能。
* 模型库易用性和丰富度提升
    * PaddleNLP，发布全新seq2seq相关API和文本生成模型样例。语义表示库新增XLNet预训练模型；开源EMNLP2019阅读理解竞赛冠军模型D-NET，同时支持18个不同抽取式阅读理解数据集打榜。发布飞桨多任务学习库PALM （PAddLe Multi-task learning），更便捷支持多任务机器学习调研。
    * PaddleCV，发布训练部署端到端的图像分割库PaddleSeg。图像分类新增EfficientNet等43个预训练模型。PaddleDetection新增2019 Objects365 Full Track冠军模型、BlazeFace等人脸检测小模型，行人检测和车辆检测的预训练模型。PaddleVedio新增ActivityNet Challenge 2019夺冠模型，扩展包含video caption、video grounding等模型。
    * 发布PaddleSpeech，包含语音识别模型DeepSpeech和语音合成模型 DeepVoice3 ；
    * 增加PaddleRec的模型覆盖
* 配套工具组件全面升级：
    * PaddleHub新增超参优化Auto Fine-tune功能，并全面提升Fine-tune功能的灵活性和易用性，预训练模型数量大幅增加。
    * 飞桨图学习框架PGL正式版发布，易用性、规模性、丰富性全面提升。
    * 飞桨深度强化学习框架PARL并行能力进一步提升，支持进化算法。
    * Paddle2ONNX和X2Paddle全面升级，飞桨和其他框架的模型互转更加方便。
    * 发布飞桨联邦学习框架PaddleFL


用户体验提升
#########
* 编程易用性提升
    * fetch变量便利化：针对以往基于变量重命名的存储优化策略必须要求fetch变量设置persistable = True的bug，重构了Inplace复用和跨Operator复用策略，不再强制要求fetch变量必须设置persistable=True，且不会改变任何变量的名称，且均能保证结果的正确性。
    * optimizer.minimize和其他接口调用的位置敏感性问题 
针对用户搭建网络时易将exe.run(startup_program)置于optimizer.minimize之后执行，从而导致不明报错的问题，在Optimizer类Op中增加了初始化检查以及易于理解的提示信息，使再出现此类问题时用户能够快速定位错误。
    * 针对用户搭建网络时易将test_program = main_program.clone(for_test=True)置于optimizer.minimize之后执行，从而导致模型测试结果错误的问题，增加了prune_backward接口对在minimize之后clone的test_program进行反向部分的裁剪，使test_program的clone操作的正确执行不再依赖于optimizer.minimize的先后关系。
* 默认配置项
    * 显存Garbage collection开关默认打开（对应FLAGS_eager_delete_tensor_gb环境变量=0）。
    * build_strategy的选项： 
        * build_strategy.enable_inplaceinplace策略默认打开。这样显存Garbage collection策略和inplace策略全默认打开，默认策略即已验证过的最优策略。
        * build_strategy.memory_optimize跨Op显存复用优化策略的默认行为调整为：在Garbage collection策略打开时默认关闭（规避两者合用会比只用Garbage collection策略效果差的问题）；而在Garbage Collection策略关闭时默认打开规避两者合用会比只用Garbage collection策略效果差的问题。用户可显式设置build_strategy.memory_optimize = True/False强制打开或关闭跨op显存复用优化策略。
        * 提升了一些速度优化策略的普适性，将fuse_all_reduce_ops、fuse_broadcast_ops 选项默认打开，可以减少计算图中的计算节点个数，进而加速计算图执行。
    * execution_strategy选项: 
        * 将num_iteration_per_drop_scope默认值从1改成100，每次迭代之后都要进行一次同步操作，提升速度。
* 接口优化
    * 针对Python存储优化接口paddle.fluid.memory_optimize优化效果欠佳、不稳定等问题，彻底废弃了此接口，此版本后该接口不会对用户网络进行任何优化，并可能在后续版本中彻底移除，建议用户删除代码中的paddle.fluid.memory_optimize调用。
    * 统一DataLoader接口。针对以往Reader接口繁多、名称晦涩难懂等问题，统一了PyReader和Dataset接口，用户可通过fluid.io.DataLoader.from_xxx创建数据加载器，可通过for-range方式迭代，简化使用方法，统一接口形式。
    * RecordIO接口移除，不再支持RecordIO接口。
    * 优化data接口，新的fluid.data接口相对fluid.layes.data 接口将对输入的数据的 shape 和 dtype 进行检查，使用None 和 -1 支持可变长维度。如果输入的 shape 或者 dtype 不对，将会报错。
* 报错信息优化
    * 简化C++信息栈输出，过滤和paddle函数无关的、对调试几乎没有帮助的栈信息和符号，大幅缩短了信息栈长度，提升了调试体验。
    * 对报错信息栈重新排版，添加清晰的分段标识与提示，并将核心提示置于最后，便于用户迅速定位重要信息，提升了调试体验。
    * 对34个重点python api增加输入类型检查，能正确报出输入类型不符合的错误，避免误导性报错。
    * 增强34个重点Op的维度检查报错信息，能打出详细维度信息，便于用户调试。
    * 针对sequence类op输入不含LoD的Tensor时报错不清晰的问题，为sequence类op增加了Input Tensor LoD信息检查，使错误提示更加直观易懂。
    * 强化机器自动化报错信息输出，在CI中强制推荐使用PADDLE_ENFORCE_XXX来替换PADDLE_ENFORCE接口，模版化打印出更具体的报错信息，并对应完成修复存量修复。
* 文档优化
    * 全面优化了所有API的中英文文档，保证文档的正确性、规范性、易读性，完善对应示例。
    * 增加了动态图中相关的更多文档说明和实例。
    * 对预测教程文档进行整体修改，重新组织结构和内容，提高了可读性和实用性。
    * 优化了部分指南性文档。
* 编译优化
    * 将默认的CMAKE_BUILD_TYPE从RelWithDebInfo改成Release，减少初次接触的开发者的编译目录大小，避免因为编译目录太大导致编译失败。
    * 修复inference_lib.cmake编译随机失败的问题。
    * 去掉use_fast_math编译选项，避免为了提升性能而降低了CPU/GPU上的精度。
* Windows支持增强
    * 支持vs2017编译。
    * 编译流程优化，拆分第三方和Paddle的编译依赖关系，不再依赖openblas的预编译库。
    * 支持cuda10。
    * 增加模型支持，修复之前在windows无法正常运行的模型。
    * 支持Paddle CPU 版本离线安装包。
    * 支持预测SDK C-API。

训练框架
##########
* 性能优化
    * GPU性能优化 
        * 使用cuRAND库优化dropout的GPU实现，dropout op本身加速3.4倍，Transformer base模型和big模型在V100上的训练分别加速3.8%和3.0%。
        * 对smooth_label的CUDA核函数完成代替Eigen实现，smooth_label op本身加速1.47倍。
        * 对 recurrent_op 的冗余 tensor copy 进行 share data，和删除运算过的 scope，该优化使得 benchmark 中 RNN 相关模型显存占用减少了 3 - 4 倍，速度有 2% - 数倍的提升。
    * CPU性能优化
        * BERT优化：新增matmul multi-head MKL的支持。
        * 对lookup_table_op和sequence_pool_op (sum类型)做fuse，使用sparse GEMM优化，PyramidDNN模型在CPU上的训练速度获得8%的提升。
    * 内存/显存优化
        * 新增变长输入下的MKLDNN分层缓存策略和清理策略，修复MKLDNN在变长输入下内存泄漏问题 。
        * 添加了控制流 op 多层嵌套情况下的显存优化策略支持。
        * Allocator容错机制。针对多线程并发申请显存导致显存可能瞬间峰值超标问题，设计了Allocator重试策略，在第一次申请显存失败后会等待最长10s进行失败重试（若期间有显存释放，会提前触发失败重试）。
        * 显存Cache清理。解决了以往TemporaryAllocator和Cudnn workspace单例会cache显存不释放的问题，提高显存利用率。
        * 新增AutoGrowth显存分配策略。用户可通过设置环境变量FLAGS_allocator_strategy=auto_growth开启显存自增长策略，按需分配显存，解决了原有预分配92%可用显存策略占用显存过多、难以按需分配的问题，且不影响模型训练速度。
        * 显存的Allocator容错机制完善，保证Allocator的稳定性。针对多线程并发申请显存导致显存可能瞬间峰值超标问题，设计了Allocator重试策略，在第一次申请显存失败后会等待最长10s进行失败重试（若期间有显存释放，会提前触发失败重试）。
* OP优化
    * 支持用户在框架外部、脱离框架自定义C++/CUDA OP。
    * 新增OP
        * 新增eye_op，用于构建单位矩阵，或一批单位矩阵。
        * 新增gather_nd_op，gather_op的高维推广，用于将输入数据中的切片，收集到由索引指定的形状的张量中。
        * 新增scatter_nd_op，scatter_op的高维推广，这个操作与scatter_nd_add_op类似，除了相加的张量是通过零初始化的。相应地，scatter_nd(index, updates, shape) 等价于 scatter_nd_add(fluid.layers.zeros(shape, updates.dtype), index, updates)。 用于根据索引indices将更新数据updates散布到新的(初始为零)张量中。
        * 新增scatter_nd_add_op：通过对Variable中的单个值或切片应用稀疏加法，从而得到输出的Variable。
        * 新增center_loss：用以辅助Softmax Loss进行人脸的训练，利用softmax loss来分开不同类别，利用center loss来压缩同一类别。center loss意思为：为每一个类别提供一个类别中心，最小化mini-batch中每个样本与对应类别中心的距离，从而达到缩小类内距离的目的。
        * 新增LookAHead Optimizer：针对Paddle不支持Lookahead优化算法这一问题，我们新增了这一优化算法。它的核心原理是：维护两个参数，快参数正常做前向反向运算，当快参数更新k次后，用它来更新慢参数，使二者同步。他的效果是在某些模型上能收敛更快。
        * 新增InstanceNorm op 实例归一化：根据每个样本的每个通道的均值和方差做归一化，一般用在图像生成模型中，把一个样本的风格迁移到另一个样本中。
        * 新增PreciseRoiPooling ：PrROI Pooling采用积分方式计算每个pool区域的值，这种计算方式将区域中的插值看作是连续的，计算所有插值点求积分得到该区域所包围点的总和，最后除以pool区域面积就得到该区域的值，因此结果更加准确。
        * 新增hard_swish_op：hard_swish激活函数，在MobileNetV3架构中被提出，相较于swish激活函数，具有数值稳定性好，计算速度快等优点。
        * 新增mse_loss_op：均方损失函数，用于计算两个输入间的均方差。
        * 新增elementwise_mod的float/doule kernel 。
        * 新增strided_slice op 。
        * MKLDNN kernel更新：
            * 新增Leaky_relu的MKL-DNN kernel 和 conv + activation fusion pass。
            * 支持不同axis的softmax MKL-DNN kernel。
            * 重构5个op （conv， pooling， batch_norm， softmax，LRN）的FP32 MKL-DNN kernel代码，增强代码可维护性和可读性。
    * OP功能优化升级
        * 部分op参数升级支持tensor及包含tensor的list，支持常数对应维度的推断
            * slice op 涉及参数starts 和ends。
            * reshape op 涉及参数shape。
            * expand op 涉及参数expand_times。
            * pow op 涉及参数factor。
            * fill_constant op 涉及参数 shape ，并将calc_gradient接口中使用的fill_constant_batch_size_like替换为fill_constant。
            * uniform_random op 涉及参数shape, 支持tensor及包含tensor的list。
            * image_resize、resize_nearest、resize_bilinear、resize_trilinear支持out_shape为tensor或者包含tensor的list，支持常数对应维度的推断，scale 参数支持tensor。
            * 新增crop_tensor，支持shape参数为tensor或者包含tensor的list，支持常数对应维度的推断。
        * 优化部分op输入tensor的维度检查
            * 移除huber_loss 、rank_loss和cross_entropy op中输入shape的最后一维强制为1的限制，输出loss的shape与label保持一致。
            * 新增fluid.one_hot和fluid.embeddingop，移除input参数shape最后一维为1的限制。
            * 优化sequence_pad和sequence_unpadop中length的shape，由[n,1]简化为[n]。
        * 部分op升级支持channel_last格式输入
            * conv2d、conv3d、pool2d、pool3d新增data_format参数，支持channel_last格式输入。
            * conv2d_transpose、conv3d_transpose新增data_format参数，支持channel_last格式输入。
            * image_resize、resize_nearest、resize_bilinear、resize_trilinear新增data_format参数，支持channel_last格式输入。
            * group_norm支持channel_last格式输入。
        * 涉及padding操作的OP，支持非对称padding，以及SAME和VALID 两种padding方式
            * conv2d、conv3d、pool2d、pool3d支持上述padding方式。
            * conv2d_transpose、conv3d_transpose支持上述padding方式。
        * 对以下op进行inplace显存优化支持
            * elementwise_add_grad_grad, elementwise_sub_grad_grad, elementwise_mul_grad_grad, elementwise_div_grad_grad, relu_grad_grad, leaky_relu_grad_grad, sqrt_grad_grad, square_grad_grad。针对GAN模型梯度惩罚显存占用较高的问题，为二重反向op添加inplace，优化其显存占用。
        * 升级部分仅支持LoDTensor输入的OP兼容padding模式，包括linear_crf_op, crf_decoding_op, hash_op, edit_distance_op, chunk_eval_op, warpctc_op, ctc_align_op, row_conv_op。
* Intel N-Graph集成
    * 增加了ngraph_subgraph_pass对训练的支持，通过build strategy激活N-Graph提供对parallel executor的支持。
    * 修正N-Graph对多线程问题，提供对多线程预测的支持。
* 动态图
    * 性能优化 
        * 对动态图底层执行机制进行了重构，在大部分模型上有30%左右的速度提升 ，显存开销有2%左右下降。
    * 功能完善
        * 支持基于stop_gradient设置的自动剪枝功能和detach接口，满足冻结部分子网的需求。
        * 支持模型在不同设备上执行data_transform， 可以使用less_than/greater_than等功能。
        * 重新实现op（unsqueezed_op、unstack_op、flatten_op、fill_constant_op）等，使之能够支持动态图。
    * 易用性提升
        * 针对部分动态图不支持的接口提供了优化的报错 （包括Variable相关接口和Optimizer相关接口）。
        * 针对Layer中的参数提供了可供访问的接口。
        * 优化动态图save load接口，旧的dygraph下面的 save_persistables 删除。
        * 支持了Layer call()可以使用关键字传入，使得前向执行时可以自定义传入的参数。

预测部署
########
* 服务器云端预测库
    * 接口优化 
        * 增加预测C API。
        * 针对设置环境变量GLOG_v=4可以打印出预测过程中包含模型op及op fuse的详细log会暴露较多信息，为AnalysisConfig添加DisableGlogInfo()接口（当前仅支持全局最多调用一次），方便使用者关闭GLOG输出，避免模型结构泄漏。
        * 针对用户在使用C++预测库时不易获得模型描述中的输入shape的问题，为AnalysisPredictor添加GetInputTensorShape()接口，方便用户在运行预测引擎之前从模型中拿到输入shape，以避免输入错误的shape。
    * 功能优化
        * 在模型中添加了模型版本号及算子兼容性信息。在此版本之后，旧版本模型在新版本 Paddle 库上使用 AnalysisPredictor 执行预测时会进行兼容性检查。
        * CPU INT8量化预测支持持续加强：支持mobilenet-ssd的训练后量化， 精度下降1%内， 性能提升3倍在第二代智强可扩展处理器6271上；新增Mul op的INT8 MKL-DNN kernel。
    * 性能优化
        * 优化了Mobilenetv2, ShuffleNet, Effecientnet 在CUDA GPU下的预测速度，mobilenetv2 从 5.3ms 减至 1.9ms，Shufflenetv2 从 6.3ms 减至1.4ms，Effecientnet 从60ms 减至 32ms。
        * 实现一个简化Graph中基础op的Pass，预测时，upscale_in_train类型的dropout op直接移除，downgrade_in_infer类型的dropout op使用scale op代替。该优化使ERNIE模型在P40上的预测速度提升1.8%。
        * 实现一个cudnn_placement_pass，将Graph中所有op的use_cudnn设置成true。该优化使ERNIE模型在P40上的预测速度提升10%。
        * 实现fc op的GPU Kernel，并支持将激活操作融合到fc op中。该优化使ERNIE模型在P40上的预测速度提升2.1%。
        * 实现融合fc+elementwise_add+layer_norm操作的Pass和GPU Kernel。该优化使ERNIE模型在P40上的预测速度提升4%。
        * 实现了multihead matmul 融合算法的相关PASS和Kernel。该优化使Ernie模型在P4 GPU上的速度提升超过30%。
        * 优化QAT（训练中量化）训练出来的模型在CPU INT8 kernel上执行的速度。通过PASS对训练出的QAT模型进行修改，结合训练后优化的PASS，使QAT训练出的模型可以在MobilenetV1， MobilenetV2， ResNet50，VGG16上精度变化（相比于FP32模拟量化）在0.1%内，ResNet101和VGG19精度变化在0.3%内，性能在6个模型上提升相比于原始未优化的QAT模型在第二代智强可扩展处理器6271上可达到4-9倍的性能提升。
    * 问题修复 
        * 针对之前AnalysisPredictor中设置FLAGS_profile无效的问题，为AnalysisConfig添加EnableProfile()接口，现在用户可以调用该接口开启预测的profiler，而无需设置FLAG。
        * 对ZeroCopyTensor的copy_from_cpu、mutable_data等方法添加了uint8模板支持，目前ZeroCopyRun已经可以正确地接收uint8输入进行预测。
        * 针对Paddle-TRT在包含多个op共享同一参数的模型如retinanet、faster_rcnn、cascade_rcnn中出现的重复设定weight、过早删除参数等bug进行了修复，Paddle-TRT已可以支持上述模型。
* 移动、嵌入式端侧预测库
    * 发布PaddleLite，定位高性能、多平台、轻量化的端侧预测引擎，并可作为服务器端飞桨原生预测库的加速库。具体见https://github.com/PaddlePaddle/Paddle-Lite
* Paddle Serving
    * 新增支持超大规模分布式预估服务能力
        * 发布了来源于百度内部经过海量数据检验的高性能分布式版本kv存储器组件cube，提供稀疏参数的分布式存储和查找，在高并发条件下单位时间吞吐总量是redis的13倍，是单机版kv存储器rocksDB的6倍。
        * 发布了Elastic CTR解决方案：针对超大规模稀疏参数的CTR任务，提供了基于k8s集群的分布式训练以及serving分布式参数部署预测的流程文档，并提供了一键式的解决方案。
    * PaddleServing编译速度提升 
        * 预测接口的编译依赖由paddle源码改为paddle inference lib，编译速度提升6倍。
    * PaddleServing易用性提升 
        * 支持Python client
* PaddleSlim
    * 添加基于硬件的小模型结构搜索功能。
    * 对量化训练、蒸馏和通道裁剪三种策略扩充分类模型示例，添加检测模型示例。 
    * 新增部分量化功能的支持，目前用户可选择对同一类型的op仅部分进行量化。
    * 新增对pool2d、elementwise_add等op的量化训练支持。

分布式训练
############
* 性能优化 
    * 新增LocalSGD多机训练算法：针对GPU多机多卡同步训练过程中存在trainer速度不一致（随机）导致同步等待问题，设计了局部异步训练策略，通过多步异步训练（无通信阻塞）实现慢trainer时间均摊，从而提升同步训练性能。在4机32块V100 GPU卡的配置下，在Resnet50 Imagenet分类任务上，测试集top5准确率达到93%的情况下，训练吞吐提升8.16%。模型链接： https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet 。
    * 新增GEO-SGD分布式CPU多线程全异步训练算法：通过训练节点维护独立参数且局部多轮更新，同时全局参数增量更新，大幅降低了训练中的通信占比。在文本匹配Simnet_bow模型上，GEO-SGD相比飞桨1.5全异步模式，在25节点12线程下，训练速度提升2.65倍，保持效果对齐。在Word2Vec模型上，GEO-SGD相比飞桨1.5全异步模式，在4、8、16、32节点16线程下，训练速度分别提升3.79倍、3.92倍、4.69倍、6.88倍，效果保持对齐。
    * Fast Resnet：采用可变图像大小、可变batch size和矩形验证图像等策略，显著提升Resnet50模型在ImageNet数据集的训练速度。在4机32块V100 GPU卡的配置下，top5准确率达到93%的时间缩短至35分钟，收敛速度提升2.21倍。在8机64块V100 GPU卡的配置下，top5准确率达到93%的时间缩短至27分钟。模型链接：https://github.com/PaddlePaddle/Fleet/tree/develop/examples/fast_imagenet 。
* 新增超大Batch训练优化器RecomputeOptimizer。在内存固定的情况下，Recompute优化器可以显著提高模型可以运行的batch size,提升为原来的 17%-309%；训练效果是无损的，收敛趋势一致，但实际吞吐会有一定损失。
* 新增Collective Op：all_reduce_op、broadcast_op、all_gahter_op、reduce_scatter_op，支持在组网中实现进程通信。
* 容错 
    * CPU全异步训练模式加入训练节点心跳检查，及时发现异常节点。
    * 加入retry机制 修复rpc errorcode 14的错误。
* 部署 
    * Paddle-K8S-Operator新增支持Volcano Job的提交，支持CPU分布式训练。 
 
模型建设（PaddlePaddle/models）
##############################
* 易用性优化
    * 全面优化了PaddleNLP和PaddleCV主要模型（Transformer，BERT，DMTK，PaddleDetection，PaddleGAN，PaddleVideo，ImageClassification）的安装、自定义数据以及对windows平台的支持等功能和体验。
* PaddleNLP
    * 发布文本生成库Seq2seq
        * 开源多个文本生成模型，包括vanilla seq2seq，seq2seq with memory network，variational seq2seq。
    * 升级阅读理解库
        * 开源EMNLP2019阅读理解竞赛百度夺冠模型D-Net和相关预训练模型，兼容MRQA2019开放的18个抽取式阅读理解公开数据集的并行训练、高性能评估以及搭建阅读理解serving的相关工作。
    * 升级语义表示库升级
        * 开源EMNLP2019阅读理解竞赛百度夺冠模型D-Net和相关预训练模型，兼容MRQA2019开放的18个抽取式阅读理解公开数据集的并行训练、高性能评估以及搭建阅读理解serving的相关工作。
    * 升级语义表示库升级
        * 新增语义表示模型XLNet。
    * 发布开放多任务学习库PALM
        * 开源MRQA2019比赛百度夺冠使用的多任务学习框架PALM，只需要几十行代码就可以完成基于ERNIE、BERT等预训练模型的硬共享、层次共享等多任务学习算法。
* PaddleCV
    * 发布图像分割库 PaddleSeg：具备丰富数据增强、模块化设计、高性能和端到端部署四大特点。
        * 模型
            * 新增DeeplabV3+/UNet/PSPNet/ICNet四种网络支持，对应预训练模型共18个。
            * 新增车道线分割、人像分割、人体部件分割三个预测模型。
        * 功能 
            * 支持softmax loss、bce loss、dice loss以及损失函数组合配置。
            * 支持翻转、旋转、多尺度变换、模糊、色彩饱和度调整等十余种数据增强策略。
            * 支持数据检查、边训边评估、模型导出、自动可视化、调参模式等易用性功能。
            * 支持FP16混合精度训练以及动态Loss Scaling。
            * 支持多进程训练与数据预处理。
        * 端到端部署 
            * 提供多平台（Windows/Linux）的C++高性能预测库编译、开发和部署。
            * 基于Paddle Serving提供高性能图像分割服务化部署能力。
    * 升级检测库 PaddleDetection
        * 新增2019 Objects365 Full Track比赛夺冠模型；新增DeformableConv系列模型；新增VGG-SSD系列模型；新增Cascade+Mask+FPN模型；新增更多基于的COCO两阶段模型；新增行人检测和车辆检测预训练模型；新增人脸检测模型Faceboxes和BlazeFace系列模型，并发布改进版的轻量级模型。
        * 功能
            * 支持multi-scale的训练、multi-scale测试，支持group norm等。支持FP16训练。增加C++预测部署能力，支持Windows和Linux系统。
            * 增加模型压缩量化和剪枝示例。
        * 增加中文文档，增加基于小数据的快速开始、迁移学习、模型导出、预测部署等文档，增加预测benchmark文档。
    * 完善图像分类模型
        * 发布9个EfficientNet预训练模型：EfficientNet-b0,EfficientNet-b1,EfficientNet-b2,EfficientNet-b3,EfficientNet-b4,EfficientNet-b5,EfficientNet-b6,EfficientNet-b7,EfficientNet-small。精度与论文持平。
        * 持续新增34个预训练模型：DarkNet53, DenseNet121，Densenet161, DenseNet169, DenseNet201, DenseNet264, SqueezeNet1_0, SqueezeNet1_1, ResNeXt50_vd_32x4d, ResNeXt152_64x4d, ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl，ResNet18_vd，ResNet34_vd，MobileNetV1_x0_25，MobileNetV1_x0_5，MobileNetV1_x0_75，MobileNetV2_x0_75，MobilenNetV3_small_x1_0，DPN68，DPN92，DPN98，DPN107，DPN131，ResNeXt101_vd_32x4d，ResNeXt152_vd_64x4d，Xception65，Xception71，Xception41_deeplab，Xception65_deeplab，SE_ResNet50_vd。
    * 升级PaddleVedio
        * 新增动作定位模型: BMN和BSN，其中BMN模型是ActivityNet2019比赛的冠军。
        * 新增VideoGrounding方向的BaseLine模型：TALL。
        * 新增VideoCaption方向的BaseLine模型：ETS。
    * 升级PaddleGAN 
        * 新增SPADE模型。
        * 替换Instanceorm实现，STGAN上判别器速度提升12%左右。
* PaddleSpeech
    * 升级语音识别模型 DeepSpeech 至飞桨最新版本。
    * 开源语音合成模型 DeepVoice3 。
* PaddleRec
    * 新增支持分布式训练的DeepFM、XDeepFM、DeepCrossNetwork。
 
工具组件
#########
* PaddleHub
    * 新增超参优化Auto Fine-tune功能，实现给定超参搜索空间，自动给出较佳的超参组合。
        * 支持两种超参优化算法：基于贝叶斯优化的HAZero和哈密尔顿系统的PSHE2。
        * 支持两种评估方式：Full-Trail和Population-Based。
    * 预训练模型丰富 
        * 升级ERNIE 1.0中文模型，提升模型载长文本情况下的效果(max_seq_len=512)。
        * 升级LAC模型至v2.0.0，保持效果的同时精简模型结构，提升预测速度。
        * 新增ERNIE 2.0 英文预训练模型。
        * 新增Ultra-Light-Fast-Generic-Face-Detector-1MB人脸检测模型。
        * 新增人体部件分割ACE2P模型。
        * 新增基于DeepLabv3+的人像分割模型HumanSeg。
        * 新增图像生成模型STGAN、AttGAN、StarGAN。
    * Fine-tune API升级，灵活性与易用性提升
        * 新增阅读理解Fine-tune任务。
        * 新增多指标评估功能。
        * 优化predict接口，提升预测性能。
        * 新增优化策略ULMFiT，包括以下三种配置
            * Slanted triangular learning rates：斜三角形学习率微调。
            * Discriminative fine-tuning：支持计算图按拓扑序分层采用不同学习率微调。
            * Gradual unfreezing：根据计算图的拓扑结构逐层参数解冻。
* PGL 图学习框架
    * 对应发布飞桨图学习框架PGL v1.0正式版。
    * 易用性：新增异构图的Metapath采样与Message Passing消息传递双机制，支持包含多种类型节点和边特征的异构图建模，新增Metapath2vec、GATNE等异构图算法。同时，文档、API、Tutorial等材料也进一步完善。
    * 规模性：新增分布式图引擎和分布式Embedding，可支持十亿节点百亿边的超巨图的多种分布式训练模式。新增distributed deepwalk和distributed graphSage两个分布式样例。
    * 丰富性：新增8个、累计13个图学习模型，涵盖了图神经网络和图表征学习的主流模型。新增的8个模型分别是LINE、struc2vec、metapath2vec、GES、GATNE、SGC、Unsup-GraphSage、DGI。
* PARL 深度强化学习框架
    * 对应发布飞桨强化学习框架PARL 1.2。
    * 更全更完善的并行RL机制，资源调度集群化，进一步降低并行算法实现门槛。
    * 支持大规模并行进化算法，可数百个CPU并发搜索索（https://github.com/PaddlePaddle/PARL/tree/develop/examples/ES）。
    * 上线更加全面的官方PARL文档（https://parl.readthedocs.io/en/latest/）。
* PaddleFL 联邦学习 
    * eFL 联邦学习 
发布飞桨联邦学习框架PaddleFL，方便快捷地支持联邦学习和AI隐私算法研究，并实现了FedAvg算法和基于差分隐私的SGD算法，支持分布式安全共享学习算法调研。https://github.com/PaddlePaddle/PaddleFL
* Paddle2ONNX
    * 对应升级paddle2onnx至0.2版本。
    * 新增pip安装方式。
    * 适配飞桨 v1.6的算子和ONNX v1.5版本。
    * 新增精度对齐框架，提供新增代码和模型转换的正确性验证功能。
    * 支持ResNet、DenseNe等10个Paddle图像分类模型的转换。
    * 支持SSD_MobileNet、YoloV3_DarkNet5等4个Paddle目标检测模型的转换。
* X2Paddle
    * 对应升级x2paddle至0.5版本。
    * 新增pip安装方式。
    * 新增统一的caffe、tensorflow和onnx模型计算图中间表示。
    * 支持caffe多分支模型的转换。
    * 大幅提升主流框架的模型转换能力，支持44个tensorflow OP，33个caffe Layer和48个onnx OP。
    * 为Paddle Lite提供多框架模型部署能力，支持包括图像分类、目标检测和语义分割在内共18个模型的无损转换。

BUG修复
##########
* 修复 rnn_search 模型无法跑起来的bug。
* 修复 save_inference_model 在 prune recurernt_op 时的 bug（该 bug 会导致一些 RNN 模型在 save inference model 后 load 预测出错）。
* 修复了动态图中多个Layer中act和bias等参数不生效的问题（其中包括：BilinearTensorProduct， GRUUnit，Conv2DTranspose ，LayerNorm，NCE ）、优化器保存的bug 、python端内存泄漏的问题、部分参数minimize段错误的问题、使用python中has_attr的失效的问题进行了修复。
* 修复FC mkldnn pass在AVX2机器上的精度diff问题。
* 升级MKL-DNN到0.20，并提升MKL-DNN单侧覆盖率到90%以上。
* 修复MKL-DNN训练后量化convolution和dequant op的squash问题。 

代码重构和升级
#########
* 清理了6个废弃的第三方库recordio，snappystream，snappy，jemalloc，anakin，gzstream。
