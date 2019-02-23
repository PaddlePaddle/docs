==============
版本说明
==============

Paddle Fluid v1.3
##########################

重要更新
=========
* 统一Executor和ParallelExecutor接口，用户只需通过CompiledProgram将单卡模型转化多卡模型，并利用Executor进行训练或者预测。
* 正式发布AnalysisConfig 预测接口，支持计算图分析、算子融合等优化，并支持利用 Intel MKLDNN、Nvidia TensorRT 子图引擎等第三方库的加速.
* 模型库新增发布PaddlePaddle视频模型库，提供5个视频分类经典模型以及适合视频分类任务的通用骨架代码，用户可一键式高效配置模型完成训练和评测。
* 新增支持NLP语义表示BERT模型，支持多机多卡训练，支持混合精度训练，训练速度对比主流实现提升50%+，提供完整部署示例。
* 大规模稀疏参数服务器Benchmark发布， CPU多机异步训练发布显著提升点击率预估任务IO吞吐的built-in reader，多机多卡训练性能多方面提升。

基础框架
==========
* 安装
	* 新增Linux和MacOS下的中文版本辅助安装脚本，提供交互式安装方式，协助用户在复杂环境下快速完成PaddlePaddle安装。
	* Windows支持优化：新增cuda8，cudnn7的GPU支持，新增AVX指令集、MKLDNN、mnist数据集支持。修复Windows加载Linux/Mac下同版本paddle训练模型的问题。
* 增加动态图基础功能
	* 动态图tracer、 autograd、python Layer/PyLayer，动态图支持MLP、GAN、ptbRNN、Resnet模型，动态图支持Optimizer、GPU训练。
* Executor和ParallelExecutor接口优化
	* 对Executor和ParallelExecutor接口进行统一，用户只需通过CompiledProgram将单卡模型转化多卡模型，并利用Executor进行训练或者预测。
	* ParallelExecutor优化
		对MultiDevSSAGraphBuilder进行重构，使得MultiDevSSAGraphBuilder更易扩展。
		去除ParallelExecutor中的设备锁，提升ParallelExecutor多卡调度性能。
* 中间表达IR和Pass方面的优化
	* 完善C++ IR graph的python接口以及C++ IR pass的python接口。
	* 在framework.py中新增IRGraph类，为在Python层编写IR Pass做准备。
	* 新增支持网络无锁更新的Pass。
	* 新增QuantizationTransformPass，此为Quantization Aware Training量化模式训练前的图修改操作部分。
* 内存和显存方面的优化
	* 新增支持在编译时加入 Jemalloc 作为动态链接库，提升内存管理的性能，降低基础框架内存管理开销
	*新增memory optimize，inplace pass, memory pool early deletion等显存优化策略。
	* 新增支持网络无锁更新的Pass。
	* 新增QuantizationTransformPass，此为Quantization Aware Training量化模式训练前的图修改操作部分。
* Operator整体层面的优化
	* 每个op在执行前只做一次scope查询，减少读写锁操作（原来需要做1~5次scope查询）
	* 新增Temporary Allocator，减少op中的同步操作
	* 新增py_func operator，支持python op接入，用户可以借助py_func Operator快速实现所需要的特有操作
* 重构DDim，Variable Type等，降低基础框架调度开销。
* INTEL FP32计算相关优化
	* 优化density_prior_box operator，单op四线程提速3倍。
	* 优化Stack operator，单op提速16倍。
	* 开发Transpose，Concat和Conv3d三个基于MKLDNN的kernel。
	* 修复lrn operator中MKLDNN kernel精度bug，同时单op提速1.3倍。
	* 修复MKLDNN初始化占用5G内存的问题，目前初始化占用500MB。
	* 减少从MKLDNN OP kernel到非MKLDNN OP kernel时不必要的reorder。
* 完善CPU JitKernel
	* sequence pooling 的jitkernel，纯op提升2倍。
	* softmax 的jitkernel，纯op提升2倍，同时使得Bert模型CPU预测提升26%。
	* 常见的基本逻辑：向量的每个元素求平方kVSquare、矩阵乘法kMatMul、向量的最大值kHMax、向量所有元素的和kHSum。

预测引擎
==========

服务器预测
+++++++++++
* 正式发布AnalysisConfig 预测接口，支持计算图分析、算子融合等优化，并支持利用 Intel MKLDNN、Nvidia TensorRT 子图引擎等第三方库的加速。
* 预发布 intel CPU上的 预测 INT8 离线量化方案
	* 开发Conv2D，Pool2D，Quantize，Dequantize四个基于MKL-DNN的INT8 kernel。
	* 预发布Calibration的3个核心Python API（paddle.fluid.contrib.Calibrator）。
	* 开发Calibration工具，保证FP32和INT8的精度在ResNet-50和MobileNet-V1在ImageNet验证数据集上相差在1%内。
	* 支持Intel Xeon CascadeLake Server（VNNI指令）及Intel Xeon SkyLake Server，性能提升约为1.33倍。
* CPU预测速度提升
	* fuse sequence pooling concatop，支持N (<200)个sequence_pooling op concat起来组成一个新op，整体使得seqpool模型 CPU预测提升56%。
	* fuse 连续重复的fc op为一个大op，使得seqpool模型CPU预测速度提升15%。
	* fuse 逻辑为 $$((X * Y).^2 - (X.^2 * Y.^2) ) .* scalar$$ 的op组合 , 使得seqpool模型CPU预测速度提升8.2%。
	* 针对输入tensor元素个数为1的情况，优化compare_op的CPU Kernel。
* 新增Paddle-TRT 对Calibration INT8的支持，GPU预测速度提升
	* 模型VGG，Resnet50上预测速度达到了Paddle-TRT float32的两倍性能。
	* 模型VGG，Resnet50在imagenet数据集上测试，精度下降0.3%以内。
* 算子融合
	* 增加 fc和 con 相关两个 fuse，作用于 conv_op CUDNN kernel。
	* 新增Conv+Affine Channel的融合pass，Faster RCNN运行的性能提升26.8%。
	* 新增Transpose+Flatten+Concat 融合pass，MobilenetSSD模型性能提升15%。
	* 实现beam_search operator的CUDA Kernel，并且将相应的top-k、elementwise_add、reshape、log计算融合到beam_search operator中。
* 功能完善及易用性提升
	* 新增C++ IR graph的Python接口。
	* 新增预测库的Python接口。
	* 服务端预测支持从内存加载模型。
* 其他
	* 删除legacy V2代码。从1.3版本起，不再支持V1&V2老版本功能。
	* 修复Paddle-TRT elementwise-mul模型运行出现问题的bug。
	* 修复Paddle-TRT  trt_engine stream多个连续输入情况下模型输出结果异常的bug。

移动端预测
+++++++++++
* 效率优化，常见模型预测速度提升
	* int8预测支持dequantize和其他op（batch normalization/relu/elementwise add）进行自动kernel融合。
	* transpose2 operator对于shuffle channel操作进行优化。
	* gru operator使用neon指令进行优化，并针对batch size为1时进行优化。
	* 优化和实现pooling，支持任意的padding。
	* 优化和实现batch normalization、softmax、elementwise add。
* 新增支持多个输入和多个输出的模型预测。
* 新增实现prelu6 operator、cast operator、top_k operator。
* 修复int8 offline量化溢出结果不对的问题。
* 修复winograd实现在输入feature map的height和width不相等时结果可能为0的bug。

模型建设
==========
* PaddleCV 智能视觉
	* 新增发布PaddlePaddle视频模型库，包括五个视频分类模型：Attention Cluster、NeXtVLAD、LSTM,、stNet、TSN。提供适合视频分类任务的通用骨架代码，包括数据读取和预处理、训练和预测、网络模型以及指标计算等多个模块。用户根据需要添加自己的网络模型，直接复用其他模块的代码，快速部署模型。
	* 新增支持目标检测Mask R-CNN模型，效果与主流实现打平。
	* 语义分割DeepLabV3+模型，depthwise_conv op融合，显存优化，显存占用对比上一版本减少50%。
* PaddleNLP 智能文本处理
	* 新增支持NLP语义表示BERT模型，支持多机多卡训练，支持混合精度训练，训练速度对比主流实现提升50%+，提供完整部署示例。
	* 机器翻译Transformer模型优化解码计算，decoder中加入对encoder output计算结果的cache，预测速度提升一倍。
* PaddleRec 智能推荐
	* Sequence Semantic Retrieval 新增单机多线程、单机多卡运行示例，添加预测功能、数据预处理优化，完善部署示例。
	* GRU4Rec新增负采样功能，使用bpr loss和cross entropy loss的效果与原作打平。

分布式训练
===========
* 大规模稀疏参数服务器Benchmark发布
	* 测试真实业务场景下，特征规模百亿、样本平均特征数1k的点击率预估任务，在batch=512情况下，100worker加速比95.0，吞吐量1.56M/s 。
* CPU多机异步训练
	* 发布面向点击率预估任务的built-in reader，Criteo数据集下IO总吞吐提升1300%。
* GPU多机多卡水平扩展性能提升
	* 新增并行模式：PG（ParallelGraph）、MP（Multi-Process），独立GPU卡之间的计算，提升性能同时，不影响模型精度。
	* 在ResNet50模型，单机8卡V100下，PG, MP模式提升训练性能30%以上；4机32卡，PG模式提速46%，MP模式提速60%。
	* 在BERT模型，8卡V100下，PG, MP模式提升训练性能26%。
	* Multi-Process模式相比Parallel-Graph模式对Reader速度敏感度不高。
* GPU多机多卡垂直扩展性能提升
	* 新增功能：fp16和混合精度训练
	* Fp16单机单卡加速情况：ResNet50提速约87%，BERT提速约70%。
	* BERT同时开启PG和混合精度，单机8卡下单位时间吞吐提升120%。
	* ResNet50同时开启混合精度训练和MP模式，在V100单机8卡、4机32卡下，单位时间吞吐提升100%。
* 典型模型收敛速度优化
	* 新增功能：动态Batch Size，动态Image Resize方法。
	* Resnet50 on Imagenet数据集：训练收敛轮数下降为标准训练方法的1/3左右。

VisualDL
==========
* VisualDL graph支持Paddle fluid保存的模型可视化展示。



Paddle Fluid v1.2
##########################

Paddle Fluid v1.2在基础框架、预测引擎、模型建设、分布式训练各个方向上完成多项更新。基础框架支持python3.5及以上全版本。预测引擎优化，预测性能大幅提升。增强了对RL相关的支持能力。模型库新增图像分类任任务的预训练模型、语言模型任务新增基于cudnn的LSTM实现、分布式word2vec模型。CPU多机异步训练升级了包括worker异步并发和IO、通信优化在内多项功能，整体吞吐大幅提升。

基础框架
==========
* 安装
	* 提供新pip安装包，支持Windows下CPU执行。
* 编程语言
	* 新增对python3.6、python3.7的支持。
* 重构内存分配模块Allocator，提升CPU下内存分配策略，提升显存利用率(默认关闭，需要使用FLAGS_allocator_strategy)。
* 限制SelectedRows的使用。修复了稀疏正则和稀疏优化器的bug。
* Tensor支持DLPack，方便被其他框架集成和集成其他训练框架。
* OP
	* 修复 expand op shape 推理错误的bug
	* 支持 Selu 激活函数

预测引擎
==========
* 服务器预测
	* GPU 支持图融合，且支持和 TensorRT引擎混合改图，在Resnet50和Googlenet等图像通用模型上bs=1下性能提升 50%~100%。
	* GPU支持DDPG Deep Explore预测。
	* Paddle-TRT对更多模型的支持，其中包括Resnet， SE-Resnet， DPN，GoogleNet。
	* CPU, GPU, TensorRT 等加速引擎合并入 AnalysisPredictor，统一由 AnalysisConfig 控制。
	* 增加调用多线程数学库的接口。
	* 新增TensorRT plugin的支持，包括 :code:`split operator` ， :code:`prelu operator` ，  :code:`avg_pool operator` ,  :code:`elementwise_mul operator` 。
	* 增加了JIT CPU Kernel，支持基本的向量操作，以及常见的算法包括ReLU，LSTM和GRU的部分实现，可以实现在AVX和AVX2指令集之间自动runtime切换。
	* 优化CRF decoding和LayerNorm在AVX以及AVX2指令集上的实现。
	* 修复了 AnalysisPredictor 在GPU，在CPU 到 GPU 的 transfer data 不删除的问题。
	* 修复了 Variable 中包含 container 内存持续增长的问题。
	* 修复 :code:`fc_op` 不支持3-D Tensor的问题。
	* 修复了Analysis predictor 在GPU下执行pass时的问题。
	* 修复了TensorRT下运行GoogleNet的问题。
	* 预测性能提升
		* Max Sequence pool optimization，单op提高10%。
		*  :code:`Softmax operator` 优化，单op提升14%。
		*  :code:`Layer Norm operator` 优化，支持avx2指令集，单op提升5倍。
		*  :code:`Stack operator` 优化，单op提升3.6倍。
		* 增加depthwise_conv_mkldnn_pass，加速MobileNet预测。
		* 加速analysis模式的图分析时间，提升70倍。
		* DAM开源模型，提升118.8%。
* 移动端预测
	* 实现winograd算法， GoogleNet v1性能大幅提升35%。
	* GoogleNet 8bit优化，相比float加速14%。
	* MobileNet v1 8bit支持，相比float加速20%。
	* MobileNet v2 8bit支持，相比float加速19%。
	* FPGA V1 开发了Deconv算子。
	* android gpu支持MobileNet、MobileNetSSD、GoogleNet、SqueezeNet、YOLO、ResNet等主流的网络模型。


模型建设
===========
* CV图像分类任务发布MobileNet V1, ResNet101, ResNet152，VGG11预训练模型。
* CV Metric Learning模型新增arcmargin损失，并调整训练方式，采用element-wise作为预训练模型，pair-wise继续微调的训练方式提升精度。
* NLP语言模型任务新增基于cudnn的LSTM实现，对比PaddingRNN的实现方式，在不同参数配置下速度提升3~5倍。
* 增加分布式word2vec模型，包括新增的tree-based softmax operator，negative sampling等，与经典word2vec算法对齐。
* 新增GRU4Rec、Tag-Space算法的分布式配置。
* 完善Multi-view Simnet模型，并增加inference配置。
* 支持强化学习算法 DQN。
* 现已支持python3.x的模型：语义匹配DAM，阅读理解BiDAF，机器翻译Transformer，语言模型，强化学习DQN、DoubleDQN模型、DuelingDQN模型，视频分类TSN，度量学习Metric Learning，场景文字识别CRNN-CTC 、OCR Attention，生成式对抗网络ConditionalGAN、DCGAN、CycleGAN，语义分割ICNET、DeepLab v3+，目标检测Faster-RCNN、MobileNet-SSD 、PyramidBox ，图像分类SE-ResNeXt、ResNet等，个性化推荐TagSpace、GRU4Rec、SequenceSemanticRetrieval、DeepCTR、Multiview-Simnet。

分布式训练
=============
* CPU多机异步训练
	* worker异步并发：增加 :code:`AsyncExecutor` ，以训练文件作为执行粒度，支持分布式训练中的worker端计算异步无锁计算，同时支持单机训练。以CTR任务为例，单机训练速度，在充分利用单机线程的情况下，整体吞吐提升14倍。
	* IO优化：增加支持 :code:`AsyncExecutor` 的DataFeed，支持可定制化的通用分类任务格式。面向CTR任务，增加CTRReader，使数据读取速度线性提升，在PaddleRec/ctr任务中，整体吞吐提升1倍。
	* 通信优化：针对稀疏访问的Dense参数例如Embedding，增加稀疏通信机制，以语义匹配任务为例，获取参数的总量可以压缩到1%以下，在搜索真实场景的数据下，整体训练吞吐可以提升50倍。
* GPU多机同步训练
	* 修复Transformer、Bert模型下P2P训练模式会Hang住的问题。

文档
=========
* API
	* 新增13篇API​使用指南。
	* 新增300个API Reference中文文档。
	* 优化77个API Reference英文文档：包括代码示例、参数说明等。
* 安装文档
	* 新增python3.6、python3.7安装说明。
	* 新增windows pip install安装说明。
* Book文档
	* Book文档中的代码示例更改为Low level API。
* 使用文档
	* 新增《Operator相关注意事项》，更新《保存与载入模型变量》、《C++预测API介绍》、《使用TensorRT库预测》、《如何贡献代码》等多篇使用文档。
