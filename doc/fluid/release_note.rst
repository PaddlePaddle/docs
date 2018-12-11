==============
版本说明
==============

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
