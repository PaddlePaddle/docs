==============
版本说明
==============

PaddlePaddle v1.1
#####################

PaddlePaddle v1.1 在基础框架、模型建设、分布式训练、预测引擎各个方向上完成多项更新。OP进行了全面完善和优化，模型库新增了自然语言处理、视觉和推荐等领域的大量经典模型，分布式训练能力显著提升，支持千亿规模稀疏参数大规模多机异步训练，预测库易用性和效率提升，移动端预测支持更多模型和更多硬件。详情如下：

基础框架
=========
* 安装
	* Mac OS X 10.11及以上pip安装支持。
	* Mac OS X 10.12及以上从源码编译安装支持。
* 编程语言
	* Python3的支持（python3.5版本）。
* IO
	* 新增PyReader，支持用户基于python自定义数据读取和预处理的的高性能数据输入。在ResNet50模型上，单机情况下：单卡数据读取速度提升4%、4卡数据读取速度提升38%、8卡数据读取速度提升60%。
	* 实现一个多进程PyReader decorator，配合PyReader可以实现数据读取性能线性提升。
* OP优化
	* 优化了 :code:`split operator` ，显著提升性能。
	* 扩展 :code:`multiclass_nms operator` ，支持多边形的预测框。
	* 通过 :code:`generatoe_proposals operator` 的CUDA实现，显著提升性能。
	* 通过 :code:`affine_channel operator` 融合batch_norm operator，显著提升性能。
	* 优化 :code:`depthwise_conv operator` 的forward和backward，显著提升性能。
	* 优化 :code:`reduce_mean operator` 。
	* 优化 :code:`sum operator` ，该operator在输入是 :code:`Tensor` 的情况下，减少一次zero memory耗时。
	* 优化 :code:`top_k operator` ，显著提升性能。
	* 优化 :code:`sequence_pool operator` ，显著提升性能。
	* 优化 :code:`elementwise_add operator` ，显著提升性能。
	*  :code:`while operator` 性能优化，相关的模型性能整体提升 30%+。
	*  :code:`sequence_slice operator` 的实现，对于一个sequence，可以从指定位置开始，slice出指定长度的subsequence。
	*  :code:`sequence_unpad operator` 的实现，支持padding Tensor转LoDTensor。
	* 支持截断正态分布初始化方法(truncated normal initializer)。
	* 二维 :code:`padding operator` 的实现，支持一个每个纬度的首尾padding不同的大小。
	* 更多 operator支持： :code:`sequence_reverse operator` ， :code:`sequence_enumerate operator` , :code:`sequence_scatter operator` , :code:`roi_align operator` ， :code:`affine_channel operator` , :code:`anchor_generator operator` , :code:`generate_proposal_labels operator` , :code:`generate_proposals operator` , :code:`rpn_target_assign operator` 、 :code:`roi透视变换operator` ,  :code:`seq_pool operator` 、 :code:`seq_expand operator` 、 :code:`seq_concat operator` 、 :code:`seq_softmax operator` 、 :code:`lod_to_array operator` 、 :code:`array_to_lod operator` 。
* 显存优化
	* 显存优化策略eager deletion支持control flow (e.g. if-else, while)中子block的优化。显著降低包含control flow的模型的显存开销。

模型建设
=========
* 自然语言处理方向增加开源语义匹配DAM模型和阅读理解BiDAF模型，机器翻译Transformer模型性能优化后训练速度提升超过30%，模型效果和训练速度均达到业界领先水平。
* 计算机视觉方向增加开源OCR识别Seq2Seq-Attention模型，目标检测Faster-RCNN模型，图像语义分割DeepLab v3+模型，视频分类TSN模型，图像生成CircleGAN/ConditionalGAN/DCGAN模型，以及Deep Metric Learning模型，模型效果和训练速度均达到业界领先水平。
* 个性化推荐任务系列模型支持：新闻自动标签模型TagSpace，序列语义召回模型GRU4Rec、SequenceSemanticRetrieval，点击率预估模型DeepCTR，多视角兴趣匹配模型multiview-simnet。
	* TagSpace : TagSpace: Semantic Embeddings from Hashtags。
	* SequenceSemanticRetrieval  : Multi-Rate Deep Learning for Temporal Recommendation。
	* multiview-simnet  : A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems。
	* GRU4Rec  : Session-based Recommendations with Recurrent Neural Networks。
	* DeepCTR  : DeepFM: A Factorization-Machine based Neural Network for CTR Prediction。

* 公开的Quora数据集上，实现并复现了四个公开匹配算法，具有较好的通用性，可应用于NLP、搜索、推荐等场景。
	* cdssmNet：Learning semantic representations using convolutional neural networks for web search 。
	* decAttNet：Neural paraphrase identification of questions with noisy pretraining 。
	* inferSentNet：Supervised learning of universal sentence representations from natural language inference data 。
	* SSENet：Shortcut-stacked sentence encoders for multi-domain inference。

分布式训练
==========
* GPU多机多卡同步训练支持参数同步频率可配置化，在V100上支持的batch size提升为v1.0版本的8倍，通过合理的通信间隔配置，使GPU卡数较少的情况下超大Batch同步训练成为可能，并在优化算法方面保证了收敛效果不变。
* 支持千亿规模稀疏参数服务器，用于大规模多机异步训练，适用于推荐、广告等领域的点击率预估模型。


预测引擎
========
* 服务器预测
	* 预测库Windows支持。
	* PaddlePredictor C++ 接口稳定版发布，已经实际支持一部分业务上线，并持续向前兼容。
	* 预发布整合了 TensorRT 子图加速方案。运行时自动切割计算子图调用TensorRT加速。目前Paddle TensorRT 依旧在持续开发中，已支持的模型有 AlexNet, MobileNet, ResNet50, VGG19, ResNet, MobileNet-SSD等。
	* 基于图优化的 CPU 加速 feature，实现了 LSTM，GRU 等一系列 operator 的 fuse，理论上可以大幅提升序列相关模型的性能。
	* 增加了部署时 AVX 和 NOAVX 自动切换的feature，可以针对重点模型实现AVX, AVX2, AVX512自动切换。
	* 提升预测库易用性：只需要 include一个头文件和一个库。
	* ICNet 预测性能大幅提升。
* 移动端预测
	* 树莓派上MobileNet、GoogleNet、ResNet 34等多个模型支持。
	* Mali GPU和Andreno GPU上MobileNet v1模型支持。
	* ZU5、ZU9等FPGA开发板上ResNet 34和ResNet 50模型支持。
