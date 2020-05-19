#  Release Note

## 重要声明

 - 此版本为测试版，还在迭代中，目前还没有稳定，后续API会根据反馈有可能进行不兼容的升级。对于想要体验飞桨最新特性的开发者，欢迎试用此版本；对稳定性要求高的工业级应用场景推荐使用Paddle 1.8稳定版本。

 - 此版本主推动态图的开发方式，并提供了高层API的封装。动态图模式具有很好的灵活性，高层API可以大幅减少重复代码。对于初学者或基础的任务场景，推荐使用高层API的开发方式，简单易用；对于资深开发者想要实现复杂的功能，推荐使用动态图的API，灵活高效。

 - 此版本同时对飞桨的API目录体系做了优化，原目录下API会建立alias仍然可用，但建议新的程序使用新目录结构。

## 基础框架

### 基础API
- 组网类API实现动静统一，支持在动态和静态图两种模式下运行
- API目录结构调整，Paddle 1.x 版本的API主要位于paddle.fluid目录，本版本对API目录结构进行调整，使得分类更为合理，具体调整规则如下：
  - 原fluid.layers下跟tensor操作相关的API移动到paddle.tensor目录
  - 原fluid.layers下跟组网相关的操作移动到paddle.nn目录，带有参数的类型放到paddle.nn.layers目录，函数式的API放到paddle.nn.functional目录
  - 原fluid.dygraph下动态图专用API移动到paddle.imperative目录
  - 创建paddle.framework目录，用来存放跟框架相关的Program, Executor等API
  - 创建paddle.distributed目录，用来存放分布式相关的API
  - 创建paddle.optimizer目录，用来存放优化算法相关的API
  - 创建paddle.metric目录，用来创建评估指标计算相关的API
  - 创建paddle.incubate目录，用来存放孵化中的代码，其中的API有可能会发生调整，该目录存放了复数计算complex和高层API相关的代码
  - 所有在paddle.tensor和paddle.framework目录下的API，在paddle目录下创建别名，比如：paddle.tensor.creation.ones可以使用paddle.ones别名

- 新增API如下：
  - 在paddle.nn目录新增8个组网类的API: interpolate, LogSoftmax, ReLU, Sigmoid, loss.BCELoss, loss.L1Loss, loss.MSELoss, loss.NLLLoss
  - 在paddle.tensor目录新增59个Tensor相关API：add, addcmul, addmm, allclose, arange, argmax, atan, bmm, cholesky, clamp, cross, diag_embed, dist, div, dot, elementwise_equal, elementwise_sum, equal, eye, flip, full, full_like, gather, index_sample, index_select, linspace, log1p, logsumexp, matmul, max, meshgrid, min, mm, mul, nonzero, norm, ones, ones_like, pow, randint, randn, randperm, roll, sin, sort, split, sqrt, squeeze, stack, std, sum, t, tanh, tril, triu, unsqueeze, where, zeros, zeros_like
  - 新增device_guard用来指定设备，新增manual_seed用来初始化随机数种子


### 高层API
- 新增paddle.incubate.hapi目录，对模型开发过程中常见的组网、训练、评估、预测、存取等操作进行封装，实现低代码开发，MNIST手写数字识别任务对比动态图实现方式，高层API可减少80%执行类代码。
- 新增Model类封装，继承Layer类，封装模型开发过程中常用的基础功能，包括：
  - 提供prepare接口，用于指定损失函数和优化算法
  - 提供fit接口，实现训练和评估，可通过callback方式实现训练过程中执行自定义功能，比如模型存储等
  - 提供evaluate接口，实现评估集上的预测和评估指标计算
  - 提供predict接口，实现特定的测试数据推理预测
  - 提供train_batch接口，实现单batch数据的训练
- 新增Dataset接口，对常用数据集进行封装，支持数据的随机访问
- 新增常见Loss和Metric类型的封装
- 新增CV领域Resize, Normalize等16种常见的数据处理接口
- 新增CV领域lenet, vgg, resnet, mobilenetv1, mobilenetv2图像分类骨干网络
- 新增NLP领域MultiHeadAttention, BeamSearchDecoder, TransformerEncoder, TransformerDecoder , DynamicDecode API接口
- 发布基于高层API实现的12个模型，Transformer，Seq2seq,  LAC，BMN, ResNet,  YOLOv3, , VGG, MobileNet, TSM, CycleGAN,  Bert, OCR

### 性能优化
- 新增`reshape+transpose+matmul` fuse，使得Ernie量化后 INT8 模型在原来基础上性能提升~4%（在6271机器上）。量化后INT8模型相比未经过DNNL优化（包括fuses等）和量化的FP32模型提速~6.58倍

### 调试分析

- 针对Program打印内容过于冗长，在调试中利用效率不高的问题，大幅简化Program、Block、Operator、Variable等对象的打印字符串，不损失有效信息的同时提升调试效率
- 针对第三方库接口`boost::get`不安全，运行中抛出异常难以调试的问题，增加`BOOST_GET`系列宏替换了Paddle中600余处存在风险的`boost::get`，丰富出现`boost::bad_get`异常时的报错信息，具体增加了C++报错信息栈，出错文件及行号、期望输出类型和实际类型等，提升调试体验

## Bug修复

 - 修复while loop中存在slice操作时计算结果错误的bug
 - 修复inplace ops引起的transformer 模型性能下降问题
 - 通过完善cache key, 解决Ernie精度测试最后一个batch运行失败的问题
 - 修复fluid.dygraph.guard等context中出现异常时无法正确退出的问题
