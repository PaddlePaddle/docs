# 飞桨官网文档体验报告

## 上下文介绍

### 文档体系的分类

按照 https://documentation.divio.com/ 中的理论，任何软件产品，都应该配套以下四类文档：

![](./images/documentation-system.png)

本文将基于这个文档分类系统，参考比较多个深度学习框架的文档组织方式，形成最终报告。

### 竞品比较

#### TensorFlow

    - Tutorial：[Tutorial](https://www.tensorflow.org/tutorials)
    - GUIDE：[Guide](https://www.tensorflow.org/guide)
    - EXPLANATION：除 [ML Glossary](https://developers.google.com/machine-learning/glossary/tensorflow) 外，没有显式存在。
    - REFERENCE： [API 文档](https://www.tensorflow.org/versions)

TensorFlow 的文档规划，比较直接地匹配了本文所介绍的分类标准。

#### PyTorch

    - Tutorial：https://pytorch.org/tutorials/beginner/basics/intro.html
    - GUIDE：https://pytorch.org/tutorials/recipes/recipes_index.html
    - EXPLANATION：没有显式存在，分散在 API 文档中。
    - REFERENCE：[API 文档](https://pytorch.org/docs/stable/index.html)

[PyTorch 的文档](https://docs.pytorch.org) 分为 API 文档和 Tutorials 两大类。但实际 Tutorials 中可以继续分为 Learn the Basics 和 PyTorch Recipes 两大类。

并且，PyTorch 在自己的 API 文档首页中有 Notes。在必要时，也会在模块 API 开始做背景介绍，这些内容可以归为 “Explanation” 象限。

#### MindSpore

    - Tutorial：在 [教程](https://www.mindspore.cn/tutorials/zh-CN/r1.6/index.html) 前部分和 [编程指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/index.html) 的前半部分话题中
    - GUIDE：在 [教程](https://www.mindspore.cn/tutorials/zh-CN/r1.6/index.html) 后部分和 [编程指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/index.html) 的后半段话题以及 [开发者精华分享](https://bbs.huaweicloud.com/forum/forumdisplay-fid-1076-orderby-lastpost-filter-typeid-typeid-1255sub.html) 中
    - EXPLANATION：[规格和说明](https://www.mindspore.cn/docs/note/zh-CN/r1.6/index.html) 及 [FAQ](https://www.mindspore.cn/docs/faq/zh-CN/r1.6/index.html)
    - REFERENCE：[API 文档](https://www.mindspore.cn/docs/api/zh-CN/r1.6/index.htmls)

MindSpore 的有自己独立的文档分类标准和风格，所以硬套本文提及的文档分类标准，结果会显得有些复杂。以上所列的各类文档中，《开发者精华分享》是比较独特的一个栏目，他更像是 MindSpore 搭建的开源社区平台，吸收了用户贡献的各种经验，包括 Numpy 的使用，MindSpore 的安装问题如何解决等社区贡献的知识。


## 文档的完备性 & 宏观组织分析

现有 Paddle 文档，结构如下：

- 整体介绍
    - 基本概念
    - 升级指南
    - 版本迁移工具
- 模型开发
    - 10 分钟快速上手飞桨
    - 数据集的定义和加载
    - 数据预处理
    - 模型组网
    - 训练与预测验证
    - 单机多卡训练
    - 自定义指标
    - 模型保存与载入
    - 模型导出 ONNX 协议
- VisualDL 工具
    - VisualDL 工具简介
    - VisualDL 使用指南
- 动态图转静态图
    - 使用样例
    - 转换原理
    - 支持语法
    - 案例解析
    - 报错调试
- 推理部署
    - 服务器部署
    - 移动端/嵌入式部署
    - 模型压缩
- 分布式训练
    - 分布式训练开始
    - 使用 FleetAPI 进行分布式训练
- 自定义算子
    - 自定义原生算子
    - 原生算子开发注意事项
    - 自定义外部算子
    - 自定义 Python 算子
    - Kernel Primitive API
        - API 介绍
        - API 示例
- 性能调优
    - 飞桨模型量化
- 算子映射
    - Paddle 1.8 与 Paddle 2.0 API 映射表
    - PyTorch-PaddlePaddle API 映射表
- 硬件支持
    - 飞桨产品硬件支持表
    - 昆仑 XPU 芯片运行飞桨
    - 海光 DCU 芯片运行飞桨
    - 昇腾 NPU 芯片运行飞桨
- 参与开发
    - 本地开发指南
    - 提交 PR 注意事项
    - FAQ
- 环境变量 FLAGS
    - cudnn
    - 数值计算
    - 调试
        - check nan inf 工具
    - 设备管理
    - 分布式
    - 执行器
    - 存储管理
    - 曻腾 NPU
    - 其它

### 完备性分析

综合 Paddle 及其它竞品，先罗列一个较为完备的“知识点体系”。即如果是各个级别的用户，面向他们应该提供哪类信息，做一个全集式的罗列。

- 初级用户：初级用户定位是未有使用深度学习框架完成项目的经验，如高校低年级学生
- 中级用户：中级用户定位是通常意义上的算法工程师，在日常的项目中需要使用深度学习框架，但还不涉及较为“高级”的特性。如独立参与比赛的高校学生，工业界的初级、中级算法工程师。
- 高级用户：需要紧随前沿使用框架高级特性的工业届、学术界用户。如复杂的并行技术、算子融合、AMP 等高级优化技术。

对应的信息分类和罗列：

- 初级：
    - 基本数据（Tensor） 的概念及基本使用
    - 基本操作（算子）的概念及基本使用
    - 数据加载
    - 如何组网
    - 如何训练
    - 保存与加载模型
    - 可视化
    - 动态图、静态图的概念及基本使用

- 中级：
    - 动态图与静态图的转换
    - 如何转为 ONNX 格式
    - 如何部署
    - CV 领域的实践指南
    - NLP 领域的实践指南
    - 推荐系统领域的实践指南

- 高级：
    - 如何自定义算子
    - “高级”优化特性（如量化、AMP 等）
    - 框架设计文档

从内容完备性的角度看，飞桨应该是目前各个框架中完备性做得最好的，包括了以上除“框架设计文档”之外的所有点。并且提供了一系列与产业下沉，AI 助力有关的“产业实践文档”。

### 组织结构分析

从组织结构角度分析，以我个人的感受，飞桨还有一些值得讨论，和可能改进的地方。

#### 整体介绍

首先是 [整体介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/index_cn.html) 里的目录结构和内容可以考虑调整。

如 [整体介绍/基本概念](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/index_cn.html) 中包括了以下内容：

- Tensor 概念及介绍
- Paddle 中的模型与层
- 广播介绍
- 自动微分
- 自动混合精度训练
- 梯度裁剪

这部分应该是面对想要了解飞桨并上手使用的初级用户准备的。所以它定位应该是 “Tutorial”。但是有些地方可能讨论改进：

- [Paddle 中的模型与层](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/layer_and_model_cn.html) 一节与“升级指南” 里的 [组网方式](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#sequential) 一节有内容重叠。
- [广播介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/broadcasting_cn.html) 更像是 “Explanation” 类的东西，而且内容比较少，可以考虑放置到 API 文档中，以链接的方式引用就可以了。
- [自动微分机制介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/autograd_cn.html) 一节的内容与 [PyTorch 的 Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) 相比，在介绍 “飞桨自动微分运行机制” 时感觉内容选题复杂了一点。而在“飞桨中自动微分相关所有的使用方法说明” 一节中，没有继续细分“计算梯度”，“关闭梯度”等小标题，作为 Tutorial 感觉友好度还可以加强
- [自动混合精度训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html) 和 [梯度剪裁](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/gradient_clip_cn.html) 两节内容感觉归为 Guide 更合理，放在“基本概念”中会有点学习曲线陡增的感觉。

其次，在 [升级指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html) 包含了这些内容：

- 数据处理
- 组网方式
- 模型训练
- 单机多卡启动
- 模型保存
- 推理

这里面包含了很多“干货”的内容，但是我觉得放在“升级指南”中有“明珠暗投”的风险。因为一般人对“升级指南”的刻板印象，应该类似是“relase log”级别的东西，可能会不会优先点开。

据我体验，升级指南中的 [使用高层 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#id4) 和 [使用基础 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#id5) 包含了其它地方没有的内容，这些内容被用户错过的话，是可惜的。

而升级指南中的 [单机多卡启动](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#danjiduokaqidong) 和 [分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/index_cn.html) 中的内容非常接近，感觉可以考虑合并。

我建议可以对整体介绍中的内容进行重新组织，主要方向有：

- “整体介绍部分”尽量瘦身，主要以“信息索引”存在
- 将 “Tensor 概念介绍”、“模型与层”以及上文提到的升级指南中的操作性强又基础的内容，考虑与一级目录 [模型开发](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/index_cn.html) 合并
- 将自动混合精度、梯度剪裁以及分布式训练有关的内容，整理为专门领域的 Guide，作为独立的一级目录或二级目录存在


#### 其它一级目录

除了上文提到的个人看法外，飞桨的一级目录，大部分都内容充实、结构合理。只是对于少数章节，我个人觉得可以再讨论改进。

[性能调优](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/index_cn.html) 作为一个一级目录存在，但其中的二级目录只有 [模型量化](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/quantization.html) 一文。这显得有些单薄。不过在现有的目录结构下，这种选择可能已经是最优的了。我还没有具体的改进建议。

但是，如果未来考虑增加新栏目的话，感觉可以把增加一个“开发者实践”类型的栏目，其内容可以参考 MindSpore 的 [开发者精华分享](https://bbs.huaweicloud.com/forum/forumdisplay-fid-1076-orderby-lastpost-filter-typeid-typeid-1255sub.html)。将一些任务导向的话题，如量化、profiling 等“杂项”放在其中。

其次 [环境变量](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/flags/flags_cn.html) 这种纯粹的 flags 说明，应该归类为 Reference，感觉将它直接放到 API 文档中可能会更合理。

## 用户角色代入分析

### 初级用户

初级用户可以通过 [整体介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)、[模型开发](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/index_cn.html) 上手飞桨，掌握从数据加载、预处理到模型训练、导出到 ONNX 等各项基本技能。

### 中级用户

中级用户可以通过 [VisualDL 工具](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/03_VisualDL/index_cn.html)、[动态图转静态图](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/03_VisualDL/index_cn.html)、[推理部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/index_cn.html)、以及 [应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/index_cn.html)、[分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/index_cn.html) 中的各类信息，掌握实际落地过程中各个领域的遇到的模型调试、性能调优、部署等问题。

### 高级用户

想为飞桨开源项目增加功能的用户，可以参考 [自定义算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/index_cn.html) 了解如何增加算子，以及 [参与开发](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/10_contribution/index_cn.html) 了解提交 PR 的流程。

不过，其它框架，一般会有文档介绍它组件中的设计理念和原理，如 Pytorch 的这篇 [autograd](https://pytorch.org/docs/stable/notes/autograd.html)，OneFlow 的一些 [技术博客](https://zhuanlan.zhihu.com/p/337851255)，以及 MindSpore 的 [设计文档](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/technical_white_paper.html) （这个信息量偏薄弱点）。

在 Paddle 的文档体系中，好像没有找到类似的内容，这对于想学习 Paddle 底层原理甚至参与开源建设的爱好者，是一种遗憾。希望可以考虑推出。

## 报告总结

飞桨文档的内容完备性已经做到了同类产品中的顶尖水平，能够满足初级、中级以及定制开发用户的需要。
但在个别文档结构上，存在一些可以去除冗余、调整层级的改进点，值得讨论。同时也期待飞桨能整理开放出设计文档，让更多人可以了解飞桨开源代码背后的原理和巧妙。