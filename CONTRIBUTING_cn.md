# 贡献文档

飞桨框架自开源以来，就提供了高质量的中英文文档。也有许多开发者为文档质量的提升，做出了重要的贡献，非常感谢每一位开发者对飞桨文档的支持。

## 文档说明

飞桨框架非常欢迎你参与到飞桨文档的建设中，与我们一同提升飞桨文档的质量。目前，飞桨文档分为以下几个部分，你可以选择任意你感兴趣的部分来参与：

- 使用教程
- 应用实践
- API 文档

### 使用教程

这部分内容主要是对飞桨框架的使用指南的说明，你可以对现有的内容进行纠错或者是改进，也可以新增你认为重要的文档在这个栏目中。我们非常欢迎你提出任何关于使用教程文档的建议以及修改。

### 应用实践

应用实践主要是使用飞桨框架进行具体的案例实现。目前已经有许多开发者贡献了非常优秀的案例，如 OCR 识别、人脸关键点检测等，我们非常欢迎你提交你的项目到我们的 repo 中来，并最终呈现在飞桨的官网上。

### API 文档

API 文档是飞桨框架的 API 文档，包含了飞桨框架 API 的说明介绍。我们非常欢迎你对我们的 API 文档提出修改，不管是 typo 或者是修改说明与示例，我们都非常感谢你对于 API 文档所作出的任何贡献。

## 参与方式

### 使用教程

这部分内容存放在 [docs/docs/guides](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides) 目录下，你可以通过提交 PR 的方式，来作出你的修改。具体修改方式请参考：[文档贡献指南](https://github.com/PaddlePaddle/docs/wiki/%E6%96%87%E6%A1%A3%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97)。

### 应用实践

这部分内容分为源代码与官网文档两部分，源代码的部分以 notebook 的形式，存放在 [book/paddle2.0_docs](https://github.com/PaddlePaddle/book/tree/develop/paddle2.0_docs) 目录下，你可以提交你的 notebook 格式的源码于该目录中；在你的 notebook 文件被合入后，我们会将其转为 md 文件，存储在 [docs/docs/tutorial](https://github.com/PaddlePaddle/docs/tree/develop/docs/tutorial) 中，然后呈现到官网。具体信息请参考：[[Call for Contribution] Tutorials for PaddlePaddle 2.0](https://github.com/PaddlePaddle/book/issues/905)。

### API 文档

飞桨框架同时提供中英文 API 文档。其中，英文 API 文档存于 [Paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 源代码中，绝大部分通过官网文档的源代码即可链接到，你可以在此位置对英文文档进行修改；而中文 API 文档存放在 [docs/docs/api](https://github.com/PaddlePaddle/docs/tree/develop/docs/api) 目录下。你可以针对文档中的任何错误与内容进行修复与完善，或者是新增你认为该文档中所需要的内容，我们非常感谢你对于 API 文档所付出的一切。具体修改方式请参考：[英文 API 文档贡献指南](https://github.com/PaddlePaddle/docs/wiki/%E8%8B%B1%E6%96%87API%E6%96%87%E6%A1%A3%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97)、[中文 API 文档贡献指南](https://github.com/PaddlePaddle/docs/wiki/%E4%B8%AD%E6%96%87API%E6%96%87%E6%A1%A3%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97)。

## 提交 PR

你对于飞桨文档的任何修改，都应该通过提交 PR 的方式来完成，具体的方法可以参考[提交 PR](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_contribution/local_dev_guide.html)。