# 贡献文档

[View English](./CONTRIBUTING_DOC.md)

非常欢迎您参与到飞桨文档的建设中，飞桨框架自开源以来，一直提供着优质的中英文文档。在这期间，也有许多开发者为文档质量的提升与内容的补充，做出了重要的贡献。非常感谢您对飞桨文档的支持，感谢您与我们一同提供优质的飞桨文档，供所有开发者参考学习。

## 文档说明

飞桨框架非常欢迎您参与到飞桨文档的建设中，与我们一同提升飞桨文档的质量。目前，飞桨文档分为以下几个部分，您可以选择任意您感兴趣的部分来参与：

- 使用教程
- 应用实践
- API文档

### 使用教程
这部分内容主要是对飞桨框架的使用指南的说明，您可以对现有的内容进行纠错或者是改进，也可以新增您认为重要的文档在这个栏目中。我们非常欢迎您提出任何关于使用教程文档的建议以及修改。

### 应用实践
应用实践主要是使用飞桨框架进行具体的案例实现。目前已经有许多开发者贡献了非常优秀的案例，如OCR识别、人脸关键点检测等，我们非常欢迎您提交您的项目到我们的repo中来，并最终呈现在飞桨的官网上。

### API文档
API文档是飞桨框架的API文档，包含了飞桨框架API的说明介绍。我们非常欢迎您对我们的API文档提出修改，不管是typo或者是修改说明与示例，我们都非常感谢您对于API文档所作出的任何贡献。

## 参与方式

### 使用教程
这部分内容存放在 [FluidDoc/doc/paddle/guides](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/paddle/guides) 目录下，您可以通过提交PR的方式，来作出您的修改。

### 应用实践
这部分内容分为源代码与官网文档两部分，源代码的部分以notebook的形式，存放在 [book/paddle2.0_docs](https://github.com/PaddlePaddle/book/tree/develop/paddle2.0_docs) 目录下，您可以提交您的notebook格式的源码于该目录中；官网文档通过notebook to rst的方式，将notebook转为rst的格式，存放于 [FluidDoc/doc/paddle/tutorial](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/paddle/tutorial)中，在您的notebook文件被合入后，我们会将其转为rst文件，然后呈现到官网中。

### API文档
飞桨框架同时提供中英文API文档。其中，英文API文档存于[Paddle](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle)源代码中，绝大部分通过官网文档的源代码即可链接到，您可以在此位置对英文文档进行修改；而中文API文档存放在[FluidDoc/doc/paddle/api/paddle](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/paddle/api/paddle)目录下。您可以针对文档中的任何错误与内容进行修复与完善，或者是新增您认为该文档中所需要的内容，我们非常感谢您对于API文档所付出的一切。

## 提交PR
您对于飞桨文档的任何修改，都应该通过提交PR的方式来完成，具体的方法可以参考[提交PR](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_contribution/local_dev_guide.html)

## 其他
我们会不定期释放一些问题，您可以在下方的清单中查看并认领您感兴趣的问题，然后对其进行优化，再次感谢您对于飞桨文档的贡献。

**应用实践**：[Call for Contribution-Tutorials for PaddlePaddle 2.0](https://github.com/PaddlePaddle/book/issues/905)