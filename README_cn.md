
<h1 align="center">FluidDoc</h1>

[English](./README.md) | 简体中文

# 介绍

FluidDoc包含了所有PaddlePaddle相关的文档，它通过CI系统为PaddlePaddle.org提供文档内容

# 架构

FluidDoc将Paddle, Book, Models, Mobile and Anakin作为子模块，并放置在 `external` 目录下。按照标准做法，所有的子模块应当置于`external` 目录下

FluidDoc通过引用这些子模块来加载这些Repo中的文档。FluidDoc在 `FluidDoc/doc/fluid` 目录下构建了文档的整体树形结构。可以分别在 `FluidDoc/doc/fluid/index_cn.rst` 和 `FluidDoc/doc/fluid/index_en.rst` 查看。

当一个新发布的分支被push到了Github上，Travis-CI 将会自动启动编译文档并把文档部署到服务器

## 注意：
FluidDoc 需要Paddle Repo的python模块去编译生成API文档。但由于Paddle的python模块过于庞大，超过了Travis CI允许的最大时长，通常Travis CI将会因为超时问题失败。这是Travis上有三项作业的原因，其中两项用于构建库。当Travis缓存了这些库以后，下一次的构建将会变得非常的快。

## 通过PaddlePaddle.org预览文档

为了预览FluidDoc的文档，请按照[常规预览步骤](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.md)，但请在这一步将 paddle 的路径替换为 Fluid 的路径
`./runserver --paddle <path_to_FluidDoc_dir>`

## 发布新的分支
1. 创建一个新的分支，此分支的名字应遵循`release/<version>`
1. 在FluidDoc和子模块中更新文档
1. 确认所有的子模块中处于发布就绪的状态。Paddle, book, model, mobile and Anakin 应全部有稳定的commit
请注意：如果Paddle Repo更改了module/classes，涉及API文档的RST文件应当也被更新
1. 在 `external` 中更新文件然后commit文档变更
1. 将这个分支push到Github，Travis CI将会启动几项构建工作以把文档发布到PaddlePaddle.org的服务器
1. 请告知PaddlePaddle.org团队，发布的内容已经就绪。PaddlePaddle.org团队将使版本生效并更新默认的版本到最新版。PaddlePaddle.org也应当更新相应的搜索引擎文件
