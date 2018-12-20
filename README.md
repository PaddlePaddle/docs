# Introduction
FluidDoc consolidates all the documentations related to Paddle. It supplies the contents to PaddlePaddle.org via CI. 

FluidDoc包含了所有PaddlePaddle相关的文档，它通过CI系统为PaddlePaddle.org提供文档内容

# Architecture
FluidDoc submodules Paddle, Book, Models, Mobile and Anakin under `external` folder. All submodules should be put under `external` as standard practice. 

FluidDoc将Paddle, Book, Models, Mobile and Anakin作为子模块，并放置在 `external` 目录下。按照标准做法，所有的子模块应当置于`external` 目录下

FluidDoc then uses them as references to load up the documents. The FluidDoc constructs the whole doc-tree under the `FluidDoc/doc/fluid` folder. The entry point is `FluidDoc/doc/fluid/index_cn.rst` and `FluidDoc/doc/fluid/index_en.rst`

FluidDoc通过引用这些子模块来加载这些Repo中的文档。FluidDoc在 `FluidDoc/doc/fluid` 目录下构建了文档的整体树形结构。可以分别在 `FluidDoc/doc/fluid/index_cn.rst` 和 `FluidDoc/doc/fluid/index_en.rst` 查看。

When a release branch is pushed to Github, Travis-CI will start automatically to compile documents and deploy documents to the server. 

当一个新发布的分支被push到了Github上，Travis-CI 将会自动启动编译文档并把文档部署到服务器

## Note: 
FluidDoc needs Paddle python module to compile API documents. Unfortunately, compiling Paddle python module takes longer time Travis CI permits. Usually Travis CI will fail due because of timeout. That's why there three jobs on Travis, two of them are to build libraries. Once the libraries are cached on the Travis, next build will be a lot faster.

## 注意：
FluidDoc 需要Paddle Repo的python模块去编译生成API文档。但由于Paddle的python模块过于庞大，超过了Travis CI允许的最大时长，通常Travis CI将会因为超时问题失败。这是Travis上有三项作业的原因，其中两项用于构建库。当Travis缓存了这些库以后，下一次的构建将会变得非常的快。

## Preview with PPO
To preview documents constructured by FluidDoc. Please follow the [regular preview step](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.md), but replace the path to paddle with the path to FluidDoc
`./runserver --paddle <path_to_FluidDoc_dir>`

## 通过PaddlePaddle.org预览文档

为了预览FluidDoc的文档，请按照[普通预览步骤](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.md)，但请在这一步将路径用paddle替代
`./runserver --paddle <path_to_FluidDoc_dir>`


# Publish New release
1. Checkout a new release branch. The branch name should follow `release/<version>`
1. Update the documentations on the submodules or within FluidDoc
1. Make sure all the submodules are ready for release. Paddle, book, model, mobile and Anakin should all have stable commits. Note: Paddle repo should update the API RST files accordinly if Paddle changes the included module/classes. 
1. Update the submodules under `external` folder and commit the changes.
1. Git push the branch to Github, Travis CI will start several builds to publish the documents to the PaddlePaddle.org server
1. Please notify the PaddlePaddle.org team that the release content is ready. PaddlePaddle.org team should enable the version and update the default version to the latest one. PaddlePaddle.org should also update the search index accordingly (Until the search server is up)

## 发布新的分支
1. 创建一个新的分支，此分支的名字应遵循`release/<version>`
1. 在FluidDoc和子模块中更新文档
1. 确认所有的子模块中处于发布就绪的状态。Paddle, book, model, mobile and Anakin 应全部有稳定的commit
请注意：如果Paddle Repo更改了module/classes，涉及API文档的RST文件应当也被更新
1. 在 `external` 中更新文件然后commit文档变更
1. 将这个分支push到Github，Travis CI将会启动几项构建工作以把文档发布到PaddlePaddle.org的服务器
1. 请告知PaddlePaddle.org团队，发布的内容已经就绪。PaddlePaddle.org团队将使版本生效并更新默认的版本到最新版。PaddlePaddle.org也应当更新相应的搜索引擎文件


