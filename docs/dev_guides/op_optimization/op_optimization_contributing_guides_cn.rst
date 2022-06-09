#####################
算子性能优化 开发&提交流程
#####################

飞桨作为一个开源项目，我们鼓励生态开发者为Paddle框架贡献高性能算子，当开发者想要为飞桨开发或优化算子性能时，请遵守此下述贡献流程，在Github上完成文档设计和代码设计并提交至相应的github仓库。


算子性能优化贡献流程
::::::::::::::::::::::

.. image:: ../images/op_optimization_contributing_guides.png
  :width: 1000
  :alt: op_optimization_contributing_guides
  :align: center

流程介绍
::::::::::::::::::::::

**1. 任务认领**

如果您想参与飞桨 OP 开源贡献，可以在Github Paddle 项目上的 Issue 区域进行任务认领，飞桨官网会发布一些OP性能优化任务，开发者可以认领算子优化任务，并按照此贡献流程提交算子性能优化设计文档。

**2. 签订贡献者许可协议（CLA）**

对于您贡献的源代码，您将拥有合法的知识产权，为了保护您的权益，您需要先签署一份`贡献者许可协议 <https://cla-assistant.io/PaddlePaddle/Paddle?pullRequest=39047>`_ 。

**注意**：只有当您签署完CLA后，我们才能继续对您提交的设计方案和实现代码进行评审及合入

**3. 提交算子性能优化设计文档**

算子性能优化设计文档的目的是促进社区开发者更容易的参与开源项目共建，开发者通过与飞桨专家和社区其他用户进行广泛的交流，完善设计方案和PR请求，在提交实现代码之前确保OP性能优化方案设计方案符合飞桨的设计理念，同时也便于后续的代码评审及合入工作。

当开发者想要发起一个算子性能优化的贡献时，需要首先进行算子优化方案设计并设计文档。飞桨为大家提供了 算子性能优化设计文档模版 ，您可以使用这份模版编写设计文档。完成后，您需要将设计文档提交至 Github开发者社区仓库 ，并根据本地开发指南提交PR。

此过程请参考相应的开发规范，并提交以下内容：

.. csv-table::
    :header: "提交内容", "参考文档", "提交位置"
    :widths: 10, 30, 30

    "算子性能优化设计文档", "- `算子性能优化 设计文档模版 <https://github.com/PaddlePaddle/community/tree/master/rfcs/OPs/op_optimization_template.md>`_ 
    - `算子性能优化设计文 设计文档示例 <https://github.com/PaddlePaddle/community/blob/master/rfcs/OPs/20220607_op_optimization_for_quantile.md>`_ ", "`Github开发者社区仓库 <https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs>`_"


**4. 设计文档评审&公示**

飞桨专家对您提交的设计文档进行审核，同时此文档也将接受来自开发者社区的评估，大家可以在pr评论区进行广泛的交流。开发者根据飞桨专家和其他开发者的反馈意见进行讨论并做出修改，最终评审通过后会在开源社区中同步。

如果您的设计方案比较复杂，我们可能会在社区中针对算子的设计文档发起评审会议，会提前在pr评论区公布会议时间、会议地址、参会人、议题等内容，请及时关注pr中最新动态，您也可以在评论区自行申请评审会。会议结束后，我们会在pr中发出会议结论。

**5. 公布评审结果&合入文档**

当设计文档评审&公示通过后，您的算子性能优化设计文档将会合入至`飞桨开发者社区仓库 <https://github.com/PaddlePaddle/community>`_ ，并在开源社区中同步。

**6. 提交API实现代码**

随后，开发者根据评审通过的设计内容进行代码开发。此过程请参考相应的开发规范，并提交以下内容：

.. csv-table::
    :header: "提交内容", "参考文档", "提交位置"
    :widths: 10, 30,30

    "算子性能优化实现代码", "- `Paddle代码规范 <https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/style_guides_cn.html>`_
    - `C++ OP开发指南 <../api_contributing_guides/new_cpp_op_cn.html>`_
    - `OP Benchmark使用指南 <https://github.com/PaddlePaddle/benchmark/blob/master/api>`_
    - `算子性能优化 验收标准 <./op_optimization_accpetance_criteria_cn.html>`_
    ", "`Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_"

当开发者完成以上代码设计后，需要将代码提交至 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ ，并根据 `本地开发指南 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/local_dev_guide_cn.html>`_ 提交PR、准备接受社区的评审。

**7. 实现代码评审&公示**

飞桨官方会及时安排专家进行算子性能优化代码审核，代码也将接受来自开发者社区的评审，开发者可以在pr评论区进行广泛的交流，开发者对飞桨专家和其他开发者的反馈意见进行讨论并做出修改，最终评审通过后会在开源社区中同步。

如果您的代码实现逻辑比较复杂，官方可能会在社区中针对API实现代码发起评审会议，会提前在pr评论区公布会议时间、会议地址、参会人、议题等内容，请及时关注pr 中最新动态，您也可以在评论区自行申请评审会。会议结束后，我们会在pr 中发出会议结论。

**8. 公布评审结果&合入代码**

当算子优化代码评审&公示通过后，官方会在开源社区中同步，您所实现的优化代码将会合入至 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ 。

**9. 通过模型集成及验收**

当您的代码合入 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ 后，官方会对您的性能优化代码进行模型级集成测试，并通知您测试结果。如果测试通过，恭喜您贡献流程已经全部完成；如果测试不通过，我们会联系您进行代码修复，请及时关注 Github上的最新动态；

**注意**：代码合入Develop分之后的第二天您可以从官网下载develop 编译的安装包体验此功能。飞桨后续也会将此功能纳入正式版的发版计划.


**10. 贡献完成**

感谢您的贡献！


..  toctree::
    :hidden:

    op_optimization_accpetance_criteria_cn.md