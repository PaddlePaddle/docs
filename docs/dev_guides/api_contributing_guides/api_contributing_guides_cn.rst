#####################
新增API 开发&提交流程
#####################

飞桨作为一个开源项目，我们鼓励生态开发者为paddlepaddle贡献API，当你想要为飞桨开发新API功能时，请遵守此API贡献流程在Github上完成文档设计和代码设计并提交至相应的github仓库。


API 贡献流程如下
::::::::::::::::::::::

.. image:: ../images/api_contributing_guides.png
  :width: 1000
  :alt: api_contributing_guides
  :align: center



流程介绍
::::::::::::::::::::::

**1、任务认领**

如果你想参与飞桨 API 开源贡献，可以在Github paddle 项目上的issue 区域进行任务认领，飞桨官网会发布一些新增 API 的任务，用户可以认领官方的开发任务，也可以产出自己的新增API想法，并按照此贡献流程提交设计文档。

**2、签订贡献者许可协议（CLA）**

对于你贡献的源代码，你将拥有合法的知识产权，为了保护你的权益，你需要先签署一份 `贡献者许可协议 <https://cla-assistant.io/PaddlePaddle/Paddle?pullRequest=39047>`_ 。

注意：当你签署完CLA后，我们才会继续对您提交的设计方案和实现代码进行评审及合入

**3、提交API设计文档**

当你想要发起一个新增API的贡献时，你需要先提交一份新增API 设计文档，并确保你的API 设计遵守 `飞桨API 设计及命名规范 <./api_design_guidelines_standard_cn.html>`_ 。API设计文档的目的是为了社区开发者更容易的参与开源项目共建，开发者通过与飞桨官方和社区其他用户进行广泛的交流，完善设计方案和pr请求，在后续提交实现代码之前确保API设计方案与飞桨设计理念一致，也让后续代码评审及代码合入变得更加容易。

同时，飞桨为大家提供了 `API 设计文档模版 <https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs/api_design_template.md>`_ 和 `API 设计文档demo <https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20200301_api_design_for_quantile.md>`_ ，你可以使用这份模版撰写API设计文档。完成后，你需要将设计文档提交至 `Github开发者社区仓库 <https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs>`_ ，并根据 `本地开发指南 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/local_dev_guide_cn.html>`_ 提交PR。

**4、设计文档评审&公示**

飞桨官方会及时安排专家对你提交的API设计文档进行审核，同时此文档也将接受来自开发者社区的评估，大家可以在pr评论区进行广泛的交流，开发者根据飞桨专家和其他开发者的反馈意见进行讨论并做出修改，最终评审通过后会在社区中公示3天，并准备合入。

如果你的API功能比较复杂，官方可能会在社区中针对API设计文档发起评审会议，会提前在pr评论区公布会议时间、会议地址、参会人、议题等内容，请及时关注pr 中最新动态，你也可以在评论区自行申请评审会。会议结束后，我们会在pr中发出会议结论。


**5、公布评审结果&合入文档**

当设计文档评审&公示通过后，官方会在开源社区中公布评审结果，你的API设计文档将会合入至飞桨开发者社区仓库。

**6、提交API实现代码**

当API设计文档合入后，开发者根据评审通过的API设计内容进行coding开发。coding过程请参考相应的开发规范，并提交以下内容：

.. csv-table::
    :header: "提交内容", "参考文档", "提交位置"
    :widths: 15, 30, 30

    "API实现代码 ", "- 飞桨API设计及命名规范 
    - Python API开发指南（请期待） 
    - C++ API开发指南（请期待）", "`Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_"
    "API英文文档 ", "- API文档规范", "`Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_"
    "API中文文档 ", "- API文档规范", "`Github飞桨文档仓库 <https://github.com/PaddlePaddle/docs>`_"
    "API单测代码 ", "- API验收标准", "`Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_"


当用户完成以上代码设计后，需要将代码提交至 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ ，并根据 `本地开发指南 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/local_dev_guide_cn.html>`_ 提交PR、准备接受社区的评审。

**7、实现代码评审&公示**

飞桨官方会及时安排专家进行API代码审核，代码也将接受来自开发者社区的评审，开发者可以在pr评论区进行广泛的交流，开发者对飞桨专家和其他开发者的反馈意见进行讨论并做出修改，最终评审通过后会在社区中公示3天，并准备合入。

如果你的API 功能比较复杂，官方可能会在社区中针对API实现代码发起评审会议，会提前在pr评论区公布会议时间、会议地址、参会人、议题等内容，请及时关注pr 中最新动态，你也可以在评论区自行申请评审会。会议结束后，我们会在pr 中发出会议结论。

**8、公布评审结果&合入代码**

当设计文档评审&公示通过后，官方会在开源社区中公布评审结果，你的API 实现代码将会合入至 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ 。

**9、通过API测试及验收**

当你的代码合入 `Github飞桨训练框架仓库 <https://github.com/PaddlePaddle/Paddle>`_ 后，官方会对你的代码进行集成测试，并通知你测试结果。如果测试通过，恭喜你贡献流程已经全部完成；如果测试不通过，我们会联系你进行代码修复，请及时关注github上的最新动态；

**10、贡献完成**

感谢您的贡献！

**引用文献**

..  toctree::
    :maxdepth: 1

    api_design_guidelines_standard_cn.md
    api_dosc_guidelines_cn.md
    api_accpetance_criteria_cn.md
