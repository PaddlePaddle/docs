.. _cn_guides_api_mapping:

飞桨框架API映射表
=====================

本文档基于PaddlePaddle v1.X 梳理了常用API与PaddlePaddle v2.0对应关系。可根据对应关系，快速熟悉PaddlePaddle 2.0的接口使用。

.. note::

    其中，迁移工具能否转换，是指使用迁移工具能否直接对PaddlePaddle 1.X的API进行迁移，了解更多关于迁移工具的内容，请参考  :ref:`版本迁移工具 <cn_guides_migration>` 

..  csv-table::
    :header: "序号", "PaddlePaddle 1.X API", "PaddlePaddle 2.0 API", "迁移工具能否转换"
    :widths: 1, 8, 8, 8

    "0",  "`paddle.fluid.BuildStrategy <https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/fluid_cn/BuildStrategy_cn.html>`_ ",  "`paddle.static.BuildStrategy <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/compiler/BuildStrategy_cn.html>`_", "True"