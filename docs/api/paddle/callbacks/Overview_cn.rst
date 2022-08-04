.. _cn_overview_callbacks:

paddle.callbacks
---------------------

paddle.callbacks 目录下包含飞桨框架支持的回调函数相关的 API。具体如下：

-  :ref:`回调函数相关 API <about_callbacks>`

.. _about_callbacks:

回调函数相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Callback <cn_api_paddle_callbacks_Callback>` ", "回调函数的基类，用于自定义回调函数"
    " :ref:`EarlyStopping <cn_api_paddle_callbacks_EarlyStopping>` ", "停止训练回调函数"
    " :ref:`LRScheduler <cn_api_paddle_callbacks_LRScheduler>` ", "学习率回调函数"
    " :ref:`ModelCheckpoint <cn_api_paddle_callbacks_ModelCheckpoint>` ", "保存模型日志回调类"
    " :ref:`ProgBarLogger <cn_api_paddle_callbacks_ProgBarLogger>` ", "打印损失和评估指标日志回调类"
    " :ref:`ReduceLROnPlateau <cn_api_paddle_callbacks_ReduceLROnPlateau>` ", "根据评估指标降低学习率回调函数"
    " :ref:`VisualDL <cn_api_paddle_callbacks_VisualDL>` ", "可视化回调函数"
