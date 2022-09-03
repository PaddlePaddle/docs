.. _cn_api_paddle_callbacks_VisualDL:

VisualDL
-------------------------------

.. py:class:: paddle.callbacks.VisualDL(log_dir)

 ``VisualDL`` 是一个 visualdl( `飞桨可视化分析工具 <https://github.com/PaddlePaddle/VisualDL>`_ )的回调类。该类将训练过程中的损失值和评价指标储存至日志文件中后，启动面板即可查看可视化结果。

参数
::::::::::::

  - **log_dir** (str) - 输出日志保存的路径。


代码示例
::::::::::::

COPY-FROM: paddle.callbacks.VisualDL
