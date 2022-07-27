.. _cn_api_paddle_callbacks_LRScheduler:

LRScheduler
-------------------------------

.. py:class:: paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)

 ``LRScheduler`` 是一个学习率回调函数。

参数
::::::::::::

  - **by_step** (bool，可选) - 是否每个step都更新学习率。默认值：True。
  - **by_epoch** (bool，可选) - 是否每个epoch都更新学习率。默认值：False。


代码示例
::::::::::::

COPY-FROM: paddle.callbacks.LRScheduler