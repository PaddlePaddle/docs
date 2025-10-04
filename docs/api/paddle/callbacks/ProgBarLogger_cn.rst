.. _cn_api_paddle_callbacks_ProgBarLogger:

ProgBarLogger
-------------------------------

.. py:class:: paddle.callbacks.ProgBarLogger(log_freq=1, verbose=2)

 ``ProgBarLogger`` 是一个日志回调类，用来打印损失函数和评估指标。支持静默模式、进度条模式、每次打印一行三种模式，详细的参考下面参数注释。

参数
::::::::::::

  - **log_freq** (int，可选) - 损失值和指标打印的频率。默认值：1。
  - **verbose** (int，可选) - 打印信息的模式。设置为 0 时，不打印信息；
    设置为 1 时，使用进度条的形式打印信息；设置为 2 时，使用行的形式打印信息。
    设置为 3 时，会在 2 的基础上打印详细的计时信息，比如 ``average_reader_cost``。
    默认值：2。


代码示例
::::::::::::

COPY-FROM: paddle.callbacks.ProgBarLogger
