.. _cn_api_paddle_scatter_:

scatter\_
-------------------------------

.. caution::

    本接口根据输入参数的不同，包含两种不同的功能。

    下面列举的两种功能参数输入方式 **互斥**，混用非公共的参数输入方法将会导致报错，请谨慎使用。

=====

.. py:function:: paddle.scatter_(x, index, updates, overwrite=True, name=None)

Inplace 版本的 :ref:`cn_api_paddle_scatter` API，对输入 `x` 采用 Inplace 策略。

=====

.. py:function:: paddle.scatter_(input, dim, index, src, reduce=None, out=None)

PyTorch 兼容 Inplace 版本的 :ref:`cn_api_paddle_scatter` API，对输入 `input` 采用 Inplace 策略。


更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
