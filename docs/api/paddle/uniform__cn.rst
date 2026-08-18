.. _cn_api_paddle_uniform_:

uniform\_
-------------------------------

.. py:function:: paddle.uniform_(x, min=0, max=1.0, seed=0, name=None)

Inplace 版本的 :ref:`cn_api_paddle_uniform` API，对输入 ``x`` 采用 Inplace 策略。

参数 ``min`` 和 ``max`` 分别支持别名 ``from`` 和 ``to``。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
