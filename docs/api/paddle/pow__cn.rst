.. _cn_api_paddle_pow_:

pow\_
-------------------------------

.. py:function:: paddle.pow_(x, y, name=None)
Inplace 版本的 :ref:`cn_api_paddle_pow` API，对输入 ``x`` 采用 Inplace 策略。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，参数名 ``exponent`` 可替代 ``y``，如 ``pow_(input=tensor_x, exponent=2)`` 等价于 ``pow_(x=tensor_x, y=2)``。
