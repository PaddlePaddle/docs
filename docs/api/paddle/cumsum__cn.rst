.. _cn_api_paddle_cumsum_:

cumsum\_
-------------------------------

.. py:function:: paddle.cumsum_(x, axis=None, dtype=None, name=None)
Inplace 版本的 :ref:`cn_api_paddle_cumsum` API，对输入 `x` 采用 Inplace 策略。

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor。
    - **axis** (int，可选) - 累加的维度。别名 ``dim``。
    - **dtype** (str|paddle.dtype|np.dtype|None，可选) - 输出 Tensor 的数据类型。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

 .. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
