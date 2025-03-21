.. _cn_api_paddle_normal_:

normal\_
-------------------------------

.. py:function:: paddle.normal_(x, mean=0.0, std=1.0, name=None)

Inplace 版本的 :ref:`cn_api_paddle_normal` API，对输入 x 采用 Inplace 策略。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3


参数
::::::::::
    - **x** (Tensor) - 随机值填充的输入 Tensor。
    - **mean** (float|complex|Tensor，可选) - 输出 Tensor 的正态分布的平均值，默认值为 0.0。
    - **std** (float|Tensor，可选) - 输出 Tensor 的正态分布的标准差，默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::
   Tensor：符合正态分布（均值为 ``mean``，标准差为 ``std`` 的正态随机分布）的随机 Tensor。

示例代码
::::::::::

 COPY-FROM: paddle.normal_
