.. _cn_api_paddle_bitwise_right_shift_:

bitwise_right_shift\_
-------------------------------

.. py:function:: paddle.bitwise_right_shift_(x, y, is_arithmetic=True, name=None)

Inplace 版本的 :ref:`cn_api_paddle_bitwise_right_shift` API，对输入 `x` 采用 Inplace 策略。

参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。别名 ``input``。
        - **y** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。别名 ``other``。
        - **is_arithmetic** （bool） - 用于表明是否执行算术位移，True 表示算术位移，False 表示逻辑位移。默认值为 True，表示算术位移。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3
