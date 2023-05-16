.. _cn_api_tensor_cn_less_than:

less_than
-------------------------------
.. py:function:: paddle.less_than(x, y, name=None)


逐元素地返回 :math:`x < y` 的逻辑值，相同位置前者输入小于后者输入则返回 True，否则返回 False。使用重载算子 `<` 可以有相同的计算函数效果。

.. note::
    输出的结果不返回梯度。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float16、float32、float64、int32、int64。
    - **y** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
输出结果的 Tensor，输出 Tensor 的 shape 和输入一致，Tensor 数据类型为 bool。


代码示例
::::::::::::

COPY-FROM: paddle.less_than
