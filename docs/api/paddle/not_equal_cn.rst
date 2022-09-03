.. _cn_api_tensor_not_equal:

not_equal
-------------------------------
.. py:function:: paddle.not_equal(x, y, name=None)


返回 :math:`x!=y` 逐元素比较 x 和 y 是否相等，相同位置的元素不相同则返回 True，否则返回 False。使用重载算子 `!=` 可以有相同的计算函数效果。

.. note::
    输出的结果不返回梯度。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **y** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
Tensor，shape 和输入一致，数据类型为 bool。


代码示例
::::::::::::

COPY-FROM: paddle.not_equal
