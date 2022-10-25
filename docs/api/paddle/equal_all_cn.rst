.. _cn_api_tensor_equal_all:

equal_all
-------------------------------

.. py:function:: paddle.equal_all(x, y, name=None)


返回 x==y 的布尔值，如果所有相同位置的元素相同返回 True，否则返回 False。

**注：输出的结果不返回梯度。**


参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **y** (Tensor) - 输入 Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出结果为 Tensor，Tensor 数据类型为 bool。

代码示例
::::::::::::

COPY-FROM: paddle.equal_all
