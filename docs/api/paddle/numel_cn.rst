.. _cn_api_tensor_numel:

numel
-------------------------------

.. py:function:: paddle.numel(x)


返回一个长度为1并且元素值为输入 ``x`` 元素个数的 Tensor。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型为int32、int64、float16、float32、float64、int32、int64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
在静态图模式下，返回一个长度为1并且元素值为输入 ``x`` 元素个数的 Tensor；在动态图模式下，返回一个标量数值。


代码示例
::::::::::::

COPY-FROM: paddle.numel