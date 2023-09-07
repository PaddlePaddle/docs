.. _cn_api_paddle_numel:

numel
-------------------------------

.. py:function:: paddle.numel(x)


返回 shape 为[]的 0-D Tensor，其值为输入 Tensor 中元素的个数。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型为 int32、int64、float16、float32、float64、int32、int64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回值为 ``x`` 元素个数的 0-D Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.numel
