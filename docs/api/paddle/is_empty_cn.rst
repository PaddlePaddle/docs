.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.is_empty(x, name=None)




测试输入 Tensor x 是否为空。

参数
::::::::::::

   - **x** (Tensor) - 测试的 Tensor。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，布尔类型的 Tensor，如果输入 Tensor x 为空则值为 True。


代码示例
::::::::::::

COPY-FROM: paddle.is_empty
