.. _cn_api_paddle_reduce_as:

reduce_as
-------------------------------

.. py:function:: paddle.reduce_as(x, target, name=None):
对x在某些维度上求和，使其结果与target的shape一致。


参数
::::::::::::

  - **x** (Tensor) - 输入变量为多维 Tensor，支持数据类型为 float16、float32、float64、int32、int64。
  - **target** (Tensor) - 输入变量为多维 Tensor，x的shape长度必须大于或等于target的shape长度。支持数据类型为 float16、float32、float64、int32、int64。
  - **name** (str, 可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输入x沿某些轴进行求和，使其结果与target的shape相同，如果x的type为bool或者为int32，返回值的type将变为int64，否则返回值的type与x相同。

代码示例
::::::::::::

COPY-FROM: paddle.reduce_as