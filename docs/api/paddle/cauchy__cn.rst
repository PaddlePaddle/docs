.. _cn_api_paddle_cauchy_:

cauchy\_
-------------------------------

.. py:function:: paddle.cauchy_(x, loc = 0, scale = 1, name = None)



直接修改输入张量 ``x``，将所有元素替换为从柯西分布中随机采样的数值。

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，支持的数据类型为：float32、float64。
    - **loc** (scalar，可选) - 分布峰值的位置参数，数据类型为 float32 或 float64。默认值为 0，对应标准柯西分布。
    - **scale** (scalar，可选) - 表示分布峰值一半高度处的半宽度（HWHM），控制曲线的陡峭程度。必须为正数，数据类型为 float32 或 float64。默认值为 1，对应标准柯西分布。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，对输入的张量 ``x`` 直接进行了修改，数据类型与输入时相同。

代码示例
::::::::::::

COPY-FROM: paddle.cauchy_
