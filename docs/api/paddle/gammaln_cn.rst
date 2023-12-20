.. _cn_api_paddle_gammaln:

gammaln
-------------------------------

.. py:function:: gammaln(x, name=None)

逐元素计算伽马函数的绝对值的自然对数。

参数
::::::::::::

    - **x** - 输入 Tensor。数据类型必须为 float16, float32, float64, bfloat16。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出为 Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.gammaln
