.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.round(x, name=None)




将输入中的数值四舍五入到最接近的整数数值。

.. code-block:: text

    输入：
        x.shape = [4]
        x.data = [1.2, -0.9, 3.4, 0.9]

    输出：
        out.shape = [4]
        out.data = [1., -1., 3., 1.]

参数
::::::::::::


    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 float32，float64 或 float16。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.round
