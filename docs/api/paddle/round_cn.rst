.. _cn_api_paddle_round:

round
-------------------------------

.. py:function:: paddle.round(x, decimals=0, name=None)




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


    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 bfloat16，float32，float64 或 float16。
    - **decimals** (int，可选)  - 要舍入到的小数点位数。如果 decimals 为负数，则指定小数点左边的位数。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.round
