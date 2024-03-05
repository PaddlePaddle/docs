.. _cn_api_paddle_reciprocal:

reciprocal
-------------------------------

.. py:function:: paddle.reciprocal(x, name=None)




reciprocal 对输入 Tensor 取倒数


.. math::
    out = \frac{1}{x}

参数
::::::::::::


    - **x** - 输入的多维 Tensor，支持的数据类型为 float32，float64，complex64，complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
 对输入取倒数得到的 Tensor，输出 Tensor 数据类型和维度与输入相同。

代码示例
::::::::::::

COPY-FROM: paddle.reciprocal
