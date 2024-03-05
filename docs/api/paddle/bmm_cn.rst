.. _cn_api_paddle_bmm:

bmm
-------------------------------

.. py:function:: paddle.bmm(x, y, name=None)




对输入 x 及输入 y 进行矩阵相乘。

两个输入的维度必须等于 3，并且矩阵 x 和矩阵 y 的第一维必须相等。同时矩阵 x 的第三维必须等于矩阵 y 的第二维。

例如：若 x 和 y 分别为 (b, m, k) 和 (b, k, n) 的矩阵，则函数的输出为一个 (b, m, n) 的矩阵。

参数
:::::::::

    - **x** (Tensor) - 输入变量，类型为 Tensor。
    - **y** (Tensor) - 输入变量，类型为 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，矩阵相乘后的结果。

代码示例
:::::::::

COPY-FROM: paddle.bmm
