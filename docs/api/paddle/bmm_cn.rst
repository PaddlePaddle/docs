.. _cn_api_paddle_tensor_bmm:

bmm
-------------------------------

.. py:function:: paddle.bmm(x, y, name=None):




对输入x及输入y进行矩阵相乘。

两个输入的维度必须等于3，并且矩阵x和矩阵y的第一维必须相等。同时矩阵x的第二维必须等于矩阵y的第三维。

例如：若 x 和 y 分别为 (b, m, k) 和 (b, k, n) 的矩阵，则函数的输出为一个 (b, m, n) 的矩阵。

参数
:::::::::

    - **x** (Tensor) - 输入变量，类型为 Tensor。
    - **y** (Tensor) - 输入变量，类型为 Tensor。
    - **name** (str，可选) - 操作的名称(可选，默认值为None)。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
Tensor，矩阵相乘后的结果。

代码示例
:::::::::

COPY-FROM: paddle.bmm

