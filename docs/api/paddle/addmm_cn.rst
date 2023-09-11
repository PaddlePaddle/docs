.. _cn_api_paddle_addmm:


addmm
-------------------------------

.. py:function:: paddle.addmm(input, x, y, alpha=1.0, beta=1.0, name=None)




计算 x 和 y 的乘积，将结果乘以标量 alpha，再加上 input 与 beta 的乘积，得到输出。其中 input 与 x、y 乘积的维度必须是可广播的。

计算过程的公式为：

..  math::
    out = alpha * x * y + beta * input

参数
::::::::::::

    - **input** (Tensor) - 输入 Tensor input，数据类型支持 float32、float64。
    - **x** (Tensor) - 输入 Tensor x，数据类型支持 float32、float64。
    - **y** (Tensor) - 输入 Tensor y，数据类型支持 float32、float64。
    - **alpha** (float，可选) - 乘以 x*y 的标量，数据类型支持 float32、float64，默认值为 1.0。
    - **beta** (float，可选) - 乘以 input 的标量，数据类型支持 float32、float64，默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
计算得到的 Tensor。Tensor 数据类型与输入 input 数据类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.addmm
