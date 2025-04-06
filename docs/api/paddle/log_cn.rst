.. _cn_api_paddle_log:
log
-------------------------------
    paddle.log(x, name=None)
Log 激活函数（计算自然对数）

.. math:: 
    Out=ln(x)

参数
:::::::::
  - **x** (Tensor) – 输入的 Tensor。数据类型只能为 int32、int64、float16、bfloat16、float32、float64、 complex64 或 complex128
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor, Log 算子自然对数输出，数据类型与输入一致。

代码示例
::::::::::

>>>import paddle

>>>x = [[2, 3, 4], [7, 8, 9]]

>>>x = paddle.to_tensor(x, dtype='float32')

>>>print(paddle.log(x))

Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,

[[0.69314718, 1.09861231, 1.38629436],
 [1.94591010, 2.07944155, 2.19722462]])
