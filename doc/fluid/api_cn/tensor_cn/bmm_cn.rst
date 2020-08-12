.. _cn_api_paddle_tensor_bmm:

bmm
-------------------------------

.. py:function:: paddle.tensor.bmm(x, y, name=None):



对输入x及输入y进行矩阵相乘。

两个输入的维度必须等于3，并且矩阵x和矩阵y的第一维必须相等

同时矩阵x的第二维必须等于矩阵y的第三维

例如：若x和y分别为（b, m, k）和 （b, k, n)的矩阵，则函数的输出为一个（b, m, n）的矩阵

参数
:::::::::
    
    -**x** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    -**y** (Tensor) : 输入变量，类型为 Tensor 或 LoDTensor。
    -**name** (str|None) : 该层名称（可选），如果设置为空，则自动为该层命名。

返回
:::::::::
``Tensor``，矩阵相乘后的结果。


示例
:::::::::

.. code-block:: python
    
    import paddle
    # In imperative mode:
    # size input1: (2, 2, 3) and input2: (2, 3, 2)
    input1 = np.array([[[1.0, 1.0, 1.0],[2.0, 2.0, 2.0]],[[3.0, 3.0, 3.0],[4.0, 4.0, 4.0]]])
    input2 = np.array([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],[[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
    paddle.enable_imperative()
    
    x = paddle.imperative.to_variable(input1)
    y = paddle.imperative.to_variable(input2)
    out = paddle.bmm(x, y)
    #output size: (2, 2, 2)
    #output value:
    #[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
    out_np = out.numpy()

