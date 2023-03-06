.. _cn_api_nn_Softmax2D:

Softmax2D
-------------------------------
.. py:class:: paddle.nn.Softmax2D(name=None)

Softmax2D 是 Softmax 的变体，其针对 3D 或者 4D 的 ``Tensor`` 在空间维度上计算 Softmax。具体来说，输出的 ``Tensor`` 的每个空间维度 :math:`(channls, h_i, w_j)` 求和为 1。

Softmax 的详细介绍请参考 :ref:`cn_api_nn_Softmax`

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。


形状:
::::::::::
    - input: 任意形状的 Tensor。
    - output: 和 input 具有相同形状的 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Softmax2D
