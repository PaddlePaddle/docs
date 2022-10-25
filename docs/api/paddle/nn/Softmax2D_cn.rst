.. _cn_api_nn_Softmax2D:

Softmax2D
-------------------------------
.. py:class:: paddle.nn.Softmax2D(name=None)

Softmax2D是Softmax的变体，其针对 3D 或者 4D 的 `Tensor` 在空间维度上计算softmax。具体来说，输出的 `Tensor` 的每个空间维度 :math:`(channls, h_i, w_j)` 求和为1。

Softmax的详细介绍请参考 :ref:`cn_api_nn_Softmax`

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。


形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Softmax2D
