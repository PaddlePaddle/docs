.. _cn_api_nn_initializer_Dirac:

Dirac
-------------------------------

.. py:class:: paddle.nn.initializer.Dirac(groups=1, name=None)


通过 ``狄拉克delta函数`` 来初始化3D/4D/5D Tensor。

该初始化方式一般用于 Conv1D/Conv2D/Conv3D 卷积层，能尽可能多的保留卷积层输入的特性。（如果 `out_channels` > `in_channels` ，则可保留全部的输入 `channel` 特性）

被初始化的参数，每个卷积核中间的元素会被置为1，其余元素为0。公式可以描述为：

.. math::

    X[d, d, shape[2]//2, shape[3]//2, ...]=1 ; d=0,1...N

其中 N 为 `out_channels` 和 `in_channels` 中的较小值。


参数
:::::::::
    - groups (int，可选) - 将参数在0维上进行等分为 `groups` 份，每一份执行相同的初始化。默认：1。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
该参数初始化的类实例对象

代码示例
:::::::::

.. code-block:: python

    import paddle
    
    #1. For kernel_size is uneven number:
    
    attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
    conv = paddle.nn.Conv1D(3, 2, 3, weight_attr=attr)
    conv.weight
    # Tensor(shape=[2, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
    #       [[[0., 1., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.]],
    # 
    #        [[0., 0., 0.],
    #         [0., 1., 0.],
    #         [0., 0., 0.]]])

    input = paddle.rand([8, 3, 10])
    output = conv(input)
    output == input[:, 0:2, 1:9]  
    # output.shape is [8, 2, 8], It means output is almost the same with input, 2 channels are reserved


    #2. For kernel_size is even number:
    attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
    conv = paddle.nn.Conv1D(3, 2, 4, weight_attr=attr)
    conv.weight
    # Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=False,
    #       [[[0., 0., 1., 0.],
    #         [0., 0., 0., 0.],
    #         [0., 0., 0., 0.]],
    # 
    #        [[0., 0., 0., 0.],
    #         [0., 0., 1., 0.],
    #         [0., 0., 0., 0.]]])
