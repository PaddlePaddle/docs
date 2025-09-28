.. _cn_api_paddle_nn_initializer_Dirac:

Dirac
-------------------------------

.. py:class:: paddle.nn.initializer.Dirac(groups=1, name=None)


通过 ``狄拉克 delta 函数`` 来初始化 3D/4D/5D Tensor。

该初始化方式一般用于 Conv1D/Conv2D/Conv3D 卷积层，能尽可能多的保留卷积层输入的特性。（如果 `out_channels` > `in_channels`，则可保留全部的输入 `channel` 特性）

被初始化的参数，每个卷积核中间的元素会被置为 1，其余元素为 0。公式可以描述为：

.. math::

    X[d, d, shape[2]//2, shape[3]//2, ...]=1 ; d=0,1...N

其中 N 为 `out_channels` 和 `in_channels` 中的较小值。


参数
:::::::::
    - **groups** (int，可选) - 将参数在 0 维上进行等分为 `groups` 份，每一份执行相同的初始化。默认：1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
该参数初始化的类实例对象

代码示例
:::::::::

COPY-FROM: paddle.nn.initializer.Dirac
