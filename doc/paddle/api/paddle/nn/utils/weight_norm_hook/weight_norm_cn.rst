.. _cn_api_nn_cn_weight_norm:

weight_norm
-------------------------------

.. py:function:: paddle.nn.utils.weight_norm(layer, name='weight', dim=0)

该接口根据以下公式对传入的 ``layer`` 中的权重参数进行归一化:

.. math::
    \mathbf{w} = g \dfrac{v}{\|v\|}

权重归一化可以将神经网络中权重向量的长度与其方向解耦，权重归一化可以用两个变量(例如: 代表长度的变量 `weight_g` 和代表方向的变量 `weight_v`)来代替由名字(例如: `weight`)指定的变量。详细可以参考论文: `A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_

参数：
   - **layer** (paddle.nn.Layer) - 要添加权重归一化的层。
   - **name** (str, 可选) - 权重参数的名字。默认：'weight'. 
   - **dim** (int|None, 可选) - 进行归一化操作的切片所在维度，是小于权重Tensor rank的非负数。比如卷积的权重shape是 [cout,cin,kh,kw] , rank是4，则dim可以选0,1,2,3；fc的权重shape是 [cout,cin] ，rank是2，dim可以选0，1。 如果为None就对所有维度上的元素做归一化。默认：0。 

返回：
   ``Layer`` , 添加了权重归一化hook的层

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.nn import Conv2D
    from paddle.nn.utils import weight_norm
    x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
    conv = Conv2D(3, 5, 3)
    wn = weight_norm(conv)
    print(conv.weight_g.shape)
    # [5]
    print(conv.weight_v.shape)
    # [5, 3, 3, 3]
