.. _cn_api_nn_cn_remove_weight_norm:

remove_weight_norm
-------------------------------

.. py:function:: paddle.nn.utils.remove_weight_norm(layer, name='weight')

移除传入 ``layer`` 中的权重归一化。

参数：
   - **layer** (paddle.nn.Layer) - 要添加权重归一化的层。
   - **name** (str, 可选) - 权重参数的名字。默认：'weight'. 

返回：
   ``Layer`` , 移除权重归一化hook之后的层

**代码示例**

.. code-block:: python

    import paddle
    from paddle.nn import Conv2D
    from paddle.nn.utils import weight_norm, remove_weight_norm
    paddle.disable_static()
    conv = Conv2D(3, 5, 3)
    wn = weight_norm(conv)
    remove_weight_norm(conv)
    # print(conv.weight_g)
    # AttributeError: 'Conv2D' object has no attribute 'weight_g'
