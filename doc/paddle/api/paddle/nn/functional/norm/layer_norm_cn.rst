.. _cn_api_nn_functional_layer_norm:

layer_norm
-------------------------------

.. py:class:: paddle.nn.functional.layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):

推荐使用nn.LayerNorm。

详情见 :ref:`cn_api_nn_LayerNorm` . 

参数：
    - **x** (int) - 输入，数据类型为float32, float64。
    - **normalized_shape** (int|list|tuple) - 期望的输入是 :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]` ,如果是一个整数，会作用在最后一个维度。
    - **weight** (Tensor) - 权重的Tensor, 默认为None。
    - **bias** (Tensor) - 偏置的Tensor, 默认为None。
    - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **name** (string, 可选) – LayerNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。


返回：无

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    np.random.seed(123)
    x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    layer_norm_out = paddle.nn.functional.layer_norm(x, x.shape[1:])

    print(layer_norm_out)

