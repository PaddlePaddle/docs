.. _cn_api_nn_functional_layer_norm:

layer_norm
-------------------------------

.. py:function:: paddle.nn.functional.layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None)

推荐使用 nn.LayerNorm。

详情见 :ref:`cn_api_nn_LayerNorm` 。

参数
::::::::::::

    - **x** (int) - 输入，数据类型为 float32, float64。
    - **normalized_shape** (int|list|tuple) - 期望的输入是 :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`，如果是一个整数，会作用在最后一个维度。
    - **weight** (Tensor，可选) - 权重的 Tensor，默认为 None。
    - **bias** (Tensor，可选) - 偏置的 Tensor，默认为 None。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.layer_norm
