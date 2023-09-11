.. _cn_api_paddle_static_nn_bilinear_tensor_product:

bilinear_tensor_product
-------------------------------


.. py:function:: paddle.static.nn.bilinear_tensor_product(x, y, size, act=None, name=None, param_attr=None, bias_attr=None)




该层对两个输入执行双线性 Tensor 积。

例如：

.. math::
       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

在这个公式中：

  - :math:`x`：第一个输入，包含 :math:`M` 个元素，形状为 [batch_size, M]。
  - :math:`y`：第二个输入，包含 :math:`N` 个元素，形状为 [batch_size, N]。
  - :math:`W_{i}`：第 :math:`i` 个被学习的权重，形状是 [M, N]。
  - :math:`out_{i}`：输出的第 :math:`i` 个元素，形状是 [batch_size, size]。
  - :math:`y^\mathrm{T}`: :math:`y_{2}` 的转置。

参数
::::::::::::

    - **x** (Variable): 2-D 输入 Tensor，形状为 [batch_size, M]，数据类型为 float32 或 float64。
    - **y** (Variable): 2-D 输入 Tensor，形状为 [batch_size, N]，数据类型与 **x** 一致。
    - **size** (int)：此层的维度。
    - **act** (str，可选)：应用到该层输出的激活函数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **param_attr** (ParamAttr，可选)：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选)：指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。

返回
::::::::::::
 Variable，一个形为 [batch_size, size] 的 2-D Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.bilinear_tensor_product
