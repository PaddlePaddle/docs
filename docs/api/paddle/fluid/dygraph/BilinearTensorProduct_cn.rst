.. _cn_api_fluid_dygraph_BilinearTensorProduct:

BilinearTensorProduct
-------------------------------

.. py:class:: paddle.fluid.dygraph.BilinearTensorProduct(input1_dim, input2_dim, output_dim, name=None, act=None, param_attr=None, bias_attr=None, dtype="float32")




该接口用于构建 ``BilinearTensorProduct`` 类的一个可调用对象，具体用法参照 ``代码示例``。双线性乘积计算式子如下。

.. math::

    out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

式中，

- :math:`x`：第一个输入，分别包含M个元素，维度为 :math:`[batch\_size, M]` 
- :math:`y`：第二个输入，分别包含N个元素，维度为 :math:`[batch\_size, N]` 
- :math:`W_i`：第i个学习到的权重，维度为 :math:`[M,N]` 
- :math:`out_i`：输出的第i个元素
- :math:`y^T` ： :math:`y` 的转置


参数
::::::::::::

    - **input1_dim**  (int) – 第一个输入的维度大小。
    - **input1_dim**  (int) – 第二个输入的维度大小。
    - **output_dim**  (int) – 输出的维度。
    - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。
    - **act**  (str，可选) – 对输出应用的激励函数。默认值为None。
    - **param_attr**  (ParamAttr) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr**  (ParamAttr) – 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值为"float32"。

返回
::::::::::::
Tensor，维度为[batch_size, size]的2D Tensor，数据类型与输入数据类型相同。

代码示例
::::::::::::

.. code-block:: python

    import paddle
    import numpy

    layer1 = numpy.random.random((5, 5)).astype('float32')
    layer2 = numpy.random.random((5, 4)).astype('float32')
    bilinearTensorProduct = paddle.nn.BilinearTensorProduct(
        input1_dim=5, input2_dim=4, output_dim=1000)
    ret = bilinearTensorProduct(paddle.to_tensor(layer1),
                                paddle.to_tensor(layer2))

属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter``


