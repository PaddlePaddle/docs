.. _cn_api_fluid_dygraph_BilinearTensorProduct:

BilinearTensorProduct
-------------------------------

.. py:class:: paddle.fluid.dygraph.BilinearTensorProduct(name_scope, size, name=None, act=None, param_attr=None, bias_attr=None)

该层可将一对张量进行双线性乘积计算，例如：

.. math::

    out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

式中，

- :math:`x` ： 第一个输入，分别包含M个元素，维度为[batch_size, M]
- :math:`y` ：第二个输入，分别包含N个元素，维度为[batch_size, N]
- :math:`W_i` ：第i个学习到的权重，维度为[M,N]
- :math:`out_i` ：输出的第i个元素
- :math:`y^T` ： :math:`y` 的转置


参数：
    - **name_scope**  (str) – 指定类的名称。
    - **size**  (int) – 该层输出Tensor的最后一维大小。
    - **name**  (str，可选) – 该层的名称。若未设置，则自动生成该层的名称。默认值为None。
    - **act**  (str，可选) – 对输出应用的激励函数。默认值为None。
    - **param_attr**  (ParamAttr) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr**  (ParamAttr) – 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。

返回：维度为[batch_size, size]的2D Tensor

返回类型： Variable

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        layer1 = numpy.random.random((5, 5)).astype('float32')
        layer2 = numpy.random.random((5, 4)).astype('float32')
        bilinearTensorProduct = fluid.dygraph.nn.BilinearTensorProduct(
               'BilinearTensorProduct', size=1000)
        ret = bilinearTensorProduct(fluid.dygraph.base.to_variable(layer1),
                           fluid.dygraph.base.to_variable(layer2))




