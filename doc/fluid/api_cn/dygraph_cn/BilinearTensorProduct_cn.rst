.. _cn_api_fluid_dygraph_BilinearTensorProduct:

BilinearTensorProduct
-------------------------------

.. py:class:: paddle.fluid.dygraph.BilinearTensorProduct(name_scope, size, name=None, act=None, param_attr=None, bias_attr=None)

该层可将一对张量进行双线性乘积计算，例如：

.. math::

    out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

式中，

- :math:`x` ： 第一个输入，分别包含M个元素，形为[batch_size, M]
- :math:`y` ：第二个输入，分别包含N个元素，形为[batch_size, N]
- :math:`W_i` ：第i个学习到的权重，形为[M,N]
- :math:`out_i` ：输出的第i个元素
- :math:`y^T` ： :math:`y_2` 的转置


参数：
    - **name_scope**  (str) – 类的名称。
    - **size**  (int) – 该层的维度大小。
    - **act**  (str) – 对输出应用的激励函数。默认:None。
    - **name**  (str) – 该层的名称。 默认: None。
    - **param_attr**  (ParamAttr) – 该层中可学习权重/参数w的参数属性。默认: None.
    - **bias_attr**  (ParamAttr) – 该层中偏置(bias)的参数属性。若为False, 则输出中不应用偏置。如果为None, 偏置默认为0。默认: None.

返回：形为 [batch_size, size]的二维张量

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




