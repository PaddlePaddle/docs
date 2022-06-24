.. _cn_api_fluid_dygraph_TreeConv:

TreeConv
-------------------------------

.. py:class:: paddle.fluid.dygraph.TreeConv(feature_size, output_size, num_filters=1, max_depth=2, act='tanh', param_attr=None, bias_attr=None, name=None, dtype="float32")




该接口用于构建 ``TreeConv`` 类的一个可调用对象，具体用法参照 ``代码示例``。其将在神经网络中构建一个基于树结构的卷积（Tree-Based Convolution）运算。基于树的卷积是基于树的卷积神经网络（TBCNN，Tree-Based Convolution Neural Network）的一部分，它用于对树结构进行分类，例如抽象语法树。Tree-Based Convolution提出了一种称为连续二叉树的数据结构，它将多路（multiway）树视为二叉树。详情请参考：`基于树的卷积论文 <https://arxiv.org/abs/1409.5718v1>`_ 。


参数
::::::::::::

    - **feature_size**  (int) – nodes_vector的shape的最后一维的维度。
    - **output_size**  (int) – 输出特征宽度。
    - **num_filters**  (int，可选) – 滤波器的数量，默认值为1。
    - **max_depth**  (int，可选) – 滤波器的最大深度，默认值为2。
    - **act**  (str，可选) – 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations`，默认值为None。
    - **param_attr**  (ParamAttr，可选) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr**  (ParamAttr，可选) – 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值为"float32"。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python
    
    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        nodes_vector = numpy.random.random((1, 10, 5)).astype('float32')
        edge_set = numpy.random.random((1, 9, 2)).astype('int32')
        treeConv = fluid.dygraph.nn.TreeConv(
          feature_size=5, output_size=6, num_filters=1, max_depth=2)
        ret = treeConv(fluid.dygraph.base.to_variable(nodes_vector), fluid.dygraph.base.to_variable(edge_set))

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

