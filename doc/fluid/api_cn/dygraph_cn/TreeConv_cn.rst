.. _cn_api_fluid_dygraph_TreeConv:

TreeConv
-------------------------------

.. py:class:: paddle.fluid.dygraph.TreeConv(name_scope, output_size, num_filters=1, max_depth=2, act='tanh', param_attr=None, bias_attr=None, name=None)

基于树结构的卷积Tree-Based Convolution运算。

基于树的卷积是基于树的卷积神经网络（TBCNN，Tree-Based Convolution Neural Network）的一部分，它用于对树结构进行分类，例如抽象语法树。 Tree-Based Convolution提出了一种称为连续二叉树的数据结构，它将多路（multiway）树视为二叉树。提出 `基于树的卷积论文 <https://arxiv.org/abs/1409.5718v1>`_


参数：
    - **name_scope**  (str) – 该类的名称
    - **output_size**  (int) – 输出特征宽度
    - **num_filters**  (int) – filter数量，默认值1
    - **max_depth**  (int) – filter的最大深度，默认值2
    - **act**  (str) – 激活函数，默认 tanh
    - **param_attr**  (ParamAttr) – filter的参数属性，默认None
    - **bias_attr**  (ParamAttr) – 此层bias的参数属性，默认None
    - **name**  (str) – 此层的名称（可选）。如果设置为None，则将自动命名层，默认为None


返回： （Tensor）子树的特征向量。输出张量的形状是[max_tree_node_size，output_size，num_filters]。输出张量可以是下一个树卷积层的新特征向量

返回类型：out（Variable）

**代码示例**:

.. code-block:: python
    
    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        nodes_vector = numpy.random.random((1, 10, 5)).astype('float32')
        edge_set = numpy.random.random((1, 9, 2)).astype('int32')
        treeConv = fluid.dygraph.nn.TreeConv(
          'TreeConv', output_size=6, num_filters=1, max_depth=2)
        ret = treeConv(fluid.dygraph.base.to_variable(nodes_vector), fluid.dygraph.base.to_variable(edge_set))







