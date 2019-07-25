.. _cn_api_fluid_layers_tree_conv:

tree_conv
-------------------------------

.. py:function:: paddle.fluid.layers.tree_conv(nodes_vector, edge_set, output_size, num_filters=1, max_depth=2, act='tanh', param_attr=None, bias_attr=None, name=None)

基于树结构的卷积Tree-Based Convolution运算。

基于树的卷积是基于树的卷积神经网络（TBCNN，Tree-Based Convolution Neural Network）的一部分，它用于对树结构进行分类，例如抽象语法树。 Tree-Based Convolution提出了一种称为连续二叉树的数据结构，它将多路（multiway）树视为二叉树。 提出基于树的卷积论文： https：//arxiv.org/abs/1409.5718v1

参数：
    - **nodes_vector**  (Variable) – (Tensor) 树上每个节点的特征向量(vector)。特征向量的形状必须为[max_tree_node_size，feature_size]
    - **edge_set**  (Variable) – (Tensor) 树的边。边必须带方向。边集的形状必须是[max_tree_node_size，2]
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
    # 10 代表数据集的最大节点大小max_node_size，5 代表向量宽度
    nodes_vector = fluid.layers.data(name='vectors', shape=[10, 5], dtype='float32')
    # 10 代表数据集的最大节点大小max_node_size, 2 代表每条边连接两个节点
    # 边必须为有向边
    edge_set = fluid.layers.data(name='edge_set', shape=[10, 2], dtype='float32')

    # 输出的形状会是[None, 10, 6, 1],
    # 10 代表数据集的最大节点大小max_node_size, 6 代表输出大小output size, 1 代表 1 个filter
    
    out_vector = fluid.layers.tree_conv(nodes_vector, edge_set, 6, 1, 2)
    # reshape之后, 输出张量output tensor为下一个树卷积的nodes_vector
    out_vector = fluid.layers.reshape(out_vector, shape=[-1, 10, 6])
    
    
    out_vector_2 = fluid.layers.tree_conv(out_vector, edge_set, 3, 4, 2)
    
    # 输出tensor也可以用来池化(论文中称为global pooling)
    pooled = fluid.layers.reduce_max(out_vector, dims=2) # 全局池化






