.. _cn_api_fluid_layers_edit_distance:


edit_distance
-------------------------------

.. py:function:: paddle.fluid.layers.edit_distance(input,label,normalized=True,ignored_tokens=None)

编辑距离算子

计算一批给定字符串及其参照字符串间的编辑距离。编辑距离也称Levenshtein距离，通过计算从一个字符串变成另一个字符串所需的最少操作步骤来衡量两个字符串的相异度。这里的操作包括插入、删除和替换。

比如给定假设字符串A=“kitten”和参照字符串B=“sitting”，从A变换成B编辑距离为3，至少需要两次替换和一次插入：

“kitten”->“sitten”->“sittn”->“sitting”

输入为LoDTensor,包含假设字符串（带有表示批尺寸的总数）和分离信息（具体为LoD信息）。并且批尺寸大小的参照字符串和输入LoDTensor的顺序保持一致。

输出包含批尺寸大小的结果，代表一对字符串中每个字符串的编辑距离。如果Attr(normalized)为真，编辑距离则处以参照字符串的长度。

参数：
    - **input** (Variable)-假设字符串的索引
    - **label** (Variable)-参照字符串的索引
    - **normalized** (bool,默认为True)-表示是否用参照字符串的长度进行归一化
    - **ignored_tokens** (list<int>,默认为None)-计算编辑距离前需要移除的token
    - **name** (str)-该层名称，可选

返回：[batch_size,1]中序列到序列到编辑距离

返回类型：变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[1], dtype='int64')
    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    cost, _ = fluid.layers.edit_distance(input=x,label=y)

    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    import numpy
    x_ = numpy.random.randint(5, size=(2, 1)).astype('int64')
    y_ = numpy.random.randint(5, size=(2, 1)).astype('int64')
    
    print(x_)
    print(y_)
    
    x = fluid.create_lod_tensor(x_, [[2]], cpu)
    y = fluid.create_lod_tensor(y_, [[2]], cpu)
    
    outs = exe.run(feed={'x':x, 'y':y}, fetch_list=[cost.name])
    
    print(outs)









