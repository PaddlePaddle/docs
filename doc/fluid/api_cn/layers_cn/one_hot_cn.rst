.. _cn_api_fluid_layers_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.layers.one_hot(input, depth, allow_out_of_range=False)

该Op将每个输入的词（input），表示成一个实数向量（one-hot vector），其长度为字典大小（depth），每个维度对应一个字典里的每个词，除了这个词对应维度上的值是1，其他元素都是0。

- 示例1（allow_out_of_range=False，正确执行）：

.. code-block:: python

  输入：
    X.shape = [4, 1]
    X.data = [[1], [1], [3], [0]]
    depth = 4

  输出：
    Out.shape = [4, 4]
    Out.data = [[0., 1., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.]]

- 示例2 （allow_out_of_range=True，正确执行，超出部分填0）：

.. code-block:: python

  输入：
    X.shape = [4, 1]
    X.data = [[1], [1], [5], [0]]
    depth = 4
    allow_out_of_range=True

  输出：
    Out.shape = [4, 4]
    Out.data = [[0., 1., 0., 0.],
                [0., 1., 0., 0.], 
                [0., 0., 0., 0.], ## 这一维的值是5，超过了depth，因此填成0
                [1., 0., 0., 0.]]

- 示例3 （allow_out_of_range=False，抛出异常）：

.. code-block:: python
  
  输入：
    X.shape = [4, 1]
    X.data = [[1], [1], [5], [0]]
    depth = 4
    allow_out_of_range=False

  输出：抛出`Illegal value`的异常
    X中第2维的值是5，超过了depth，而allow_out_of_range=False表示不允许超过，因此抛异常。  


参数：
    - **input** (Variable) - 维度为 :math:[N_1, ..., 1] 的多维Tensor，维度至少两维，且最后一维必须是1。数据类型为int32或int64。
    - **depth** (int) - 字典大小
    - **allow_out_of_range** (bool) - 指明输入的词是否可以超过字典大小。当超过字典大小时，如果 `allow_out_of_range` 为False，则会抛出`Illegal value`的异常；如果设置为True，超出的部分会以0填充。

返回：实数向量，数据类型为float32

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[4, 1], dtype="int64")
    one_hot_label = fluid.layers.one_hot(input=label, depth=10)
