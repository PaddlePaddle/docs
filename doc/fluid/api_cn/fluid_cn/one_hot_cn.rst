.. _cn_api_fluid_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.one_hot(input, depth, allow_out_of_range=False)

**注意：相对于 fluid.layers.** :ref:`cn_api_fluid_layers_one_hot` **接口，此OP移除了输入tensor shape的最后一维强制为1的限制，详细使用区别，请参考使用代码样例。**

该OP将输入（input）中的每个词id 转换为一个one-hot向量，其长度为字典大小（depth），该词id对应的向量维度上的值为1，其余维度的值为0。

输出的tensor的shape是在输入tensor shape的最后一维后面添加了depth的维度。

- 示例1（allow_out_of_range=False，正确执行）：

.. code-block:: python

  输入：
    X.shape = [4]
    X.data = [1, 1, 3, 0]
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
    X.shape = [4]
    X.data = [1, 1, 5, 0]
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
    X.shape = [4]
    X.data = [1, 1, 5, 0]
    depth = 4
    allow_out_of_range=False

  输出：抛出 Illegal value 的异常
    X中第2维的值是5，超过了depth，而allow_out_of_range=False表示不允许超过，因此抛异常。  


参数：
    - **input** (Variable) - 维度为 :math:`[N_1, ..., N_n]` 的多维Tensor，维度至少1维。数据类型为int32或int64。
    - **depth** (int) - 字典大小
    - **allow_out_of_range** (bool) - 指明input中所包含的id值是否可以大于depth值。当超过depth时，如果 `allow_out_of_range` 为False，则会抛出 `Illegal value` 的异常；如果设置为True，该id对应的向量为0向量。

返回：转换后的one-hot实数向量。

返回类型：Variable，数据类型为float32。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    # 该代码对应上述第一个示例，其中输入label的shape是[4]，输出one_hot_label的shape是[4, 4]
    label = fluid.layers.data(name="label", shape=[4], append_batch_size=False, dtype="int64")
    one_hot_label = fluid.one_hot(input=label, depth=4)









