.. _cn_api_fluid_layers_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.layers.one_hot(input, depth, allow_out_of_range=False)




**注意：此OP要求输入Tensor shape的最后一维必须为1。此OP将在未来的版本中被移除！推荐使用fluid.** :ref:`cn_api_fluid_one_hot` 。

该OP将输入（input）中的每个id转换为一个one-hot向量，其长度为 ``depth``，该id对应的向量维度上的值为1，其余维度的值为0。

输出的Tensor（或LoDTensor）的shape是将输入shape的最后一维替换为depth的维度。

- 示例1（allow_out_of_range=False）：

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

- 示例2 （allow_out_of_range=True）：

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

- 示例3 （allow_out_of_range=False）：

.. code-block:: python
  
  输入：
    X.shape = [4, 1]
    X.data = [[1], [1], [5], [0]]
    depth = 4
    allow_out_of_range=False

  输出：抛出 Illegal value 的异常
    X中第2维的值是5，超过了depth，而allow_out_of_range=False表示不允许超过，因此抛异常。


参数
::::::::::::

    - **input** (Variable) - 维度为 :math:`[N_1, ..., N_n, 1]` 的多维Tensor或LoDTensor，维度至少两维，且最后一维必须是1。数据类型为int32或int64。
    - **depth** (int) - 用于定义一个one-hot向量的长度。若输入为词id，则 ``depth`` 通常取值为词典大小。
    - **allow_out_of_range** (bool) - 指明input中所包含的id值是否可以大于depth值。当超过depth时，如果 `allow_out_of_range` 为False，则会抛出 `Illegal value` 的异常；如果设置为True，该id对应的向量为0向量。默认值为False。

返回
::::::::::::
转换后的one_hot Tensor或LoDTensor，数据类型为float32。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    # 该代码对应上述第一个示例，其中输入label的shape是[4, 1]，输出one_hot_label的shape是[4, 4]
    label = fluid.layers.data(name="label", shape=[4, 1], append_batch_size=False, dtype="int64")
    one_hot_label = fluid.layers.one_hot(input=label, depth=4)
