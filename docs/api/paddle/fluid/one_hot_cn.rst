.. _cn_api_fluid_layers_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.layers.one_hot(input, depth, allow_out_of_range=False)




**注意：此 OP 要求输入 Tensor shape 的最后一维必须为 1。此 OP 将在未来的版本中被移除！推荐使用 fluid.** :ref:`cn_api_fluid_one_hot` 。

该 OP 将输入（input）中的每个 id 转换为一个 one-hot 向量，其长度为 ``depth``，该 id 对应的向量维度上的值为 1，其余维度的值为 0。

输出的 Tensor（或 LoDTensor）的 shape 是将输入 shape 的最后一维替换为 depth 的维度。

- 示例 1（allow_out_of_range=False）：

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

- 示例 2 （allow_out_of_range=True）：

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
                [0., 0., 0., 0.], ## 这一维的值是 5，超过了 depth，因此填成 0
                [1., 0., 0., 0.]]

- 示例 3 （allow_out_of_range=False）：

.. code-block:: python

  输入：
    X.shape = [4, 1]
    X.data = [[1], [1], [5], [0]]
    depth = 4
    allow_out_of_range=False

  输出：抛出 Illegal value 的异常
    X 中第 2 维的值是 5，超过了 depth，而 allow_out_of_range=False 表示不允许超过，因此抛异常。


参数
::::::::::::

    - **input** (Variable) - 维度为 :math:`[N_1, ..., N_n, 1]` 的多维 Tensor，维度至少两维，且最后一维必须是 1。数据类型为 int32 或 int64。
    - **depth** (int) - 用于定义一个 one-hot 向量的长度。若输入为词 id，则 ``depth`` 通常取值为词典大小。
    - **allow_out_of_range** (bool) - 指明 input 中所包含的 id 值是否可以大于 depth 值。当超过 depth 时，如果 `allow_out_of_range` 为 False，则会抛出 `Illegal value` 的异常；如果设置为 True，该 id 对应的向量为 0 向量。默认值为 False。

返回
::::::::::::
转换后的 one_hot Tensor，数据类型为 float32。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    # 该代码对应上述第一个示例，其中输入 label 的 shape 是[4, 1]，输出 one_hot_label 的 shape 是[4, 4]
    label = fluid.layers.data(name="label", shape=[4, 1], append_batch_size=False, dtype="int64")
    one_hot_label = fluid.layers.one_hot(input=label, depth=4)
