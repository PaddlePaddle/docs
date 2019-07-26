.. _cn_api_fluid_layers_hash:

hash
-------------------------------

.. py:function::  paddle.fluid.layers.hash(input, hash_size, num_hash=1, name=None)

将输入 hash 到一个整数，该数的值小于给定的 hash size

我们使用的哈希算法是 xxHash - `Extremely fast hash algorithm <https://github.com/Cyan4973/xxHash/tree/v0.6.5>`_

提供一简单的例子：

.. code-block:: text

  给出：

    # shape [2, 2]
    input.data = [
        [[1, 2],
        [3, 4]],
    ]

    input.lod = [[0, 2]]

    hash_size = 10000

    num_hash = 4

  然后:

    哈希操作将这个二维input的所有数字作为哈希算法每次的输入。

    每个输入都将被哈希4次，最终得到一个长度为4的数组。

    数组中的每个值的范围从0到9999。



    # shape [2, 4]
    output.data = [
        [[9662, 9217, 1129, 8487],
        [8310, 1327, 1654, 4567]],
    ]

    output.lod = [[0, 2]]

参数：
  - **input** (Variable) - 输入变量是一个 one-hot 词。输入变量的维数必须是2。
  - **hash_size** (int) - 哈希算法的空间大小。输出值将保持在 :math:`[0, hash\_size - 1]` 范围内。
  - **num_hash** (int) - 哈希次数，默认为1。
  - **name** (str, default None) - 该层的名称

返回：哈希的结果变量，是一个lodtensor。

返回类型： Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    import numpy as np

    titles = fluid.layers.data(name='titles', shape=[1], dtype='int32', lod_level=1)
    hash_r = fluid.layers.hash(name='hash_x', input=titles, num_hash=1, hash_size=1000)

    place = fluid.core.CPUPlace()
    exece = fluid.Executor(place)
    exece.run(fluid.default_startup_program())

    # 初始化Tensor
    tensor = fluid.core.LoDTensor()
    tensor.set(np.random.randint(0, 10, (3, 1)).astype("int32"), place)
    # 设置LoD
    tensor.set_recursive_sequence_lengths([[1, 1, 1]])

    out = exece.run(feed={'titles': tensor}, fetch_list=[hash_r], return_numpy=False)









