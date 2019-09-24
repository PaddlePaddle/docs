.. _cn_api_fluid_layers_elementwise_mul:

elementwise_mul
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_mul(x, y, axis=-1, act=None, name=None)

该函数是逐元素相乘算子，输入x逐元素与输入y相乘，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        Out = X \odot Y

- :math:`X` ：任意维度的张量（Tensor）。
- :math:`Y` ：一个维度必须小于等于X维度的张量（Tensor）。

对于这个运算算子有2种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2:
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 是 :math:`Y` 在 :math:`X` 上的起始维度的位置。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis= rank(X)-rank(Y)` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾部维度将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: text

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

参数：
        - **x** （Variable）- 第一个输入张量（Tensor）。数据类型为 ``int`` 或 ``float32`` 或 ``float64`` 。
        - **y** （Variable）- 第二个输入张量（Tensor）。数据类型为 ``int`` 或 ``float32`` 或 ``float64`` 。
        - **axis** （int | -1）- Y的维度对应到X维度上时的索引。
        - **act** （basestring | None）- 激活函数名称，作用于输出上，例如 "relu"。
        - **name** （basestring | None）- 输出的名字。

返回：        维度与输入x相同的 ``Tensor`` 或 ``LoDTensor``, 数据类型与x相同。

返回类型：        Variable。

**代码示例 1**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    def gen_data():
        return {
            "x": np.array([2, 3, 4]),
            "y": np.array([1, 5, 2])
        }
    x = fluid.layers.data(name="x", shape=[3], dtype='float32')
    y = fluid.layers.data(name="y", shape=[3], dtype='float32')
    z = fluid.layers.elementwise_mul(x, y)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) #[2., 15., 8.]

**代码示例 2**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    def gen_data():
        return {
            "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
            "y": np.random.randint(1, 5, size=[3, 4]).astype('float32')
        }
    x = fluid.layers.data(name="x", shape=[2,3,4,5], dtype='float32')
    y = fluid.layers.data(name="y", shape=[3,4], dtype='float32')
    z = fluid.layers.elementwise_mul(x, y, axis=1)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) # z.shape=[2,3,4,5]

**代码示例 3**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    def gen_data():
        return {
            "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
            "y": np.random.randint(1, 5, size=[5]).astype('float32')
        }
    x = fluid.layers.data(name="x", shape=[2,3,4,5], dtype='float32')
    y = fluid.layers.data(name="y", shape=[3,4], dtype='float32')
    z = fluid.layers.elementwise_mul(x, y, axis=3)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) # z.shape=[2,3,4,5]







