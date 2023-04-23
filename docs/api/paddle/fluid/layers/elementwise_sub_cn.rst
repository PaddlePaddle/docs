.. _cn_api_fluid_layers_elementwise_sub:

elementwise_sub
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)




该 OP 是逐元素相减算子，输入 ``x`` 与输入 ``y`` 逐元素相减，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
       Out = X - Y

- :math:`X`：多维 Tensor。
- :math:`Y`：维度必须小于等于 X 维度的 Tensor。

对于这个运算算子有 2 种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。

对于情况 2:
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 是 :math:`Y` 在 :math:`X` 上的起始维度的位置。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis= rank(X)-rank(Y)` 。
        3. 考虑到子序列，:math:`Y` 的大小为 1 的尾部维度将被忽略，例如 shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: text

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

参数
::::::::::::

        - **x** （Variable）- 多维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **y** （Variable）- 多维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **axis** （int32，可选）-  ``y`` 的维度对应到 ``x`` 维度上时的索引。默认值为 -1。
        - **act** （str，可选）- 激活函数名称，作用于输出上。默认值为 None。详细请参考 :ref:`api_guide_activations`，常见的激活函数有：``relu`` ``tanh`` ``sigmoid`` 等。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
        维度与 ``x`` 相同的 ``Tensor``，数据类型与 ``x`` 相同。

返回类型
::::::::::::
        Variable。

代码示例 1
::::::::::::

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
    z = fluid.layers.elementwise_sub(x, y)
    # z = x - y
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) # [1., -2., 2.]

代码示例 2
::::::::::::

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
    z = fluid.layers.elementwise_sub(x, y, axis=1)
    # z = x - y
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) # z.shape=[2,3,4,5]

代码示例 3
::::::::::::

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    def gen_data():
        return {
            "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
            "y": np.random.randint(1, 5, size=[5]).astype('float32')
        }
    x = fluid.layers.data(name="x", shape=[2,3,4,5], dtype='float32')
    y = fluid.layers.data(name="y", shape=[5], dtype='float32')
    z = fluid.layers.elementwise_sub(x, y, axis=3)
    # z = x - y
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value) # z.shape=[2,3,4,5]
