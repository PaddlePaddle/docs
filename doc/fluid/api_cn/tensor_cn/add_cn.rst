.. _cn_api_tensor_add:

add
-------------------------------

.. py:function:: paddle.add(x, y, alpha=1, out=None, name=None)

该OP是逐元素相加算子，输入 ``x`` 与输入 ``y`` 逐元素相加，并将各个位置的输出元素保存到返回结果中。

等式为：

.. math::
        Out = X + Y

- :math:`X` ：多维Tensor。
- :math:`Y` ：维度必须小于等于X维度的Tensor。

对于这个运算算子有2种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。

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
        - **x** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **y** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **alpha** （int|float，可选）- 输入y的缩放因子。默认值为1. 如果alpha不为1，本api计算公式变为 :math:`Out = X + alpha * Y`
        - **out** （Variable，可选）-  指定存储运算结果的 ``Tensor`` 。如果设置为None或者不设置，将创建新的 ``Tensor`` 存储运算结果，默认值为None。
        - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回：        多维 ``Tensor`` 或 ``LoDTensor`` ，维度和数据类型都与 ``x`` 相同。

返回类型：        Variable

**代码示例 1**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    def gen_data():
        return {
            "x": np.array([2, 3, 4]).astype('float32'),
            "y": np.array([1, 5, 2]).astype('float32')
        }

    x = fluid.data(name="x", shape=[3], dtype='float32')
    y = fluid.data(name="y", shape=[3], dtype='float32')
    z1 = paddle.add(x, y)
    z2 = paddle.add(x, y, alpha=10)
    # z = x + y

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z1.name, z2.name])

    print(z_value[0]) # [3., 8., 6.]
    print(z_value[1]) # [12. 53. 24.]

**代码示例 2**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    def gen_data():
        return {
            "x": np.ones((2, 3, 4, 5)).astype('float32'),
            "y": np.zeros((4, 5)).astype('float32')
        }

    x = fluid.data(name="x", shape=[2, 3, 4, 5], dtype='float32')
    y = fluid.data(name="y", shape=[4, 5], dtype='float32')
    z = paddle.add(x, y, name='z')
    # z = x + y

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])

    print(z_value[0])
    print(z_value[0].shape) # z.shape=[2,3,4,5]

**代码示例 3**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    def gen_data():
        return {
            "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
            "y": np.random.randint(1, 5, size=[5]).astype('float32')
        }

    x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
    y = fluid.data(name="y", shape=[5], dtype='float32')
    z = paddle.add(x, y)
    # z = x / y

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value[0])
    print(z_value[0].shape) # z.shape=[2,3,4,5]


**代码示例 4**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    x = fluid.data(name="x", shape=[3], dtype="float32")
    y = fluid.data(name='y', shape=[3], dtype='float32')

    output = fluid.data(name="output", shape=[3], dtype="float32")
    z = paddle.add(x, y, out=output)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    data1 = np.array([2, 3, 4], dtype='float32')
    data2 = np.array([1, 5, 2], dtype='float32')
    z_value = exe.run(feed={'x': data1,
                            'y': data2},
                            fetch_list=[z])
    print(z_value[0]) # [3. 8. 6.]


**代码示例 5（动态图）**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        np_x = np.array([2, 3, 4]).astype('float64')
        np_y = np.array([1, 5, 2]).astype('float64')
        x = fluid.dygraph.to_variable(np_x)
        y = fluid.dygraph.to_variable(np_y)
        z = paddle.add(x, y, alpha=-0.5)
        np_z = z.numpy()
        print(np_z)  # [1.5, 0.5, 3. ]
