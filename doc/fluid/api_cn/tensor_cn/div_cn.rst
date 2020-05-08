.. _cn_api_tensor_div:

div
-------------------------------

.. py:function:: paddle.div(x, y, out=None, name=None)

该OP是逐元素相除算子，输入 ``x`` 与输入 ``y`` 逐元素相除，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        Out = X / Y

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
        - **out** （Variable，可选）-  指定存储运算结果的 ``Tensor`` 。如果设置为None或者不设置，将创建新的 ``Tensor`` 存储运算结果，默认值为None。
        - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回：        多维 ``Tensor`` 或 ``LoDTensor`` ， 维度和数据类型都与 ``x`` 相同。

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
    z = paddle.div(x, y)
    # z = x / y

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])

    print(z_value) # [2., 0.6, 2.]


**代码示例 2**

.. code-block:: python

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
    z = paddle.div(x, y, name='z')
    # z = x / y

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
    output = fluid.data(name="output", shape=[2,3,4,5], dtype="float32")
    z = paddle.div(x, y, out=output)
    # z = x / y

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])
    print(z_value[0])
    print(z_value[0].shape) # z.shape=[2,3,4,5]


**代码示例 4（动态图）**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard(fluid.CPUPlace()):
        np_x = np.array([2, 3, 4]).astype('float64')
        np_y = np.array([1, 5, 2]).astype('float64')
        x = fluid.dygraph.to_variable(np_x)
        y = fluid.dygraph.to_variable(np_y)
        z = paddle.div(x, y)
        np_z = z.numpy()
        print(np_z)  # [2., 0.6, 2.]




