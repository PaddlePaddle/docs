.. _cn_api_fluid_layers_elementwise_pow:

elementwise_pow
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_pow(x, y, axis=-1, act=None, name=None)
该OP逐元素对输入Tensor进行幂操作。

等式是：

.. math::
        Out = X ^ Y

- :math:`X` ：多维Tensor。
- :math:`Y` ：多维Tensor。

此运算算子有两种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 的 ``shape`` 匹配 :math:`X` 的 ``shape``，其中 ``axis`` 是 :math:`Y` 在 :math:`X` 上的起始维度的位置。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
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
        - **x** （Variable）- 多维Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **y** （Variable）- 多维Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **axis** （int32, 可选）- Y的维度对应到X维度上时的索引。默认值为 -1。
        - **act** （string, 可选）- 激活函数名称，作用于输出上。默认值为None。详细请参考 :ref:`api_guide_activations` ， 常见的激活函数有: ``relu`` ``tanh`` ``sigmoid`` 等。
        - **name** （string, 可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：    维度和数据类型与 ``x`` 相同的多维Tensor。

返回类型： 多维Tensor。

**代码示例**

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
    z = fluid.layers.elementwise_pow(x, y)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    z_value = exe.run(feed=gen_data(),
                        fetch_list=[z.name])

    print(z_value) #[2, 243, 16]








