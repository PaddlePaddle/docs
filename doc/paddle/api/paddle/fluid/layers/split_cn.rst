.. _cn_api_fluid_layers_split:

split
-------------------------------

.. py:function:: paddle.fluid.layers.split(input, num_or_sections, dim=-1, name=None)




该OP将输入Tensor分割成多个子Tensor。

参数：
    - **input** (Tensor) - 输入变量，数据类型为bool， float16，float32，float64，int32，int64的多维Tensor。
    - **num_or_sections** (int|list|tuple) - 如果 ``num_or_sections`` 是一个整数，则表示Tensor平均划分为相同大小子Tensor的数量。如果 ``num_or_sections`` 是一个list或tuple，那么它的长度代表子Tensor的数量，它的元素可以是整数或者形状为[1]的Tensor，依次代表子Tensor需要分割成的维度的大小。list或tuple的长度不能超过输入Tensor待分割的维度的大小。至多有一个元素值为-1，-1表示该值是由 ``input`` 待分割的维度值和 ``num_or_sections`` 的剩余元素推断出来的。
    - **dim** (int|Tenspr，可选) - 整数或者形状为[1]的Tensor，数据类型为int32或int64。表示需要分割的维度。如果 ``dim < 0`` ，则划分的维度为 ``rank(input) + dim`` 。默认值为-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：分割后的Tensor列表。


抛出异常：
    - :code:`TypeError`：``input`` 的数据类型不是bool、float16、float32、float64、int32或int64时 。
    - :code:`TypeError`：``num_or_sections`` 不是int、list 或 tuple时。
    - :code:`TypeError`：``dim`` 不是 int 或 Tensor时。当 ``dim`` 为Tensor，其数据类型不是int32或int64时。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    # input is a Tensor which shape is [3, 9, 5]
    input = fluid.data(
         name="input", shape=[3, 9, 5], dtype="float32")

    out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=1)
    # out0.shape [3, 3, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 3, 5]

    out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=1)
    # out0.shape [3, 2, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 4, 5]

    out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, -1], dim=1)
    # out0.shape [3, 2, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 4, 5]
    
    # dim is negative, the real dim is (rank(input) + axis) which real
    # value is 1.
    out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=-2)
    # out0.shape [3, 3, 5]
    # out1.shape [3, 3, 5]
    # out2.shape [3, 3, 5]








