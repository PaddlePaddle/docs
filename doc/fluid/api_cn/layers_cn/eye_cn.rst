.. _cn_api_fluid_layers_eye:

eye
-------------------------------

.. py:function:: paddle.fluid.layers.eye(num_rows, num_columns=None, batch_shape=None, dtype='float32')

:alias_main: paddle.eye
:alias: paddle.eye,paddle.tensor.eye,paddle.tensor.creation.eye
:update_api: paddle.fluid.layers.eye



该OP用来构建二维维张量，或一个批次的二维张量。

参数：
    - **num_rows** (int) - 每一个批次二维张量的行数，数据类型为非负int32。
    - **num_columns** (int, 可选) - 每一个批次二维张量的列数，数据类型为非负int32。若为None，则默认等于num_rows。
    - **batch_shape** (list(int), 可选) - 如若提供，则返回向量的主批次维度将为batch_shape。
    - **dtype** (string， 可选) - 返回张量的数据类型，可为int32，int64，float16，float32，float64，默认数据类型为float32。
    
返回：shape为batch_shape + [num_rows, num_columns]的张量。

返回类型：Variable（Tensor|LoDTensor）数据类型为int32，int64，float16，float32，float64的Tensor或者LoDTensor。

抛出异常：
    - ``TypeError``: - 如果 ``dtype`` 的类型不是float16， float32， float64， int32， int64其中之一。
    - ``TypeError``: - 如果 ``num_columns`` 不是非负整数。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.eye(3, dtype='int32')
    # [[1, 0, 0]
    #  [0, 1, 0]
    #  [0, 0, 1]]

    data = fluid.layers.eye(2, 3, dtype='int32')
    # [[1, 0, 0]
    #  [0, 1, 0]]

    data = fluid.layers.eye(2, batch_shape=[3])
    # Construct a batch of 3 identity tensors, each 2 x 2.
    # data[i, :, :] is a 2 x 2 identity tensor, i = 0, 1, 2.






