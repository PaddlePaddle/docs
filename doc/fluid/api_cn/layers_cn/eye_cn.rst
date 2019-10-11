.. _cn_api_fluid_layers_eye:

eye
-------------------------------

.. py:function:: paddle.fluid.layers.eye(num_rows, num_columns=None, batch_shape=None, dtype='float32')

该OP用来构建单位矩阵，或一个批次的单位矩阵。

参数：
    - **num_rows** (int) - 每一个批矩阵的行数，数据类型为非负int32。
    - **num_columns** (int) - 每一个批矩阵的列数，数据类型为非负int32。若为None，则默认等于num_rows。
    - **batch_shape** (list(int)) - 如若提供，则返回向量的主批次维度将为batch_shape。
    - **dtype** (string) - 返回张量的数据类型，可为int32，int64，float16，float32，float64。
    
返回：shape为batch_shape + [num_rows, num_columns]的张量。

返回类型：Variable（Tensor|LoDTensor）数据类型为int32，int64，float16，float32，float64的Tensor或者LoDTensor。

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






