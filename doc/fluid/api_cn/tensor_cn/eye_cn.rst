.. _cn_api_paddle_tensor_eye:

eye
-------------------------------

.. py:function:: paddle.tensor.eye(num_rows, num_columns=None, out=None, dtype='float32', stop_gradient=True, name=None)

该OP用来构建单位矩阵。

参数：
    - **num_rows** (int) - 生成单位矩阵的行数，数据类型为非负int32。
    - **num_columns** (int) - 生成单位矩阵的列数，数据类型为非负int32。若为None，则默认等于num_rows。
    - **out**  (Variable， 可选) -  指定算子输出结果的LoDTensor/Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
    - **dtype** (string,  可选) - 返回张量的数据类型，可为int32，int64，float16，float32，float64。
    - **stop_gradient** (bool， 可选) - 是否对此OP停止计算梯度，默认值为False。
    - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：shape为 [num_rows, num_columns]的张量。

返回类型：Variable（Tensor|LoDTensor）数据类型为int32，int64，float16，float32，float64的Tensor或者LoDTensor。

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.eye(3, dtype='int32') # paddle.eye 等价于 paddle.tensor.eye
    # [[1, 0, 0]
    #  [0, 1, 0]
    #  [0, 0, 1]]
    data = paddle.eye(2, 3, dtype='int32')
    # [[1, 0, 0]
    #  [0, 1, 0]]





