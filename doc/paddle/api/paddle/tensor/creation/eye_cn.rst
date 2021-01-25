.. _cn_api_paddle_tensor_eye:

eye
-------------------------------

.. py:function:: paddle.eye(num_rows, num_columns=None, dtype=None, name=None)

该OP用来构建二维Tensor(主对角线元素为1，其他元素为0)。

参数：
    - **num_rows** (int) - 生成2-D Tensor的行数，数据类型为非负int32。
    - **num_columns** (int，可选) - 生成2-D Tensor的列数，数据类型为非负int32。若为None，则默认等于num_rows。
    - **dtype** (np.dtype|str， 可选) - 返回Tensor的数据类型，可为float16，float32，float64， int32， int64。若为None, 则默认等于float32。
    - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： ``shape`` 为 [num_rows, num_columns]的Tensor。


抛出异常：
    - ``TypeError``: - 如果 ``dtype`` 的类型不是float16， float32， float64， int32， int64其中之一。
    - ``TypeError``: - 如果 ``num_columns`` 不是非负整数或者 ``num_rows`` 不是非负整数。

**代码示例**：

.. code-block:: python

    import paddle
    
    data = paddle.eye(3, dtype='int32')
    # [[1 0 0]
    #  [0 1 0]
    #  [0 0 1]]
    data = paddle.eye(2, 3, dtype='int32')
    # [[1 0 0]
    #  [0 1 0]]




