.. _cn_api_paddle_tensor_gather:

gather
-------------------------------

.. py:function:: paddle.gather(x, index, axis=None, name=None)

根据索引 index 获取输入 ``x`` 的指定 ``aixs`` 维度的条目，并将它们拼接在一起。

.. code-block:: text

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        axis = 0

        Then:

        Out = [[3, 4],
               [5, 6]]

**参数**:
        - **x** (Tensor) - 输入 Tensor, 秩 ``rank >= 1`` , 支持的数据类型包括 int32、int64、float32、float64 和 uint8 (CPU)、float16（GPU） 。
        - **index** (Tensor) - 索引 Tensor，秩 ``rank = 1``, 数据类型为 int32 或 int64。
        - **axis** (Tensor) - 指定index 获取输入的维度， ``axis`` 的类型可以是int或者Tensor，当 ``axis`` 为Tensor的时候其数据类型为int32 或者int64。
        - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

**返回**：和输入的秩相同的输出Tensor。


**代码示例**：

..  code-block:: python
            
    import numpy as np
    import paddle

    input_1 = np.array([[1,2],[3,4],[5,6]])
    index_1 = np.array([0,1])
    input = paddle.to_tensor(input_1)
    index = paddle.to_tensor(index_1)
    output = paddle.gather(input, index, axis=0)
    # expected output: [[1,2],[3,4]]

