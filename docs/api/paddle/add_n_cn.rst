.. _cn_api_tensor_add_n:

add_n
-------------------------------

.. py:function:: paddle.add_n(inputs, name=None)




该OP用于对输入的一至多个Tensor或LoDTensor求和。如果输入的是LoDTensor，输出仅与第一个输入共享LoD信息（序列信息）。

例1：
::
    输入：
        input.shape = [2, 3]
        input = [[1, 2, 3],
              [4, 5, 6]]

    输出：
        output.shape = [2, 3]
        output = [[1, 2, 3],
                [4, 5, 6]]

例2：
::
    输入：
    第一个输入：
            input1.shape = [2, 3]
            input1 = [[1, 2, 3],
                  [4, 5, 6]]

    第二个输入：
            input2.shape = [2, 3]
            input2 = [[7, 8, 9],
                  [10, 11, 12]]

    输出：
        output.shape = [2, 3]
        output = [[8, 10, 12],
              [14, 16, 18]]

参数：
    - **inputs** (Tensor|list(Tensor)) - 输入的一至多个Tensor。如果输入了多个Tensor，则不同Tensor的shape和数据类型应保持一致。数据类型支持：float32，float64，int32，int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor, 输入 ``inputs`` 求和后的结果，shape和数据类型与 ``inputs`` 一致。


**代码示例：**

.. code-block:: python

    import paddle
    
    input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
    input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
    output = paddle.add_n([input0, input1])
    # [[8., 10., 12.], 
    #  [14., 16., 18.]]

