.. _cn_api_paddle_add_n:

add_n
-------------------------------

.. py:function:: paddle.add_n(inputs, name=None)




对输入的一至多个 Tensor 求和。

.. code-block:: text

    Case 1:
        输入：
            input.shape = [2, 3]
            input = [[1, 2, 3],
                  [4, 5, 6]]

        输出：
            output.shape = [2, 3]
            output = [[1, 2, 3],
                    [4, 5, 6]]

    Case 2:
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

参数
::::::::::::

    - **inputs** (Tensor|list(Tensor)) - 输入的一至多个 Tensor。如果输入了多个 Tensor，则不同 Tensor 的 shape 和数据类型应保持一致。数据类型支持：bfloat16、float16、float32、float64、int32、int64、complex64、complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输入 ``inputs`` 求和后的结果，shape 和数据类型与 ``inputs`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.add_n
