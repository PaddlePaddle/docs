.. _cn_api_paddle_tensor_arange
stack
-------------------------------

.. py:function:: paddle.stack(x, axis=0, name=None)



该OP沿 axis 轴对输入 x 进行堆叠操作。要求所有输入Tensor有相同的Shape和数据类型。
例如，输入 x 为 N 个 Shape 为 [A, B]的 Tensor, 如果 ``axis==0`` , 则输出 Tensor 的 Shape 为 [N, A, B]; 如果 ``axis==1`` , 则输出 Tensor 的 Shape 为 [A, N, B]; 以此类推。

.. code-block:: text

    Case 1:

        Input:
        x[0].shape = [1, 2]
        x[0].data = [ [1.0 , 2.0 ] ]
        x[1].shape = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[2].shape = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]

        Attrs:
        axis = 0

        Output:
        Out.dims = [3, 1, 2]
        Out.data =[ [ [1.0, 2.0] ],
                    [ [3.0, 4.0] ],
                    [ [5.0, 6.0] ] ]


    Case 2:

        Input:
        x[0].shape = [1, 2]
        x[0].data = [ [1.0 , 2.0 ] ]
        x[1].shape = [1, 2]
        x[1].data = [ [3.0 , 4.0 ] ]
        x[2].shape = [1, 2]
        x[2].data = [ [5.0 , 6.0 ] ]


        Attrs:
        axis = 1 or axis = -2  # If axis = -2, axis = axis+ndim(x[0])+1 = -2+2+1 = 1.

        Output:
        Out.shape = [1, 3, 2]
        Out.data =[ [ [1.0, 2.0]
                        [3.0, 4.0]
                        [5.0, 6.0] ] ]

**参数**：
        - **x** (list[Tensor]|tuple[Tensor]) – 输入 x 是多个Tensor，且这些Tensor的维度和数据类型必须相同。支持的数据类型: float32，float64，int32，int64。

        - **axis** (int, 可选) – 指定对输入Tensor进行堆叠运算的轴，有效 axis 的范围是: [−(R+1),R+1]，R是输入中第一个Tensor的维数。如果 axis < 0，则 axis=axis+R+1 。默认值为0。

        - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**返回**：堆叠运算后的Tensor，数据类型与输入Tensor相同。

**代码示例**:

.. code-block:: python
   
    import paddle
    
    x1 = paddle.to_tensor([[1.0, 2.0]])
    x2 = paddle.to_tensor([[3.0, 4.0]])
    x3 = paddle.to_tensor([[5.0, 6.0]])
    out = paddle.stack([x1, x2, x3], axis=0)
    print(out.shape)  # [3, 1, 2]
    print(out)
    # [[[1., 2.]],
    #  [[3., 4.]],
    #  [[5., 6.]]]
