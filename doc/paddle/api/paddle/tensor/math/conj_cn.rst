.. _cn_api_tensor_conj:

conj
-------------------------------

.. py:function:: paddle.conj(x, name=None)


该OP是逐元素计算Tensor的共轭运算。

参数：
    - x (Tensor) - 输入的复数值的Tensor，数据类型为：complex64、complex128、float32、float64、int32 或int64。
    - name (str，可选） - 默认值为None。一般无需用户设置。更多信息请参见 :ref:`api_guide_Name`。

返回：
    - out (Tensor) - 输入的共轭。形状和数据类型与输入一致。如果tensor元素是实数类型，如float32、float64、int32、或者int64，返回值和输入相同。


**代码示例**

..  code-block:: python

          import paddle
          data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1+1j), (2+2j), (3+3j)],
          #        [(4+4j), (5+5j), (6+6j)]])

          conj_data=paddle.conj(data)
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1-1j), (2-2j), (3-3j)],
          #        [(4-4j), (5-5j), (6-6j)]])
