.. _cn_api_tensor_cn_cumprod:

cumprod
-------------------------------

.. py:function:: paddle.cumprod(x, dim=None, dtype=None, name=None)



沿给定维度 ``dim`` 计算输入tensor ``x`` 的累乘。

**注意**：结果的第一个元素和输入的第一个元素相同。

参数
:::::::::
    - x (Tensor) - 累乘的输入，需要进行累乘操作的tensor.
    - dim (int) - 指明需要累乘的维度，取值范围需在[-x.rank,x.rank)之间，其中x.rank表示输入tensor x的维度，-1代表最后一维。
    - dtype (str，可选) - 输出tensor的数据类型，支持int32、int64、float32、float64、complex64、complex128. 如果指定了，那么在执行操作之前，输入的tensor将被转换为dtype类型. 这对于防止数据类型溢出非常有用。默认为：None.
    - name （str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor``, 累乘操作的结果。

代码示例
::::::::::

.. code-block:: python

        import paddle

        data = paddle.arange(12)
        data = paddle.reshape(data, (3, 4))
        # [[ 0  1  2  3 ]
        #  [ 4  5  6  7 ]
        #  [ 8  9  10 11]]

        y = paddle.cumprod(data, dim=0)
        # [[ 0  1   2   3]
        #  [ 0  5  12  21]
        #  [ 0 45 120 231]]

        y = paddle.cumprod(data, dim=-1)
        # [[ 0   0   0    0]
        #  [ 4  20 120  840]
        #  [ 8  72 720 7920]]
    
        y = paddle.cumprod(data, dim=1, dtype='float64')
        # [[ 0.   0.   0.    0.]
        #  [ 4.  20. 120.  840.]
        #  [ 8.  72. 720. 7920.]]

        print(y.dtype)
        # paddle.float64

