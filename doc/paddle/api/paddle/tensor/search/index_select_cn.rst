.. _cn_api_tensor_search_index_select:

index_select
-------------------------------

.. py:function:: paddle.index_select(x, index, axis=0, name=None)



该OP沿着指定轴 ``axis`` 对输入 ``x`` 进行索引，取 ``index`` 中指定的相应项，创建并返回到一个新的Tensor。这里 ``index`` 是一个 ``1-D`` Tensor。除 ``axis`` 轴外，返回的Tensor其余维度大小和输入 ``x`` 相等 ， ``axis`` 维度的大小等于 ``index`` 的大小。
        
**参数**：
    - **x** （Tensor）– 输入Tensor。 ``x`` 的数据类型可以是float32，float64，int32，int64。
    - **index** （Tensor）– 包含索引下标的1-D Tensor。
    - **axis**    (int, 可选) – 索引轴，若未指定，则默认选取第0维。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

**返回**：
    -**Tensor**: 返回一个数据类型同输入的Tensor。
     

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()  # Now we are in imperative mode
        data = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]])
        data_index = np.array([0, 1, 1]).astype('int32')

        x = paddle.to_tensor(data)
        index = paddle.to_tensor(data_index)
        out_z1 = paddle.index_select(x=x, index=index)
        #[[1. 2. 3. 4.]
        # [5. 6. 7. 8.]
        # [5. 6. 7. 8.]]
        out_z2 = paddle.index_select(x=x, index=index, axis=1)
        #[[ 1.  2.  2.]
        # [ 5.  6.  6.]
        # [ 9. 10. 10.]]

