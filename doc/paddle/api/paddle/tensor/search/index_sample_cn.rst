.. _cn_api_tensor_search_index_sample:

index_sample
-------------------------------

.. py:function:: paddle.index_sample(x, index)




该OP实现对输入 ``x`` 中的元素进行批量抽样，取 ``index`` 指定的对应下标的元素，按index中出现的先后顺序组织，填充为一个新的张量。

该OP中 ``x`` 与 ``index`` 都是 ``2-D`` 张量。 ``index`` 的第一维度与输入 ``x`` 的第一维度必须相同， ``index`` 的第二维度没有大小要求，可以重复索引相同下标元素。
        
**参数**：
    - **x** （Tensor）– 输入的二维张量，数据类型为 int32，int64，float32，float64。
    - **index** （Tensor）– 包含索引下标的二维张量。数据类型为 int32，int64。

**返回**：
    -**Tensor** ，数据类型与输入 ``x`` 相同，维度与 ``index`` 相同。
     
**代码示例**：

.. code-block:: python
        
        import paddle

        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]], dtype='float32')
        index = paddle.to_tensor([[0, 1, 2],
                                [1, 2, 3],
                                [0, 0, 0]], dtype='int32')
        target = paddle.to_tensor([[100, 200, 300, 400],
                                [500, 600, 700, 800],
                                [900, 1000, 1100, 1200]], dtype='int32')
        out_z1 = paddle.index_sample(x, index)
        print(out_z1.numpy())
        #[[1. 2. 3.]
        # [6. 7. 8.]
        # [9. 9. 9.]]

        # 巧妙用法：使用topk op产出的top元素的下标
        # 在另一个tensor中索引对应位置的元素
        top_value, top_index = paddle.topk(x, k=2)
        out_z2 = paddle.index_sample(target, top_index)
        print(top_value.numpy())
        #[[ 4.  3.]
        # [ 8.  7.]
        # [12. 11.]]

        print(top_index.numpy())
        #[[3 2]
        # [3 2]
        # [3 2]]

        print(out_z2.numpy())
        #[[ 400  300]
        # [ 800  700]
        # [1200 1100]]


