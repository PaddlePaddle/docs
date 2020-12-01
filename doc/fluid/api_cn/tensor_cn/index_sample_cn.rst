.. _cn_api_tensor_search_index_sample:

index_sample
-------------------------------

.. py:function:: paddle.index_sample(x, index)

:alias_main: paddle.index_sample
:alias: paddle.index_sample,paddle.tensor.index_sample,paddle.tensor.search.index_sample



该OP实现对输入 ``x`` 中的元素进行批量抽样，取 ``index`` 指定的对应下标的元素，按index中出现的先后顺序组织，填充为一个新的张量。

该OP中 ``x`` 与 ``index`` 都是 ``2-D`` 张量。 ``index`` 的第一维度与输入 ``x`` 的第一维度必须相同， ``index`` 的第二维度没有大小要求，可以重复索引相同下标元素。
        
**参数**：
    - **x** （Variable）– 输入的二维张量，数据类型为 int32，int64，float32，float64。
    - **index** （Variable）– 包含索引下标的二维张量。数据类型为 int32，int64。

**返回**：
    -**Variable** ，数据类型与输入 ``x`` 相同，维度与 ``index`` 相同。
     
**代码示例**：

.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]]).astype('float32')

        data_index = np.array([[0, 1, 2],
                                [1, 2, 3],
                                [0, 0, 0]]).astype('int32')

        target_data = np.array([[100, 200, 300, 400],
                                [500, 600, 700, 800],
                                [900, 1000, 1100, 1200]]).astype('int32')


        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data)
            index = fluid.dygraph.to_variable(data_index)
            target = fluid.dygraph.to_variable(target_data)

            out_z1 = paddle.index_sample(x, index)
            print(out_z1)
            #[[1. 2. 3.]
            # [6. 7. 8.]
            # [9. 9. 9.]]

            # 巧妙用法：使用topk op产出的top元素的下标
            # 在另一个tensor中索引对应位置的元素
            top_value, top_index = fluid.layers.topk(x, k=2)
            out_z2 = paddle.index_sample(target, top_index)
            print(top_value)
            #[[ 4.  3.]
            # [ 8.  7.]
            # [12. 11.]]

            print(top_index)
            #[[3 2]
            # [3 2]
            # [3 2]]

            print(out_z2)
            #[[ 400  300]
            # [ 800  700]
            # [1200 1100]]


