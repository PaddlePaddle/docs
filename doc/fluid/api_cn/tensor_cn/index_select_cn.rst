.. _cn_api_tensor_search_index_select:

index_select
-------------------------------

.. py:function:: paddle.index_select(input, index, dim=0)

该OP沿着指定维度 ``dim`` 对输入 ``input`` 进行索引，取 ``index`` 中指定的相应项，然后返回到一个新的张量。这里 ``index`` 是一个 ``1-D`` 张量。除 ``dim`` 维外，返回的张量其余维度大小同输入 ``input`` ， ``dim`` 维大小等于 ``index`` 的大小。
        
**参数**：
    - **input** （Variable）– 输入张量。
    - **index** （Variable）– 包含索引下标的一维张量。
    - **dim**    (int, optional) – 索引轴，若未指定，则默认选取第一维。

**返回**：
    -**Variable** ，数据类型同输入。
     
**代码示例**：

.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]])
        data_index = np.array([0, 1, 1]).astype('int32')

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data)
            index = fluid.dygraph.to_variable(data_index)
            out_z1 = paddle.index_select(x, index)
            print(out_z1.numpy())
            #[[1. 2. 3. 4.]
            # [5. 6. 7. 8.]
            # [5. 6. 7. 8.]]
            out_z2 = paddle.index_select(x, index, dim=1)
            print(out_z2.numpy())
            #[[ 1.  2.  2.]
            # [ 5.  6.  6.]
            # [ 9. 10. 10.]]


