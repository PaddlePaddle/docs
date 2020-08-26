.. _cn_api_fluid_layers_transpose:

transpose
-------------------------------

.. py:function:: paddle.fluid.layers.transpose(x,perm,name=None)

:alias_main: paddle.transpose
:alias: paddle.transpose,paddle.tensor.transpose,paddle.tensor.linalg.transpose,paddle.tensor.manipulation.transpose
:old_api: paddle.fluid.layers.transpose



该OP根据perm对输入的多维Tensor进行数据重排。返回多维Tensor的第i维对应输入Tensor的perm[i]维。

参数：
    - **x** (Variable) - 输入：x:[N_1, N_2, ..., N_k, D]多维Tensor，可选的数据类型为float16, float32, float64, int32, int64。
    - **perm** (list) - perm长度必须和X的维度相同，并依照perm中数据进行重排。
    - **name** (str) - 该层名称（可选）。

返回： 多维Tensor

返回类型：Variable

**示例**:

.. code-block:: python

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]        
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # 例0
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # 例1
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]


**代码示例**:

.. code-block:: python

    # 请使用 append_batch_size=False 来避免
    # 在数据张量中添加多余的batch大小维度
    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2, 3, 4],
                    dtype='float32', append_batch_size=False)
    x_transposed = fluid.layers.transpose(x, perm=[1, 0, 2])
    print(x_transposed.shape)
    #(3L, 2L, 4L)



