.. _cn_api_paddle_transpose:

transpose
-------------------------------

.. py:function:: paddle.transpose(x,perm,name=None)




根据 perm 对输入的多维 Tensor 进行数据重排。返回多维 Tensor 的第 i 维对应输入 Tensor 的 perm[i]维。

参数
::::::::::::

    - **x** (Tensor) - 输入：x:[N_1, N_2, ..., N_k, D]多维 Tensor，可选的数据类型为 bool, float16, float32, float64, int32, int64。
    - **perm** (list|tuple) - perm 长度必须和 X 的维度相同，并依照 perm 中数据进行重排。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
多维 Tensor


代码示例
::::::::::::

.. code-block:: text

        # 这个代码块中如下的代码都是伪代码，旨在展示函数的执行逻辑与结果

        x = to_tensor([[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
                       [[13 14 15 16] [17 18 19 20] [21 22 23 24]]])
        shape(x): return [2,3,4]

        # 例一
        perm0 = [1,0,2]
        y_perm0 = transpose(x, perm0) # 将 x 按 perm0 重排

        # y_perm0 的第 0 维 是 x 的第 1 维
        # y_perm0 的第 1 维 是 x 的第 0 维
        # y_perm0 的第 2 维 是 x 的第 2 维
        # 上面两行也可以理解为交换了 x 的第 0 维和第 1 维

        y_perm0.data = [[[ 1  2  3  4]  [13 14 15 16]]
                        [[ 5  6  7  8]  [17 18 19 20]]
                        [[ 9 10 11 12]  [21 22 23 24]]]
        shape(y_perm0): return [3,2,4]

        # 例二
        perm1 = [2,1,0]
        y_perm1 = transpose(x, perm1) # Permute x by perm1

        # y_perm1 的第 0 维 是 x 的第 2 维
        # y_perm1 的第 1 维 是 x 的第 1 维
        # y_perm1 的第 2 维 是 x 的第 0 维
        # 上面两行也可以理解为交换了 x 的第 0 维和第 2 维

        y_perm1.data = [[[ 1 13]  [ 5 17]  [ 9 21]]
                        [[ 2 14]  [ 6 18]  [10 22]]
                        [[ 3 15]  [ 7 19]  [11 23]]
                        [[ 4 16]  [ 8 20]  [12 24]]]
        shape(y_perm1): return [4,3,2]

代码示例
::::::::::::

COPY-FROM: paddle.transpose
