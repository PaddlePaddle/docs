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

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # 例 0
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # 例 1
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]


代码示例
::::::::::::

COPY-FROM: paddle.transpose
