.. _cn_api_fluid_layers_similarity_focus:

similarity_focus
-------------------------------

.. py:function:: paddle.fluid.layers.similarity_focus(input, axis, indexes, name=None)




**实现 SimilarityFocus(相似度聚焦)运算**

通过以下三个步骤，该层生成一个和输入 ``input`` 同形的 similarity focus mask（相似度聚焦掩码）：

1. 根据 ``axis`` 和 ``indexes`` 提取一个三维 Tensor，第一维为 batch 大小。
   例如，如果 ``axis=1, indexes=[a]``，将得到矩阵 T=X[:, a, :, :] 。
   该例中，如果输入 X 的形为 (BatchSize, A, B, C)，则输出 TensorT 的形为 (BatchSize, B, C) 。
2. 对于每一个索引，在输出 T 中找到最大值。所以同一行、同一列最多只有一个数字，这意味着如果在第 i 行，第 j 列中找到最大值，那么在相应行、列中的其他数值都将被忽略。然后再在剩余的数值中找到下一个最大值。显然，将会产生 min（B,C）个数字，并把三维相似聚焦掩码 Tensor 相应位置的元素置为 1，其余则置为 0。对每个索引按元素进行 or 运算。
3. 将这个三维相似度聚焦掩码 Tensor broadcast 成输入 ``input`` 的形状

请参考 `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_ 。

::

    例如：

    给定四维 Tensor x 形为 (BatchSize, C, A, B)，其中 C 为通道 Channel 数目，
    特征图（feature map）的形为（A,B）：

        x.shape = (2, 3, 2, 2)
        x.data = [[[[0.8, 0.1],
                    [0.4, 0.5]],

                   [[0.9, 0.7],
                    [0.9, 0.9]],

                   [[0.8, 0.9],
                    [0.1, 0.2]]],


                  [[[0.2, 0.5],
                    [0.3, 0.4]],

                   [[0.9, 0.7],
                    [0.8, 0.4]],

                   [[0.0, 0.2],
                    [0.4, 0.7]]]]

    给定轴：1 (即 channel 轴)
    给定索引：[0]

    于是我们得到一个与输入同形的四维输出 Tensor：
        out.shape = (2, 3, 2, 2)
        out.data = [[[[1.0, 0.0],
                      [0.0, 1.0]],

                     [[1.0, 0.0],
                      [0.0, 1.0]],

                     [[1.0, 0.0],
                      [0.0, 1.0]]],

                    [[[0.0, 1.0],
                      [1.0, 0.0]],

                     [[0.0, 1.0],
                      [1.0, 0.0]],

                     [[0.0, 1.0],
                      [1.0, 0.0]]]]



参数
::::::::::::

  - **input** (Variable) – 输入 Tensor，应为一个四维 Tensor，形为[BatchSize, A, B, C]，数据类型为 float32 或者 float64。
  - **axis** (int) – 指明要选择的轴。可能取值为 1, 2 或 3。
  - **indexes** (list) – 指明选择维度的索引列表。

返回
::::::::::::
一个和输入 Variable 同形状、同数据类型的 Variable

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.similarity_focus
