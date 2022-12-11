.. _cn_api_fluid_layers_bipartite_match:

bipartite_match
-------------------------------

.. py:function:: paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)




该 OP 实现了贪心二分匹配算法，该算法用于根据输入距离矩阵获得与最大距离的匹配。对于输入二维矩阵，二分匹配算法可以找到每一行的匹配列（匹配意味着最大距离），也可以找到每列的匹配行。此算子仅计算列到行的匹配索引。对于每个实例，匹配索引的数量是
输入距离矩阵的列号。**该 OP 仅支持 CPU**

它有两个输出，匹配的索引和距离。简单的描述是该算法将最佳（最大距离）行实体与列实体匹配，并且匹配的索引在 ColToRowMatchIndices 的每一行中不重复。如果列实体与任何行实体不匹配，则 ColToRowMatchIndices 设置为-1。

注意：输入距离矩阵可以是 LoDTensor（带有 LoD）或 Tensor。如果 LoDTensor 带有 LoD，则 ColToRowMatchIndices 的高度是批量大小。如果是 Tensor，则 ColToRowMatchIndices 的高度为 1。

注意：此 API 是一个非常低级别的 API。它由 ``ssd_loss`` 层使用。请考虑使用 ``ssd_loss`` 。

参数
::::::::::::

                - **dist_matrix** （Variable）- 维度为：[K,M]的 2-D LoDTensor，数据类型为 float32 或 float64。它是由每行和每列来表示实体之间的成对距离矩阵。例如，假设一个实体是具有形状[K]的 A，另一个实体是具有形状[M]的 B. dist_matrix [i] [j]是 A[i]和 B[j]之间的距离。距离越大，匹配越好。注意：此 Tensor 可以包含 LoD 信息以表示一批输入。该批次的一个实例可以包含不同数量的实体。
                - **match_type** （str，可选）- 匹配方法的类型，应为'bipartite'或'per_prediction'。默认值为 None，即'bipartite'。
                - **dist_threshold** （float32，可选）- 如果 match_type 为'per_prediction'，则此阈值用于根据最大距离确定额外匹配的 bbox，默认值为 None，即 0.5。
                - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::


         - matched_indices（Variable）- 维度为[N，M]的 2-D Tensor，数据类型为 int32。 N 是批量大小。如果 match_indices[i][j]为-1，则表示 B[j]与第 i 个实例中的任何实体都不匹配。否则，这意味着在第 i 个实例中 B[j]与行 match_indices[i][j]匹配。第 i 个实>例的行号保存在 match_indices[i][j]中。
         - matched_distance（Variable）- 维度为[N，M]的 2-D Tensor，数据类型为 float32，。 N 是批量大小。如果 match_indices[i][j]为-1，则 match_distance[i][j]也为-1.0。否则，假设 match_distance[i][j]=d，并且每个实例的行偏移称为 LoD。然后 match_distance[i][j]=dist_matrix[d]+ LoD[i]][j]。


返回类型
::::::::::::
Tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.bipartite_match
