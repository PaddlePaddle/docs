.. _cn_api_fluid_layers_collect_fpn_proposals:

collect_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.collect_fpn_proposals(multi_rois, multi_scores, min_level, max_level, post_nms_top_n, name=None)




**该 op 仅支持 LoDTensor 输入**。连接多级 RoIs（感兴趣区域）并依据 multi_scores 选择 N 个 RoIs。此操作执行以下步骤：
1、选择 num_level 个 RoIs 和 scores 作为输入：num_level = max_level - min_level
2、连接 num_level 个 RoIs 和 scores。
3、对 scores 排序并选择 post_nms_top_n 个 scores。
4、通过 scores 中的选定位置收集 RoIs。
5、通过对应的 batch_id 重新对 RoIs 排序。


参数
::::::::::::

    - **multi_rois** (list) – 要收集的 RoIs 列表，列表中的元素为[N, 4]的 2-D LoDTensor，数据类型为 float32 或 float64，其中 N 为 RoI 的个数。
    - **multi_scores** (list) - 要收集的 RoIs 对应分数的列表，列表中的元素为[N, 1]的 2-D LoDTensor，数据类型为 float32 或 float64，其中 N 为 RoI 的个数。
    - **min_level** (int) - 要收集的 FPN 层的最低级
    - **max_level** (int) – 要收集的 FPN 层的最高级
    - **post_nms_top_n** (int) – 所选 RoIs 的数目
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
表示选定具有高分数的 RoIs 的 LoDTensor，数据类型为 float32 或 float64，同时具有 LoD 信息，维度为[M, 4]，其中 M 为 post_nms_top_n。


返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.collect_fpn_proposals
