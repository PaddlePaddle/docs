.. _cn_api_fluid_layers_beam_search:

beam_search
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search(pre_ids, pre_scores, ids, scores, beam_size, end_id, level=0, is_accumulated=True, name=None, return_parent_idx=False)




束搜索(Beam search)是在机器翻译等生成任务中选择候选词的一种经典算法

更多细节参考 `Beam Search <https://en.wikipedia.org/wiki/Beam_search>`_

**该 OP 仅支持 LoDTensor**，在计算产生得分之后使用，完成单个时间步内的束搜索。具体而言，在计算部分产生 ``ids`` 和 ``scores`` 后，对于每个源句（样本）该 OP 从 ``ids`` 中根据其对应的 ``scores`` 选择当前时间步 top-K （``K`` 是 ``beam_size``）的候选词 id。而 ``pre_id`` 和 ``pre_scores`` 是上一时间步 ``beam_search`` 的输出，加入输入用于特殊处理到达结束的翻译候选。

注意，如果 ``is_accumulated`` 为 True，传入的 ``scores`` 应该是累积分数。反之，``scores`` 是单步得分，会在该 OP 内被转化为 log 值并累积到 ``pre_scores`` 作为最终得分。如需使用长度惩罚，应在计算累积分数前使用其他 OP 完成。

束搜索的完整用法请参阅以下示例：

     fluid/tests/book/test_machine_translation.py



参数
::::::::::::

    - **pre_ids** （Variable） - LoD level 为 2 的 LodTensor，表示前一时间步选择的候选 id，是前一时间步 ``beam_search`` 的输出。第一步时，其形状应为为 :math:`[batch\_size，1]` ， lod 应为 :math:`[[0,1，...，batch\_size]，[0,1，...，batch\_size]]`。数据类型为 int64。
    - **pre_scores** （Variable） - 维度和 LoD 均与 ``pre_ids`` 相同的 LodTensor，表示前一时间步所选 id 对应的累积得分，是前一时间步 ``beam_search`` 的输出。数据类型为 float32。
    - **ids** （None|Variable） - 包含候选 id 的 LodTensor。LoD 应与 ``pre_ids`` 相同，形状为 :math:`[batch\_size \times beam\_size，K]`，其中第一维大小与 ``pre_ids`` 相同且``batch_size`` 会随样本到达结束而自动减小，``K`` 应该大于 ``beam_size``。数据类型为 int64。可为空，为空时使用 ``scores`` 上的索引作为 id。
    - **scores** （Variable） - 表示 ``ids`` 对应的累积分数的 LodTensor 变量，维度和 LoD 均与 ``ids`` 相同。
    - **beam_size** （int） - 指明束搜索中的束宽度。
    - **end_id** （int） - 指明标识序列结束的 id。
    - **level** （int，可选） - **可忽略，当前不能更改**。知道 LoD level 为 2 即可，两层 lod 的意义如下：第一级表示每个源句（样本）包含的 beam 大小，若满足结束条件（达到 ``beam_size`` 个结束）则变为 0；第二级是表示每个 beam 被选择的次数。
    - **is_accumulated** （bool，可选） - 指明输入分数 ``scores`` 是否为累积分数，默认为 True。
    - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **return_parent_idx** （bool，可选） - 指明是否返回一个额外的 Tensor，该 Tensor 保存了选择的 id 的父节点（beam）在 ``pre_id`` 中索引，可用于通过 gather OP 更新其他 Tensor 的内容。默认为 False。


返回
::::::::::::
Variable 的二元组或三元组。二元组中包含了当前时间步选择的 id 和对应的累积得分两个 LodTensor，形状相同且均为 :math:`[batch\_size×beam\_size，1]` ，LoD 相同且 level 均为 2，数据类型分别为 int64 和 float32；若 ``return_parent_idx`` 为 True 时为三元组，多返回一个保存了父节点在 ``pre_id`` 中索引的 Tensor，形状为 :math:`[batch\_size \times beam\_size]`，数据类型为 int64。

返回类型
::::::::::::
tuple

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.beam_search
