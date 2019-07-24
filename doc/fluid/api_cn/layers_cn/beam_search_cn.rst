.. _cn_api_fluid_layers_beam_search:

beam_search
-------------------------------

.. py:function:: paddle.fluid.layers.beam_search(pre_ids, pre_scores, ids, scores, beam_size, end_id, level=0, is_accumulated=True, name=None, return_parent_idx=False)

在机器翻译任务中，束搜索(Beam search)是选择候选词的一种经典算法

更多细节参考 `Beam Search <https://en.wikipedia.org/wiki/Beam_search>`_

该层在一时间步中按束进行搜索。具体而言，根据候选词使用于源句子所得的 ``scores`` , 从候选词 ``ids`` 中选择当前步骤的 top-K （最佳K）候选词的id，其中 ``K`` 是 ``beam_size`` ， ``ids`` ， ``scores`` 是计算单元的预测结果。如果没有提供 ``ids`` ，则将会根据 ``scores`` 计算得出。 另外， ``pre_id`` 和 ``pre_scores`` 是上一步中 ``beam_search`` 的输出，用于特殊处理翻译的结束边界。

注意，如果 ``is_accumulated`` 为 True，传入的 ``scores`` 应该是累积分数。反之，``scores`` 会被认为为直接得分(straightforward scores)， 并且会被转化为log值并且在此运算中会被累积到 ``pre_scores`` 中。在计算累积分数之前应该使用额外的 operators 进行长度惩罚。

有关束搜索用法演示，请参阅以下示例：

     fluid/tests/book/test_machine_translation.py



参数:
  - **pre_ids** （Variable） -  LodTensor变量，它是上一步 ``beam_search`` 的输出。在第一步中。它应该是LodTensor，shape为 :math:`(batch\_size，1)` ， :math:`lod [[0,1，...，batch\_size]，[0,1，...，batch\_size]]`
  - **pre_scores** （Variable） -  LodTensor变量，它是上一步中beam_search的输出
  - **ids** （Variable） - 包含候选ID的LodTensor变量。shape为 :math:`（batch\_size×beam\_ize，K）` ，其中 ``K`` 应该是 ``beam_size``
  - **scores** （Variable） - 与 ``ids`` 及其shape对应的累积分数的LodTensor变量, 与 ``ids`` 的shape相同。
  - **beam_size** （int） - 束搜索中的束宽度。
  - **end_id** （int） - 结束标记的id。
  - **level** （int，default 0） - **可忽略，当前不能更改** 。它表示lod的源级别，解释如下。 ``ids`` 的 lod 级别应为2.第一级是源级别， 描述每个源句子（beam）的前缀（分支）的数量，第二级是描述这些候选者属于前缀的句子级别的方式。链接前缀和所选候选者的路径信息保存在lod中。
  - **is_accumulated** （bool，默认为True） - 输入分数是否为累计分数。
  - **name** （str | None） - 该层的名称（可选）。如果设置为None，则自动命名该层。
  - **return_parent_idx** （bool） - 是否返回一个额外的Tensor变量，在输出的pre_ids中保留selected_ids的双亲indice，可用于在下一个时间步收集单元状态。


返回：LodTensor元组。包含所选的id和与其相应的分数。 如果return_parent_idx为True，则包含一个保留selected_ids的双亲indice的额外Tensor变量。

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    # 假设 `probs` 包含计算神经元所得的预测结果
    # `pre_ids` 和 `pre_scores` 为beam_search之前时间步的输出
    beam_size = 4
    end_id = 1
    pre_ids = fluid.layers.data(
        name='pre_id', shape=[1], lod_level=2, dtype='int64')
    pre_scores = fluid.layers.data(
        name='pre_scores', shape=[1], lod_level=2, dtype='float32')
    probs = fluid.layers.data(
        name='probs', shape=[10000], dtype='float32')
    topk_scores, topk_indices = fluid.layers.topk(probs, k=beam_size)
    accu_scores = fluid.layers.elementwise_add(
                                          x=fluid.layers.log(x=topk_scores)),
                                          y=fluid.layers.reshape(
                                              pre_scores, shape=[-1]),
                                          axis=0)
    selected_ids, selected_scores = fluid.layers.beam_search(
                                          pre_ids=pre_ids,
                                          pre_scores=pre_scores,
                                          ids=topk_indices,
                                          scores=accu_scores,
                                          beam_size=beam_size,
                                          end_id=end_id)











