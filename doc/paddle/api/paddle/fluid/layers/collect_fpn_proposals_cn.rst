.. _cn_api_fluid_layers_collect_fpn_proposals:

collect_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.collect_fpn_proposals(multi_rois, multi_scores, min_level, max_level, post_nms_top_n, name=None)

:alias_main: paddle.nn.functional.collect_fpn_proposals
:alias: paddle.nn.functional.collect_fpn_proposals,paddle.nn.functional.vision.collect_fpn_proposals
:old_api: paddle.fluid.layers.collect_fpn_proposals



**该op仅支持LoDTensor输入**。连接多级RoIs（感兴趣区域）并依据multi_scores选择N个RoIs。此操作执行以下步骤：
1、选择num_level个RoIs和scores作为输入：num_level = max_level - min_level
2、连接num_level个RoIs和scores。
3、对scores排序并选择post_nms_top_n个scores。
4、通过scores中的选定位置收集RoIs。
5、通过对应的batch_id重新对RoIs排序。


参数：
    - **multi_rois** (list) – 要收集的RoIs列表，列表中的元素为[N, 4]的2-D LoDTensor, 数据类型为float32或float64，其中N为RoI的个数。
    - **multi_scores** (list) - 要收集的RoIs对应分数的列表，列表中的元素为[N, 1]的2-D LoDTensor, 数据类型为float32或float64，其中N为RoI的个数。
    - **min_level** (int) - 要收集的FPN层的最低级
    - **max_level** (int) – 要收集的FPN层的最高级
    - **post_nms_top_n** (int) – 所选RoIs的数目
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。 

返回：表示选定具有高分数的RoIs的LoDTensor，数据类型为float32或float64，同时具有LoD信息，维度为[M, 4]，其中M为post_nms_top_n。


返回类型：Variable


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    multi_rois = []
    multi_scores = []
    for i in range(4):
        multi_rois.append(fluid.data(
            name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
    for i in range(4):
        multi_scores.append(fluid.data(
            name='score_'+str(i), shape=[None, 1], dtype='float32', lod_level=1))

    fpn_rois = fluid.layers.collect_fpn_proposals(
        multi_rois=multi_rois,
        multi_scores=multi_scores,
        min_level=2,
        max_level=5,
        post_nms_top_n=2000)
