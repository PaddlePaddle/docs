.. _cn_api_fluid_layers_collect_fpn_proposals:

collect_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.collect_fpn_proposals(multi_rois, multi_scores, min_level, max_level, post_nms_top_n, name=None)

连接多级RoIs（感兴趣区域）并依据multi_scores选择N个RoIs。此操作执行以下步骤：
1、选择num_level个RoIs和scores作为输入：num_level = max_level - min_level
2、连接num_level个RoIs和scores。
3、整理scores并选择post_nms_top_n个scores。
4、通过scores中的选定指数收集RoIs。
5、通过对应的batch_id重新整理RoIs。


参数：
    - **multi_ros** (list) – 要收集的RoIs列表
    - **multi_scores** (list) - 要收集的FPN层的最低级
    - **max_level** (int) – 要收集的FPN层的最高级
    - **post_nms_top_n** (int) – 所选RoIs的数目
    - **name** (str|None) – 该层的名称（可选项）

返回：选定RoIs的输出变量

返回类型：变量(Variable)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    multi_rois = []
    multi_scores = []
    for i in range(4):
        multi_rois.append(fluid.layers.data(
            name='roi_'+str(i), shape=[4], dtype='float32', lod_level=1))
    for i in range(4):
        multi_scores.append(fluid.layers.data(
            name='score_'+str(i), shape=[1], dtype='float32', lod_level=1))
     
    fpn_rois = fluid.layers.collect_fpn_proposals(
        multi_rois=multi_rois,
        multi_scores=multi_scores,
        min_level=2,
        max_level=5,
        post_nms_top_n=2000)




