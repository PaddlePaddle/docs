.. _cn_api_fluid_layers_ssd_loss:

ssd_loss
-------------------------------

.. py:function:: paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None)

:alias_main: paddle.nn.functional.ssd_loss
:alias: paddle.nn.functional.ssd_loss,paddle.nn.functional.loss.ssd_loss
:old_api: paddle.fluid.layers.ssd_loss



该OP用于SSD物体检测算法的多窗口损失层

该层用于计算SSD的损失，给定位置偏移预测，置信度预测，候选框和真实框标签，以及难样本挖掘的类型。通过执行以下步骤，返回的损失是本地化损失（或回归损失）和置信度损失（或分类损失）的加权和：

1、通过二分匹配算法查找匹配的边界框。

        1.1、计算真实框与先验框之间的IOU相似度。

        1.2、通过二分匹配算法计算匹配的边界框。

2、计算难分样本的置信度

        2.1、根据匹配的索引获取目标标签。

        2.2、计算置信度损失。

3、应用难样本挖掘算法来获取负样本索引并更新匹配的索引。

4、分配分类和回归目标

        4.1、根据生成的候选框bbox进行编码。

        4.2、分配回归目标。

        4.3、分配分类目标。

5、计算总体的物体损失。

        5.1计算置信度(confidence)损失。

        5.1计算回归(location)损失。

        5.3计算总体加权损失。

参数：
        - **location** （Variable）- 位置预测，具有形状[N，Np，4]的3D-Tensor，N是batch大小，Np是每个实例的预测总数。 4是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin代表box左上坐标，xmax，ymax代表box右下坐标，数据类型为float32或float64。
        - **confidence**  (Variable) - 置信度(分类)预测，具有形状[N，Np，C]的3D-Tensor，N是batch大小，Np是每个实例的预测总数，C是类别数量，数据类型为float32或float64。
        - **gt_box** （Variable）- 真实框(bbox),具有形状[Ng，4]的2D LoDTensor，Ng是mini-batch输入的真实框（bbox）的总数,4是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin代表box左上坐标，xmax，ymax代表box右下坐标，数据类型为float32或float64。
        - **gt_label** （Variable）- ground-truth标签, 具有形状[Ng，1]的2D LoDTensor,Ng是mini-batch输入的真实框（bbox）的总数,1表示类别号，数据类型为float32或float64。
        - **prior_box** （Variable）- 检测网络生成的候选框, 具有形状[Np，4]的2D-Tensor，Np是生成的候选框总数，4是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin代表box左上坐标，xmax，ymax代表box右下坐标，数据类型为float32或float64。。
        - **prior_box_var** （Variable）- 候选框的方差, 具有形状[Np，4]的2D张量，形状及数据类型同 ``prior_box`` 。
        - **background_label** （int）- background标签的索引，默认为0。
        - **overlap_threshold** （float）- 额外匹配的bbox阈值，当找到匹配的框，如果 ``match_type`` 为'per_prediction'，使用 ``overlap_threshold`` 确定额外匹配的bbox。默认为0.5。
        - **neg_pos_ratio** （float）- 负框与正框的比率，仅在 ``mining_type`` 为'max_negative'时使用，默认为3.0。
        - **neg_overlap** （float）- 不匹配预测的负重叠上限。仅当 ``mining_type`` 为'max_negative'时使用，默认为0.5。
        - **loc_loss_weight** （float）- 回归损失的权重，默认为1.0。
        - **conf_loss_weight** （float）- 置信度损失的权重，默认为1.0。
        - **match_type** （str）- 训练期间匹配方法的类型应为'bipartite'或'per_prediction'，默认为'per_prediction'。
        - **mining_type** （str）- 难样本挖掘类型，分为'hard_example'或'max_negative'，目前只支持'max_negative'。
        - **normalize** （bool）- 是否通过输出位置的总数将SSD损失标准化，默认为True。
        - **sample_size** （int）- 负样本框的最大样本大小，仅在 ``mining_type`` 为'hard_example'时使用。

返回：  Variable(Tensor)  定位损失和置信度损失的加权和, 具有形状[N * Np，1], N是batch大小，Np是每个实例的预测总数，数据类型为float32或float64。

抛出异常：        ``ValueError`` - 如果 ``mining_type`` 是'hard_example'，目前只支持 ``max_negative`` 的挖掘类型。

**代码示例**

..  code-block:: python

         import paddle.fluid as fluid
         pb = fluid.layers.data(
                           name='prior_box',
                           shape=[10, 4],
                           append_batch_size=False,
                           dtype='float32')
         pbv = fluid.layers.data(
                           name='prior_box_var',
                           shape=[10, 4],
                           append_batch_size=False,
                           dtype='float32')
         loc = fluid.layers.data(name='target_box', shape=[10, 4], dtype='float32')
         scores = fluid.layers.data(name='scores', shape=[10, 21], dtype='float32')
         gt_box = fluid.layers.data(
                 name='gt_box', shape=[4], lod_level=1, dtype='float32')
         gt_label = fluid.layers.data(
                 name='gt_label', shape=[1], lod_level=1, dtype='float32')
         loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)










