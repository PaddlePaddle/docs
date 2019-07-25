.. _cn_api_fluid_layers_detection_map:

detection_map
-------------------------------

.. py:function:: paddle.fluid.layers.detection_map(detect_res, label, class_num, background_label=0, overlap_threshold=0.3, evaluate_difficult=True, has_state=None, input_states=None, out_states=None, ap_version='integral')

检测mAP评估算子。一般步骤如下：首先，根据检测输入和标签计算TP（true positive）和FP（false positive），然后计算mAP评估值。支持'11 point'和积分mAP算法。请从以下文章中获取更多信息：

        https://sanchom.wordpress.com/tag/average-precision/

        https://arxiv.org/abs/1512.02325

参数：
        - **detect_res** （LoDTensor）- 用具有形状[M，6]的2-D LoDTensor来表示检测。每行有6个值：[label，confidence，xmin，ymin，xmax，ymax]，M是此小批量中检测结果的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N+1，如果LoD[i+1]-LoD[i]== 0，则表示没有检测到数据。
        - **label** （LoDTensor）- 2-D LoDTensor用来带有标签的真实数据。每行有6个值：[label，xmin，ymin，xmax，ymax，is_difficult]或5个值：[label，xmin，ymin，xmax，ymax]，其中N是此小批量中真实数据的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N + 1，如果LoD [i + 1] - LoD [i] == 0，则表示没有真实数据。
        - **class_num** （int）- 类的数目。
        - **background_label** （int，defalut：0）- background标签的索引，background标签将被忽略。如果设置为-1，则将考虑所有类别。
        - **overlap_threshold** （float）- 检测输出和真实数据下限的重叠阈值。
        - **evaluate_difficult** （bool，默认为true）- 通过切换来控制是否对difficult-data进行评估。
        - **has_state** （Tensor <int>）- 是shape[1]的张量，0表示忽略输入状态，包括PosCount，TruePos，FalsePos。
        - **input_states** - 如果不是None，它包含3个元素：

            1、pos_count（Tensor）是一个shape为[Ncls，1]的张量，存储每类的输入正例的数量，Ncls是输入分类的数量。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。当输入（PosCount）为空时，不执行累积计算，仅计算当前小批量的结果。

            2、true_pos（LoDTensor）是一个shape为[Ntp，2]的2-D LoDTensor，存储每个类输入的正实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。

            3、false_pos（LoDTensor）是一个shape为[Nfp，2]的2-D LoDTensor，存储每个类输入的负实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。

        - **out_states** - 如果不是None，它包含3个元素：

            1、accum_pos_count（Tensor）是一个shape为[Ncls，1]的Tensor，存储每个类的实例数。它结合了输入（PosCount）和从输入中的（Detection）和（label）计算的正例数。

            2、accum_true_pos（LoDTensor）是一个shape为[Ntp'，2]的LoDTensor，存储每个类的正实例。它结合了输入（TruePos）和从输入中（Detection）和（label）计算的正实例数。 。

            3、accum_false_pos（LoDTensor）是一个shape为[Nfp'，2]的LoDTensor，存储每个类的负实例。它结合了输入（FalsePos）和从输入中（Detection）和（label）计算的负实例数。

        - **ap_version** （string，默认'integral'）- AP算法类型，'integral'或'11 point'。

返回：        具有形状[1]的（Tensor），存储mAP的检测评估结果。

**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        from fluid.layers import detection
        detect_res = fluid.layers.data(
            name='detect_res',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')
        label = fluid.layers.data(
            name='label',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')

        map_out = fluid.layers.detection_map(detect_res, label, 21)











