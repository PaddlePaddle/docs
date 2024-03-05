.. _cn_api_paddle_nn_functional_class_center_sample:

class_center_sample
-------------------------------

.. py:function:: paddle.nn.functional.class_center_sample(label, num_classes, num_samples, group=None)


类别中心采样方法是提出于 PartialFC 论文，目的是从全量的类别中心采样一个子集类别中心参与训练。采样过程也非常简单直观：

1. 首先把所有正类别中心采样；
2. 然后随机采样负类别中心。

具体的过程是，给定一个维度为 [``batch_size``] 的 ``label``，从 [0, num_classes) 中把所有正类别中心选择出来，然后随机采样负类别中心补够 ``num_samples``。接着用采样出来的类别中心重新映射 ``label``。

更多的细节信息，请参考论文《Partial FC: Training 10 Million Identities on a Single Machine》，arxiv: https://arxiv.org/abs/2010.05222

.. note::
    如果正类别中心数量大于给定的 ``num_samples``，将保留所有的正类别中心，因此 ``sampled_class_center`` 的维度将是 [``num_positive_class_centers``]。

    该 API 支持 CPU ， 单 GPU 和多 GPU 。

    数据并行模式，设置 ``group=False`` 。
    模型并行模式，设置 ``group=None`` ，否则返回 ``paddle.distributed.new_group`` 的组实例。

参数
::::::::::::

    - **label** (Tensor) - 1-D Tensor，数据类型为 int32 或者 int64，每个元素的取值范围在 [0, num_classes)。
    - **num_classes** (int) - 一个正整数，表示当前卡的类别数，注意每张卡的 ``num_classes`` 可以是不同的值。
    - **num_samples** (int) - 一个正整数，表示当前卡采样的类别中心数量。
    - **group** (Group，可选) - 通信组的抽象描述，具体可以参考 ``paddle.distributed.collective.Group``。默认值为 ``None``。

返回
::::::::::::

    ``Tensor`` 二元组 - (``remapped_label``, ``sampled_class_center``)，``remapped_label`` 是重新映射后的标签，``sampled_class_center`` 是所采样的类别中心。


代码示例
::::::::::::
COPY-FROM: paddle.nn.functional.class_center_sample:code-example1
COPY-FROM: paddle.nn.functional.class_center_sample:code-example2
