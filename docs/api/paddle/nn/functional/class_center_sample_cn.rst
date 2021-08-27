.. _cn_api_paddle_nn_functional_class_center_sample:

class_center_sample
-------------------------------

.. py:function:: paddle.nn.functional.class_center_sample(label, num_classes, num_samples, group=None, seed=None)


类别中心采样方法是提出于 PartialFC 论文，目的是从全量的类别中心采样一个子集类别中心参与训练。采样过程也非常简单直观：

1. 首先把所有正类别中心采样；
2. 然后随机采样负类别中心。

具体的过程是，给定一个维度为 [``batch_size``] 的 ``label``，从 [0, num_classes) 中把所有正类别中心选择出来，然后随机采样负类别中心补够 ``num_samples``。接着用采样出来的类别中心重新映射 ``label``。

更多的细节信息，请参考论文《Partial FC: Training 10 Million Identities on a Single Machine》，arxiv: https://arxiv.org/abs/2010.05222

提示:
    如果正类别中心数量大于给定的 ``num_samples``，将保留所有的正类别中心，因此 ``sampled_class_center`` 的维度将是 [``num_positive_class_centers``]。


参数:
    - **label** (Tensor) - 1-D Tensor，数据类型为 int32 或者 int64，每个元素的取值范围在 [0, num_classes)。
    - **num_classes** (int) - 一个正整数，表示当前卡的类别数，注意每张卡的 ``num_classes`` 可以是不同的值。
    - **num_samples** (int) - 一个正整数，表示当前卡采样的类别中心数量。
    - **group** (Group, 可选) - 通信组的抽象描述，具体可以参考 ``paddle.distributed.collective.Group``。默认值为 ``None``。
    - **seed** （int, 可选）- 随机数种子。默认值为 `None`。

返回:
    ``Tensor`` 二元组 - (``remapped_label``, ``sampled_class_center``)，``remapped_label`` 是重新映射后的标签，``sampled_class_center`` 是所采样的类别中心。

抛出异常:
    - :code:`ValueError` - ``num_samples`` > ``num_classes`` 时抛出异常。

**代码示例**:

.. code-block:: python

    # CPU or single GPU
    import paddle
    num_classes = 20
    batch_size = 10
    num_samples = 6
    label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')
    remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes, num_samples)
    print(label)
    print(remapped_label)
    print(sampled_class_index)
    # the output is
    #Tensor(shape=[10], dtype=int64, place=CPUPlace, stop_gradient=True,
    #       [11, 5 , 1 , 3 , 12, 2 , 15, 19, 18, 19])
    #Tensor(shape=[10], dtype=int64, place=CPUPlace, stop_gradient=True,
    #       [4, 3, 0, 2, 5, 1, 6, 8, 7, 8])
    #Tensor(shape=[9], dtype=int64, place=CPUPlace, stop_gradient=True,
    #       [1 , 2 , 3 , 5 , 11, 12, 15, 18, 19])
    
.. code-block:: python

    # required: distributed
    # Multi GPU, test_class_center_sample.py
    import paddle
    import paddle.distributed as dist
    strategy = dist.fleet.DistributedStrategy()
    dist.fleet.init(is_collective=True, strategy=strategy)
    batch_size = 10
    num_samples = 6
    rank_id = dist.get_rank()
    # num_classes of each GPU can be different, e.g num_classes_list = [10, 8]
    num_classes_list = [10, 10]
    num_classes = paddle.sum(paddle.to_tensor(num_classes_list))
    label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
    label_list = []
    dist.all_gather(label_list, label)
    label = paddle.concat(label_list, axis=0)
    remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes_list[rank_id], num_samples)
    print(label)
    print(remapped_label)
    print(sampled_class_index)
    #python -m paddle.distributed.launch --gpus=0,1 test_class_center_sample.py
    # rank 0 output:
    #Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
    #Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
    #Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #       [0, 2, 4, 8, 9, 3])

    # rank 1 output:
    #Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
    #       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
    #Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
    #       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
    #Tensor(shape=[7], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
    #       [0, 1, 2, 3, 5, 7, 8])
