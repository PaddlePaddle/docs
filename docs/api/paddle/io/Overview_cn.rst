.. _cn_overview_io:

paddle.io
---------------------

paddle.io 目录下包含飞桨框架数据集定义、数据读取相关的 API。具体如下：

-  :ref:`多进程数据读取器相关 API <about_dataloader>`
-  :ref:`数据集定义相关 API <about_dataset_define>`
-  :ref:`数据集操作相关 API <about_dataset_operate>`
-  :ref:`采样器相关 API <about_sampler>`
-  :ref:`批采样器相关 API <about_batch_sampler>`



.. _about_dataloader:

多进程数据读取器相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`DataLoader <cn_api_paddle_io_DataLoader>` ", "多进程数据读取器"
    " :ref:`get_worker_info <cn_api_paddle_io_get_worker_info>` ", "获取当前子进程相关信息"

.. _about_dataset_define:

数据集定义相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Dataset <cn_api_paddle_io_Dataset>` ", "映射式(map-style)数据集基类定义接口"
    " :ref:`IterableDataset <cn_api_paddle_io_IterableDataset>` ", "迭代式(iterable-style)数据集基类定义接口"
    " :ref:`TensorDataset <cn_api_paddle_io_TensorDataset>` ", "Tensor 数据集基类定义接口"

.. _about_dataset_operate:

数据集操作相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`ChainDataset <cn_api_paddle_io_ChainDataset>` ", "数据集样本级联接口"
    " :ref:`ComposeDataset <cn_api_paddle_io_ComposeDataset>` ", "数据集字段组合接口"
    " :ref:`Subset <cn_api_paddle_io_Subset>` ", "数据集取子集接口"
    " :ref:`random_split <cn_api_paddle_io_random_split>` ", "给定子集合 dataset 的长度数组，随机切分出原数据集合的非重复子集合"

.. _about_sampler:

采样器相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Sampler <cn_api_paddle_io_Sampler>` ", "采样器基类定义接口"
    " :ref:`SequenceSampler <cn_api_paddle_io_SequenceSampler>` ", "顺序采样器接口"
    " :ref:`RandomSampler <cn_api_paddle_io_RandomSampler>` ", "随机采样器接口"
    " :ref:`WeightedRandomSampler <cn_api_paddle_io_WeightedRandomSampler>` ", "带权重随机采样器接口"
    " :ref:`SubesetRandomSampler <cn_api_paddle_io_SubsetRandomSampler>` ", "子集随机随机采样器接口"

.. _about_batch_sampler:

批采样器相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`BatchSampler <cn_api_paddle_io_BatchSampler>` ", "批采样器接口"
    " :ref:`DistributedBatchSampler <cn_api_paddle_io_DistributedBatchSampler>` ", "分布式批采样器接口, 用于分布式多卡场景"
