.. _cn_overview_io:

paddle.io
---------------------

paddle.io 目录下包含飞桨框架数据集定义、数据读取相关的API。具体如下：

-  :ref:`多进程数据读取器相关API <about_dataloader>`
-  :ref:`数据集定义相关API <about_dataset_define>`
-  :ref:`数据集操作相关API <about_dataset_operate>`
-  :ref:`采样器相关API <about_sampler>`
-  :ref:`批采样器相关API <about_batch_sampler>`



.. _about_dataloader:

多进程数据读取器相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`DataLoader <cn_api_fluid_io_DataLoader>` ", "多进程数据读取器"
    " :ref:`get_worker_info <cn_api_io_cn_get_worker_info>` ", "获取当前子进程相关信息"
    " :ref:`default_collate_fn <cn_api_io_cn_default_collate_fn>` ", "多进程DataLoader中默认组batch函数"
    " :ref:`default_convert_fn <cn_api_io_cn_default_convert_fn>` ", "多进程DataLoader中默认转换函数，在多进程DataLoader中不组batch时使用，只将数据转换为Tensor而不组batch"
    
.. _about_dataset_define:

数据集定义相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Dataset <cn_api_io_cn_Dataset>` ", "映射式(map-style)数据集基类定义接口"
    " :ref:`IterableDataset <cn_api_io_cn_IterableDataset>` ", "迭代式(iterable-style)数据集基类定义接口"
    " :ref:`TensorDataset <cn_api_io_cn_TensorDataset>` ", "张量(Tensor)数据集基类定义接口"
    
.. _about_dataset_operate:

数据集操作相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`ChainDataset <cn_api_io_ChainDataset>` ", "数据集样本级联接口"
    " :ref:`ComposeDataset <cn_api_io_ComposeDataset>` ", "数据集字段组合接口"
    " :ref:`Subset <cn_api_io_Subset>` ", "数据集取子集接口"
    
.. _about_sampler:

采样器相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Sampler <cn_api_io_cn_Sampler>` ", "采样器基类定义接口"
    " :ref:`SequenceSampler <cn_api_io_cn_SequenceSampler>` ", "顺序采样器接口"
    " :ref:`RandomSampler <cn_api_io_cn_RandomSampler>` ", "随机采样器接口"
    " :ref:`WeightedRandomSampler <cn_api_io_cn_WeightedRandomSampler>` ", "带权重随机采样器接口"
    
.. _about_batch_sampler:

批采样器相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`BatchSampler <cn_api_io_cn_BatchSampler>` ", "批采样器接口"
    " :ref:`DistributedBatchSampler <cn_api_io_cn_DistributedBatchSampler>` ", "分布式批采样器接口, 用于分布式多卡场景"
    
