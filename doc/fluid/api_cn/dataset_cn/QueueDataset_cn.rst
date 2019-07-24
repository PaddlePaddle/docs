.. _cn_api_fluid_dataset_QueueDataset:

QueueDataset
-------------------------------

.. py:class:: paddle.fluid.dataset.QueueDataset

流式处理数据。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")



.. py:method:: local_shuffle()

局域shuffle数据

QueueDataset中不支持局域shuffle，可能抛出NotImplementedError

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.local_shuffle()



.. py:method:: global_shuffle(fleet=None)

全局shuffle数据

QueueDataset中不支持全局shuffle，可能抛出NotImplementedError

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.global_shuffle(fleet)

