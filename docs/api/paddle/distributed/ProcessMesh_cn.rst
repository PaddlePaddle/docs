.. _cn_api_paddle_distributed_ProcessMesh:

ProcessMesh
-------------------------------

.. py:class:: paddle.distributed.ProcessMesh

ProcessMesh 对象描述了所使用进程的笛卡尔拓扑结构。


参数
:::::::::

    - **mesh** (list|numpy.array) - 表示一组设备的逻辑笛卡尔拓扑。笛卡尔网格的每个维度都被称为网格维度，以名称进行引用。同一 ProcessMesh 内的网格维度名称必须唯一。
    - **dim_names** (list，可选) - ProcessMesh 各个轴的名称。
    - **shape** (list|tuple，可选) - 定义 ProcessMesh 的形状。
    - **process_ids** (list|tuple，可选) - 进程的 id 集合。


代码示例
:::::::::

COPY-FROM: paddle.distributed.ProcessMesh
