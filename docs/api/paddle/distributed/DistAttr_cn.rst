.. _cn_api_distributed_DistAttr:

DistAttr
-------------------------------

.. py:class:: paddle.distributed.DistAttr(mesh, sharding_specs)

DistAttr 指定 Tensor 在 ProcessMesh 上的分布或切片方式。

参数
::::::::::::

    - **mesh** (paddle.distributed.ProcessMesh) - 表示进程拓扑信息的 ProcessMesh 对象。
    - **sharding_specs** (list[str|None]) - 描述 Tensor 的切分规则。

**代码示例**

COPY-FROM: paddle.distributed.DistAttr
