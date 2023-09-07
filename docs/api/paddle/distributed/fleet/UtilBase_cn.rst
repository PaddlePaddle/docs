.. _cn_api_paddle_distributed_fleet_UtilBase:

UtilBase
-------------------------------

.. py:class:: paddle.distributed.fleet.UtilBase
分布式训练工具类，主要提供集合通信、文件系统操作等接口。

方法
::::::::::::
all_reduce(input, mode="sum", comm_world="worker")
'''''''''
在指定的通信集合间进行归约操作，并将归约结果返回给集合中每个实例。

**参数**

    - **input** (list|numpy.array) – 归约操作的输入。
    - **mode** (str) - 归约操作的模式，包含求和，取最大值和取最小值，默认为求和归约。
    - **comm_world** (str) - 归约操作的通信集合，包含：server 集合(``server``)，worker 集合(``worker``)及所有节点集合(``all``)，默认为 worker 集合。

**返回**

Numpy.array|None：一个和``input``形状一致的 numpy 数组或 None。

**代码示例**

COPY-FROM: paddle.distributed.fleet.UtilBase.all_reduce

barrier(comm_world="worker")
'''''''''
在指定的通信集合间进行阻塞操作，以实现集合间进度同步。

**参数**

   - **comm_world** (str) - 阻塞操作的通信集合，包含：server 集合(``server``)，worker 集合(``worker``)及所有节点集合(``all``)，默认为 worker 集合。

**代码示例**

COPY-FROM: paddle.distributed.fleet.UtilBase.barrier

all_gather(input, comm_world="worker")
'''''''''
在指定的通信集合间进行聚合操作，并将聚合的结果返回给集合中每个实例。

**参数**

   - **input** (int|float) - 聚合操作的输入。
   - **comm_world** (str) - 聚合操作的通信集合，包含：server 集合(``server``)，worker 集合(``worker``)及所有节点集合(``all``)，默认为 worker 集合。

**返回**

   - **output** (List): List 格式的聚合结果。

**代码示例**

COPY-FROM: paddle.distributed.fleet.UtilBase.all_gather

get_file_shard(files)
'''''''''
在数据并行的分布式训练中，获取属于当前训练节点的文件列表。

.. code-block:: text

    示例 1：原始所有文件列表 `files` = [a, b, c ,d, e]，训练节点个数 `trainer_num` = 2，那么属于零号节点的训练文件为[a, b, c]，属于 1 号节点的训练文件为[d, e]。
    示例 2：原始所有文件列表 `files` = [a, b]，训练节点个数 `trainer_num` = 3，那么属于零号节点的训练文件为[a]，属于 1 号节点的训练文件为[b]，属于 2 号节点的训练文件为[]。

**参数**

    - **files** (List)：原始所有文件列表。

**返回**

    - List：属于当前训练节点的文件列表。

**代码示例**

COPY-FROM: paddle.distributed.fleet.UtilBase.get_file_shard

print_on_rank(message, rank_id)
'''''''''

在编号为 `rank_id` 的节点上打印指定信息。

**参数**

    - **message** (str) – 打印内容。
    - **rank_id** (int) - 节点编号。

**代码示例**

COPY-FROM: paddle.distributed.fleet.UtilBase.print_on_rank
