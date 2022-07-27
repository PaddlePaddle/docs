.. _cn_api_distributed_fleet_UtilBase:

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
    - **comm_world** (str) - 归约操作的通信集合，包含：server 集合(“server")，worker 集合("worker")及所有节点集合("all")，默认为 worker 集合。

**返回**

Numpy.array|None：一个和 `input` 形状一致的 numpy 数组或 None。

**代码示例**

.. code-block:: python

    # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .
    import paddle.distributed.fleet as fleet
    from paddle.distributed.fleet import PaddleCloudRoleMaker
    import sys
    import numpy as np
    import os

    os.environ["PADDLE_WITH_GLOO"] = "2"

    def train():
        role = PaddleCloudRoleMaker(
            is_collective=False,
            init_gloo=True,
            path="./tmp_gloo")
        fleet.init(role)

        if fleet.is_server():
            input = [1, 2]
            output = fleet.util.all_reduce(input, "sum", "server")
            print(output)
            # [2, 4]
        elif fleet.is_worker():
            input = np.array([3, 4])
            output = fleet.util.all_reduce(input, "sum", "worker")
            print(output)
            # [6, 8]
        output = fleet.util.all_reduce(input, "sum", "all")
        print(output)
        # [8, 12]
    if __name__ == "__main__":
        train()

barrier(comm_world="worker")
'''''''''
在指定的通信集合间进行阻塞操作，以实现集合间进度同步。

**参数**

   - **comm_world** (str) - 阻塞操作的通信集合，包含：server 集合(“server")，worker 集合("worker")及所有节点集合("all")，默认为 worker 集合。

**代码示例**

.. code-block:: python

    # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .

    import paddle.distributed.fleet as fleet
    from paddle.distributed.fleet import PaddleCloudRoleMaker
    import sys
    import os

    os.environ["PADDLE_WITH_GLOO"] = "2"

    def train():
        role = PaddleCloudRoleMaker(
            is_collective=False,
            init_gloo=True,
            path="./tmp_gloo")
        fleet.init(role)

        if fleet.is_server():
            fleet.util.barrier("server")
            print("all server arrive here")
        elif fleet.is_worker():
            fleet.util.barrier("worker")
            print("all server arrive here")
        fleet.util.barrier("all")
        print("all servers and workers arrive here")

    if __name__ == "__main__":
        train()

all_gather(input, comm_world="worker")
'''''''''
在指定的通信集合间进行聚合操作，并将聚合的结果返回给集合中每个实例。

**参数**

   - **input** (int|float) - 聚合操作的输入。
   - **comm_world** (str) - 聚合操作的通信集合，包含：server 集合(“server")，worker 集合("worker")及所有节点集合("all")，默认为 worker 集合。

**返回**

   - **output** (List): List 格式的聚合结果。

**代码示例**

.. code-block:: python

    # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .
    import paddle.distributed.fleet as fleet
    from paddle.distributed.fleet import PaddleCloudRoleMaker
    import sys
    import os

    os.environ["PADDLE_WITH_GLOO"] = "2"

    def train():
        role = PaddleCloudRoleMaker(
            is_collective=False,
            init_gloo=True,
            path="./tmp_gloo")
        fleet.init(role)

        if fleet.is_server():
            input = fleet.server_index()
            output = fleet.util.all_gather(input, "server")
            print(output)
            # output = [0, 1]
        elif fleet.is_worker():
            input = fleet.worker_index()
            output = fleet.util.all_gather(input, "worker")
            # output = [0, 1]
            print(output)
        output = fleet.util.all_gather(input, "all")
        print(output)
        # output = [0, 1, 0, 1]

    if __name__ == "__main__":
        train()

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

.. code-block:: python

    import paddle.distributed.fleet as fleet
    import paddle.distributed.fleet.base.role_maker as role_maker

    role = role_maker.UserDefinedRoleMaker(
        is_collective=False,
        init_gloo=False,
        current_id=0,
        role=role_maker.Role.WORKER,
        worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
        server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
    fleet.init(role)

    files = fleet.util.get_file_shard(["file1", "file2", "file3"])
    print(files)
    # files = ["file1", "file2"]

print_on_rank(message, rank_id)
'''''''''

在编号为 `rank_id` 的节点上打印指定信息。

**参数**

    - **message** (str) – 打印内容。
    - **rank_id** (int) - 节点编号。

**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    import paddle.distributed.fleet.base.role_maker as role_maker

    role = role_maker.UserDefinedRoleMaker(
        is_collective=False,
        init_gloo=False,
        current_id=0,
        role=role_maker.Role.WORKER,
        worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
        server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
    fleet.init(role)

    fleet.util.print_on_rank("I'm worker 0", 0)
