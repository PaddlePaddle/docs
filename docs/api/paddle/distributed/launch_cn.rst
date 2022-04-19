.. _cn_api_distributed_launch:

launch
-----

.. py:function:: paddle.distributed.launch()

使用 ``python -m paddle.distributed.launch`` 方法启动分布式训练任务。

使用方法
:::::::::
.. code-block:: bash
    :name: code-block-bash1

    python -m paddle.distributed.launch [-h] [--master MASTER] [--rank RANK]
           [--log_level LOG_LEVEL] [--nnodes NNODES]
           [--nproc_per_node NPROC_PER_NODE] [--log_dir LOG_DIR]
           [--run_mode RUN_MODE] [--job_id JOB_ID] [--devices DEVICES]
           [--host HOST] [--servers SERVERS] [--trainers TRAINERS]
           [--trainer_num TRAINER_NUM] [--server_num SERVER_NUM]
           [--gloo_port GLOO_PORT] [--with_gloo WITH_GLOO]
           [--max_restart MAX_RESTART] [--elastic_level ELASTIC_LEVEL]
           [--elastic_timeout ELASTIC_TIMEOUT]
           training_script ...
    
基础参数
:::::::::
    - ``--master``: 主节点, 支持缺省 http:// 和 etcd://, 默认缺省 http://。例如 ``--master=127.0.0.1:8080``. 默认值 ``--master=None``.

    - ``--rank``: 节点序号, 可以通过主节点进行分配。默认值 ``--rank=-1``.

    - ``--log_level``: 日志级别, 可选值为 CRITICAL/ERROR/WARNING/INFO/DEBUG/NOTSET, 不区分大小写。默认值 ``--log_level=INFO``.

    - ``--nnodes``: 节点数量，支持区间设定以开启弹性模式，比如 ``--nnodes=2:3``. 默认值 ``--nnodes=1``.

    - ``--nproc_per_node``: 每个节点启动的进程数，在 GPU 训练中，应该小于等于系统的 GPU 数量。例如 ``--nproc_per_node=8``

    - ``--log_dir``: 日志输出目录。例如 ``--log_dir=output_dir``。默认值 ``--log_dir=log``。

    - ``--run_mode``: 启动任务的运行模式，可选有 collective/ps/ps-heter。例如 ``--run_mode=ps``。默认值 ``--run_mode=collective``。

    - ``--job_id``: 任务唯一标识，缺省将使用 default，会影响日志命名。例如 ``--job_id=job1``. 默认值 ``--job_id=default``.

    - ``--devices``: 节点上的加速卡设备，支持 gpu/xpu/npu/mlu。例如 ``--devices=0,1,2,3``，这会启动 4 个进程，每个进程绑定到 1 个设备上。

    - ``training_script``: 需要运行的任务脚本，例如 ``traing.py``。

    - ``training_script_args``: ``training_script`` 的输入参数，与普通起任务时输入的参数一样，例如 ``--lr=0.1``。

Collective 参数
:::::::::
    - ``--ips``: [DEPRECATED] 需要运行分布式环境的节点 IP 地址，例如 ``--ips=192.168.0.16,192.168.0.17``。 单机默认值是 ``--ips=127.0.0.1``。

Parameter-Server 参数
:::::::::
    - ``--servers``: 多机分布式任务中，指定参数服务器服务节点的IP和端口，例如 ``--servers="192.168.0.16:6170,192.168.0.17:6170"``。

    - ``--trainers``: 多机分布式任务中，指定参数服务器训练节点的IP和端口，也可只指定IP，例如 ``--trainers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"``。

    - ``--workers``: [DEPRECATED] 同 trainers。

    - ``--heter_workers``: 在异构集群中启动分布式任务，指定参数服务器异构训练节点的IP和端口，例如 ``--heter_workers="192.168.0.16:6172,192.168.0.17:6172"``。

    - ``--trainer_num``: 指定参数服务器训练节点的个数。

    - ``--worker_num``: [DEPRECATED] 同 trainer_num。

    - ``--server_num``: 指定参数服务器服务节点的个数。

    - ``--heter_worker_num``: 在异构集群中启动单机模拟分布式任务, 指定参数服务器异构训练节点的个数。

    - ``--gloo_port``: 参数服务器模式中，用 Gloo 启动时设置的连接端口。同 http_port. Default ``--gloo_port=6767``.

    - ``--with_gloo``: 是否使用 gloo. 默认值 ``--with_gloo=0``.


Elastic 参数
:::::::::
    - ``--max_restart``: 最大重启次数. 默认值 ``--max_restart=3``.

    - ``--elastic_level``: 弹性级别设置，-1: 不开启, 0: 错误节点退出, 1: 节点内重启. 默认值 ``--elastic_level=-1``.

    - ``--elastic_timeout``: 弹性超时时间，经过该时间达到最小节点数即开启训练。默认值 ``--elastic_timeout=30``.

返回
:::::::::
    ``None``

代码示例零 (主节点, ip/port 自动识别)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash0

    # For training on multi node, run the following command in one of the nodes

    python -m paddle.distributed.launch --nnodes 2 train.py

    # Then the following info will be print

    # Copy the following command to other nodes to run.
    # --------------------------------------------------------------------------------
    # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
    # --------------------------------------------------------------------------------

    # Follow the instruction above and paste the command in other nodes can launch a multi nodes training job.

    # There are two ways to launch a job with the same command for multi nodes training
    # 1) using the following command in every nodes, make sure the ip is one of the training node and the port is available on that node
    # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
    # 2) using the following command in every nodes with a independent etcd service
    # python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2 train.py

    # This functionality works will for both collective and ps mode and even with other arguments.


代码示例一 (collective, 单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash1

    # For training on single node using 4 gpus.

    python -m paddle.distributed.launch --gpus=0,1,2,3 train.py --lr=0.01

代码示例二 (collective, 多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash2
    
    # The parameters of --gpus and --ips must be consistent in each node.

    # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 

    # On 192.168.0.16:

    python -m paddle.distributed.launch --gpus=0,1,2,3 --ips=192.168.0.16,192.168.0.17 train.py --lr=0.01

    # On 192.168.0.17:
    
    python -m paddle.distributed.launch --gpus=0,1,2,3 --ips=192.168.0.16,192.168.0.17 train.py --lr=0.01

代码示例三 (ps, cpu, 单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash3

    # To simulate distributed environment using single node, e.g., 2 servers and 4 workers.
    
    python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01

代码示例四 (ps, cpu, 多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash4

    # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

    # On 192.168.0.16:

    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    # On 192.168.0.17:

    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

代码示例五 (ps, gpu, 单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash5

    # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, each worker use single gpu.

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01

代码示例六 (ps, gpu, 多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash6

    # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

    # On 192.168.0.16:

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    # On 192.168.0.17:

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

代码示例七 (ps-heter, cpu + gpu, 单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash7

    # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, two workers use gpu, two workers use cpu.

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --server_num=2 --worker_num=2 --heter_worker_num=2 train.py --lr=0.01

代码示例八 (ps-heter, cpu + gpu, 多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash8

    # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server, 1 gpu worker, 1 cpu worker.
    
    # On 192.168.0.16:

    export CUDA_VISIBLE_DEVICES=0
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

    # On 192.168.0.17:

    export CUDA_VISIBLE_DEVICES=0
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

代码示例九 (elastic)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash9

    # With the following command, the job will begin to run immediately if 4 nodes are ready,
    # or it will run after elastic_timeout if only 2 or 3 nodes ready
    python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2:4 train.py
    
    # once the number of nodes changes between 2:4 during training, the strategy holds
