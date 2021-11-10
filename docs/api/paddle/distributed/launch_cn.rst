.. _cn_api_distributed_launch:

launch
-----

.. py:function:: paddle.distributed.launch()

使用 ``python -m paddle.distributed.launch`` 方法启动分布式训练任务。

使用方法
:::::::::
.. code-block:: bash
    :name: code-block-bash1

    python -m paddle.distributed.launch [-h] [--log_dir LOG_DIR] [--nproc_per_node NPROC_PER_NODE] [--run_mode RUN_MODE] [--gpus GPUS]
                     [--selected_gpus GPUS] [--ips IPS] [--servers SERVERS] [--workers WORKERS] [--heter_workers HETER_WORKERS]
                     [--worker_num WORKER_NUM] [--server_num SERVER_NUM] [--heter_worker_num HETER_WORKER_NUM]
                     [--http_port HTTP_PORT] [--elastic_server ELASTIC_SERVER] [--job_id JOB_ID] [--np NP] [--scale SCALE]
                     [--host HOST] [--force FORCE]
                     training_script ...    
    
基础参数
:::::::::
    - ``--log_dir``: 日志输出目录。例如 ``--log_dir=output_dir``。默认值 ``--log_dir=log``。

    - ``--nproc_per_node``: 每个节点启动的进程数，在 GPU 训练中，应该小于等于系统的 GPU 数量（或者也可以通过 --gpus 来设置）。例如 ``--nproc_per_node=8``

    - ``--run_mode``: 启动任务的运行模式，可选有 collective/ps/ps-heter。例如 ``--run_mode=ps``。默认值 ``--run_mode=collective``。

    - ``--gpus``: GPU 训练模式下对 GPU 设置。例如 ``--gpus=0,1,2,3``，这会启动 4 个进程，每个进程绑定到 1 个 GPU 上。

    - ``--selected_gpus``: ``--gpus`` 的别名， 作用是一样的，推荐使用 ``--gpus``。

    - ``--xpus``: 如果 XPU 可用且想使用 XPU 训练，则用该参数，例如 ``--xpus=0,1,2,3``。

    - ``--selected_xpus``: ``--xpus`` 的别名，推荐使用 ``--xpus``。

    - ``training_script``: 需要运行的任务脚本，例如 ``traing.py``。

    - ``training_script_args``: ``training_script`` 的输入参数，与普通起任务时输入的参数一样，例如 ``--lr=0.1``。

Collective 参数
:::::::::
    - ``--ips``: 需要运行分布式环境的节点 IP 地址，例如 ``--ips=192.168.0.16,192.168.0.17``。 单机默认值是 ``--ips=127.0.0.1``。

Parameter-Server 参数
:::::::::
    - ``--servers``: 多机分布式任务中，指定参数服务器服务节点的IP和端口，例如 ``--servers="192.168.0.16:6170,192.168.0.17:6170"``。

    - ``--workers``: 多机分布式任务中，指定参数服务器训练节点的IP和端口，也可只指定IP，例如 ``--workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"``。

    - ``--heter_workers``: 在异构集群中启动分布式任务，指定参数服务器异构训练节点的IP和端口，例如 ``--heter_workers="192.168.0.16:6172,192.168.0.17:6172"``。

    - ``--worker_num``: 单机模拟分布式任务中，指定参数服务器训练节点的个数。

    - ``--server_num``: 单机模拟分布式任务中，指定参数服务器服务节点的个数。

    - ``--heter_worker_num``: 在异构集群中启动单机模拟分布式任务, 指定参数服务器异构训练节点的个数。

    - ``--http_port``: 参数服务器模式中，用 Gloo 启动时设置的连接端口。

Elastic 参数
:::::::::
    - ``--elastic_server``: etcd 服务地址 host:port，例如 ``--elastic_server=127.0.0.1:2379``。

    - ``--job_id``: 任务唯一 ID，例如 ``--job_id=job1``。

    - ``--np``: 任务 pod/node 编号，例如 ``--np=2``。

    - ``--host``: 绑定的主机，默认等于 ``POD_IP`` 环境变量。

返回
:::::::::
    ``None``

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

    python -m paddle.distributed.launch --elastic_server=127.0.0.1:2379 --np=2 --job_id=job1  --gpus=0,1,2,3 train.py
