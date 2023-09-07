.. _cn_api_paddle_distributed_launch:

launch
-----

.. py:function:: paddle.distributed.launch()

使用 ``python -m paddle.distributed.launch`` 方法启动分布式训练任务。

Launch 模块是在每个节点运行，负责分布式协同和本地进程管理的模块。使用 launch 启动分布式训练可以简化参数配置，进行稳定可靠的分布式组网训练，同时使用优化的调试和日志收集功能。另外一些高级的分布式功能如容错和弹性都依赖 launch 启动。

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
    - ``--master``：主节点，支持缺省 ``http://`` 和 ``etcd://``，默认缺省 ``http://``。例如 ``--master=127.0.0.1:8080``。默认值 ``--master=None``。

    - ``--rank``：节点序号，可以通过主节点进行分配。默认值 ``--rank=-1``。

    - ``--log_level``：日志级别，可选值为 CRITICAL/ERROR/WARNING/INFO/DEBUG/NOTSET，不区分大小写。默认值 ``--log_level=INFO``。

    - ``--nnodes``：节点数量，支持区间设定以开启弹性模式，比如 ``--nnodes=2:3``。默认值 ``--nnodes=1``。

    - ``--nproc_per_node``：每个节点启动的进程数，在 GPU 训练中，应该小于等于系统的 GPU 数量。例如 ``--nproc_per_node=8``

    - ``--log_dir``：日志输出目录。例如 ``--log_dir=output_dir``。默认值 ``--log_dir=log``。

    - ``--run_mode``：启动任务的运行模式，可选有 collective/ps/ps-heter。例如 ``--run_mode=ps``。默认值 ``--run_mode=collective``。

    - ``--job_id``：任务唯一标识，缺省将使用 default，会影响日志命名。例如 ``--job_id=job1``。默认值 ``--job_id=default``。

    - ``--devices``：节点上的加速卡设备，支持 gpu/xpu/npu/mlu。例如 ``--devices=0,1,2,3``，这会启动 4 个进程，每个进程绑定到 1 个设备上。

    - ``training_script``：需要运行的任务脚本，例如 ``training.py``。

    - ``training_script_args``: ``training_script`` 的输入参数，与普通起任务时输入的参数一样，例如 ``--lr=0.1``。

Collective 参数
:::::::::
    - ``--ips``: [DEPRECATED] 需要运行分布式环境的节点 IP 地址，例如 ``--ips=192.168.0.16,192.168.0.17``。单机默认值是 ``--ips=127.0.0.1``。

Parameter-Server 参数
:::::::::
    - ``--servers``：多机分布式任务中，指定参数服务器服务节点的 IP 和端口，例如 ``--servers="192.168.0.16:6170,192.168.0.17:6170"``。

    - ``--trainers``：多机分布式任务中，指定参数服务器训练节点的 IP 和端口，也可只指定 IP，例如 ``--trainers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"``。

    - ``--workers``: [DEPRECATED] 同 trainers。

    - ``--heter_workers``：在异构集群中启动分布式任务，指定参数服务器异构训练节点的 IP 和端口，例如 ``--heter_workers="192.168.0.16:6172,192.168.0.17:6172"``。

    - ``--trainer_num``：指定参数服务器训练节点的个数。

    - ``--worker_num``: [DEPRECATED] 同 trainer_num。

    - ``--server_num``：指定参数服务器服务节点的个数。

    - ``--heter_worker_num``：在异构集群中启动单机模拟分布式任务，指定参数服务器异构训练节点的个数。

    - ``--gloo_port``：参数服务器模式中，用 Gloo 启动时设置的连接端口。同 http_port. Default ``--gloo_port=6767``。

    - ``--with_gloo``：是否使用 gloo。默认值 ``--with_gloo=0``。


Elastic 参数
:::::::::
    - ``--max_restart``：最大重启次数。默认值 ``--max_restart=3``。

    - ``--elastic_level``：弹性级别设置，-1：不开启，0：错误节点退出，1：节点内重启。默认值 ``--elastic_level=-1``。

    - ``--elastic_timeout``：弹性超时时间，经过该时间达到最小节点数即开启训练。默认值 ``--elastic_timeout=30``。

IPU 参数
:::::::::
    IPU 分布式训练只需要 3 个参数：``--devices``，``training_script`` 和 ``training_script_args``。对于 IPU 的参数说明如下：
    ``--devices`` 表示设备个数，例如 ``--devices=4`` 表示当前的训练程序需要 4 个 IPUs。
    ``training_script`` 只允许设置为 ``ipu`` 。
    ``training_script_args`` 表示启动 IPU 分布式训练的相关参数。请参看如下各项参数说明。
    请参考 ``代码实例十``。

    - ``--hosts``：IPU 分布式训练的主机 ip，一个主机可包含多个进程。

    - ``--nproc_per_host``： 每个主机的进程数量。一个进程可包含多个实例。

    - ``--ipus_per_replica``：每个实例包含的 IPU 数量。一个实例可包含多个 IPUs。

    - ``--ipu_partition``：分布式训练中使用的 IPU 分区名称。

    - ``--vipu_server``：IPU 设备管理服务的 ip。

    - ``training_script``：分布式训练任务脚本的绝对路径，例如 ``training.py`` 。

    - ``training_script_args``：``training_script`` 的输入参数，与普通起任务时输入的参数一样，例如 ``--lr=0.1``。

返回
:::::::::
    ``None``

代码示例零 (主节点，ip/port 自动识别)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash0

    # 在其中一个节点上运行如下命令以启动 2 机任务

    python -m paddle.distributed.launch --nnodes 2 train.py

    # 这时，日志会打印如下信息，

    # Copy the following command to other nodes to run.
    # --------------------------------------------------------------------------------
    # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
    # --------------------------------------------------------------------------------

    # 按照提示，复制命令在另外的节点上运行命令即可启动分布式训练。

    # 要想在每个节点上运行同样的命令启动分布式训练有如下两种方法：
    # 1) 使用预配置的 master 信息，其中 master 的 ip 为其中一个训练节点，端口为可用端口
    # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
    # 2) 使用额外部署的 etcd 服务作为 master
    # python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2 train.py

    # 以上功能介绍可用配合别的参数使用。


代码示例一 (collective，单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash1

    # 启动单机 4 卡任务

    python -m paddle.distributed.launch --devices=0,1,2,3 train.py --lr=0.01

代码示例二 (collective，多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash2

    # 启动两机任务，其中机器 ip 为 192.168.0.16, 192.168.0.17

    # On 192.168.0.16:

    python -m paddle.distributed.launch --devices=0,1,2,3 --master=192.168.0.16:8090 --nnodes=2 train.py --lr=0.01

    # On 192.168.0.17:

    python -m paddle.distributed.launch --devices=0,1,2,3 --master=192.168.0.16:8090 --nnodes=2 train.py --lr=0.01

代码示例三 (ps, cpu，单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash3

    # 在单机上启动多个 server 和 trainer

    python -m paddle.distributed.launch --server_num=2 --trainer_num=4 train.py --lr=0.01

代码示例四 (ps, cpu，多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash4

    # 在多机上启动，例如在 192.168.0.16, 192.168.0.17 分别启动 1 个 server 和 2 个 trainer

    # On 192.168.0.16:

    python -m paddle.distributed.launch --master=192.168.0.16:8090 --nnodes=2 --server_num=1 --trainer_num=2 train.py --lr=0.01

    # On 192.168.0.17:

    python -m paddle.distributed.launch --master=192.168.0.16:8090 --nnodes=2 --server_num=1 --trainer_num=2 train.py --lr=0.01

代码示例五 (ps, gpu，单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash5

    # 当启动 gpu ps 时，需要指定使用的 gpu，

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01

代码示例六 (ps, gpu，多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash6

    # 使用如下命令启动多机 gpu ps

    # On 192.168.0.16:

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    # On 192.168.0.17:

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

代码示例七 (ps-heter, cpu + gpu，单机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash7

    # 使用如下命令启动单机 heter ps

    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --server_num=2 --worker_num=2 --heter_worker_num=2 train.py --lr=0.01

代码示例八 (ps-heter, cpu + gpu，多机)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash8

    # 使用如下命令启动多机 heter ps

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

    # 使用如下命令启动弹性训练
    # 当 4 个节点 ready 时，训练立即开始，当只有 2 或 3 个节点 ready 时，将等待超时然后开始训练
    python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2:4 train.py

    # 在训练过程中如果节点发生变化，上述逻辑不变。

代码示例十 (ipu)
:::::::::
.. code-block:: bash
    :name: code-block-example-bash10

    # 使用如下命令启动 IPU 分布式训练
    # 要求 `devices` 表示分布式训练的设备数量
    # 要求 `training_script` 设置为 `ipu`
    # 要求 `training_script_args` 表示 IPU 分布式训练相关参数，非训练运行脚本参数
    # 请参看上述 `IPU 参数` 说明
    python -m paddle.distributed.launch --devices 4 ipu --hosts=localhost --nproc_per_host=2 --ipus_per_replica=1 --ipu_partition=pod16 --vipu_server=127.0.0.1 train.py
