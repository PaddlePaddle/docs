.. _cn_api_distributed_IPULaunch:

IPULaunch
-------------------------------

IPULaunch负责启动IPU分布式训练任务。

参数
:::::::::
    - **hosts** (String): 指定训练节点ip。举例：--hosts=192.168.0.3,192.168.0.5 表示指定使用2个节点。
    - **ipus_per_replica** (Int): 每个 replica 的IPU数量。举例：--ipus_per_replica=8 表示每个 replica 需要8个IPUs。
    - **nproc_per_host** (Int): 每个节点的进程数量。举例：--nproc_per_host=2 表示每个节点需要2个进程。
    - **ipu_partition** (String): 选择IPU分区，如果没有对应的分区，则创建分区。
    - **vipu_server** (String): 设置Vipu server的ip。可以通过 vipu-admin --server-version 在 docker 容器外查看。
    - **training_script** (String): IPU训练脚本的绝对路径。训练脚本之后所跟的参数均属于训练脚本。
    - **training_script_args** (String): IPU训练脚本参数。

代码示例
:::::::::
.. code-block:: bash
            
    # 使用如下命令启动IPU分布式训练
    # 启动1个节点，每个节点启动2个进程，每个进程需要2个replicas，每个replica需要1个IPU
    python -m paddle.distributed.launch --device_num 4 ipu --hosts=localhost --nproc_per_host=2 --ipus_per_replica=1 --ipu_partition=pod16 --vipu_server=127.0.0.1 train.py
