launch组件详解
------------------

快速开始
~~~~~~~~

飞桨通过\ ``paddle.distributed.launch``\ 组件启动分布式任务。该组件可用于启动单机多卡分布式训练任务，也可以用于启动多机多卡分布式任务。该组件为每张参与分布式训练任务的GPU卡启动一个训练进程。默认情况下，该组件将在每个节点上启动\ ``N``\ 个进程，这里\ ``N``\ 等于训练节点的卡数，即使用一个节点的所有的GPU卡。用户也可以通过\ ``gpus``\ 参数指定训练节点上使用的GPU列表，该列表以逗号分隔。需要注意的是，所有节点使用的GPU卡数量需要相同。

例如通过如下命令启动单机多卡分布式训练任务，本例中每个节点包含8张GPU卡。其中\ ``train.py``\ 为用户训练脚本，后面可以增加脚本参数，如batch size等。这里脚本参数为用户训练脚本的参数。

.. code-block::

   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train.py --batch_size=64

用户也可以只使用部分GPU卡进行训练。如下所示，通过\ ``gpus``\ 参数指定使用2号和3号卡做训练：

.. code-block::

   python -m paddle.distributed.launch --gpus 2,3 train.py --batch_size=64

如果用户希望启动多机分布式训练任务，则需要在每个节点上使用命令调用\ ``paddle.distributed.launch``\ 组件启动分布式任务，并使用\ ``ips``\ 参数指定所有节点的IP地址列表，IP地址以逗号分隔。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序需要保持一致。

例如，假设两台机器的IP地址分别为192.168.0.1和192.168.0.2，那么在这两个节点上启动多机分布式任务的命令如下所示：

.. code-block::
   
   # 192.168.0.1
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

   # 192.168.0.2
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

由上例可见，两个节点上\ ``ips``\ 指定的地址列表的顺序完全一致。此外，用户还可使用\ ``gpus``\ 参数指定每个节点上部分GPU卡参与训练任务，如下所示：

.. code-block::
   
   # 192.168.0.1
   python -m paddle.distributed.launch --gpus=0,1,2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64
   
   # 192.168.0.2
   python -m paddle.distributed.launch --gpus=0,1,2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

两个节点上还可以使用不同的训练卡进行训练，但需要保证各个节点上训练卡的数量相同。例如，第一个节点使用0、1两张卡，第二个节点使用2、3两张卡，启动命令如下所示：

.. code-block::
   
   # 192.168.0.1
   python -m paddle.distributed.launch --gpus=0,1 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64
   
   # 192.168.0.2
   python -m paddle.distributed.launch --gpus=2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

下面将介绍\ ``paddle.distributed.launch``\ 组件在不同场景下的详细使用方法。

Collective架构分布式任务
~~~~~~~~~~~~~~~~~~~~~

在Collective分布式任务场景下，\ ``paddle.distributed.launch``\ 组件支持以下参数：

.. code-block::
   
     -h, --help            给出该帮助信息并退出
     --log_dir LOG_DIR     训练日志的保存目录，默认：--log_dir=log/
     --run_mode RUN_MODE   任务运行模式, 可以为以下值: collective/ps/ps-heter；
                           当为collective模式时可省略。
     --gpus GPUS           训练使用的卡列表，以逗号分隔。例如: --gpus="4,5,6,7"
                           将使用节点上的4，5，6，7四张卡执行任务，并分别为每张卡
                           启动一个任务进程。
     --ips IPS             参与分布式任务的节点IP地址列表，以逗号分隔，例如：
                           192.168.0.16,192.168.0.17
     training_script       用户的任务脚本，其后为该任务脚本的参数。
     training_script_args  用户任务脚本的参数
   
   
各个参数的含义如下：

-  log_dir：训练日志储存目录，默认为\ ``./log``\ 目录。该目录下包含\ ``endpoints.log``\ 文件和各个卡的训练日志 \ ``workerlog.x``\ （如workerlog.0，wokerlog.1等），其中\ ``endpoints.log``\ 文件记录各个训练进程的IP地址和端口号。
-  run_mode：运行模式，如collecitve，ps（parameter-server）或者ps-heter，默认为collective。
-  gpus：每个节点上使用的gpu卡的列表，以逗号间隔。例如\ ``--gpus="0,1,2,3"``\ 。需要注意：这里的指定的卡号为物理卡号，而不是逻辑卡号。
-  ips：所有训练节点的IP地址列表，以逗号间隔。例如，\ ``--ips="192.168.0.1,192.168.0.2``\ 。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序在所有节点的任务脚本中需要保持一致。
-  training_script：训练脚本，如\ ``train.py``\ 。
-  training_script_args：训练脚本的参数，如batch size和学习率等。

通过\ ``paddle.distributed.launch``\ 组件启动分布式任务，将在控制台显示第一张训练卡对应的日志信息，并将所有的日志信息保存到\ ``log_dir``\ 参数指定的目录中；每张训练卡的日志对应一个日志文件，形式如\ ``workerlog.x``\ 。

ParameterServer架构分布式任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ParameterServer相关参数如下：

.. code-block::
   
   --servers             多机分布式任务中，指定参数服务器服务节点的IP和端口
                         例如 --servers="192.168.0.16:6170,192.168.0.17:6170"。
   --workers             多机分布式任务中，指定参数服务器训练节点的IP和端口，
                         也可只指定IP，例如 --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"。
   --heter_workers       在异构集群中启动分布式任务，指定参数服务器异构训练节点
                         的IP和端口，例如 --heter_workers="192.168.0.16:6172,192.168.0.17:6172"。
   --worker_num          单机模拟分布式任务中，指定参数服务器训练节点的个数。
   --server_num          单机模拟分布式任务中，指定参数服务器服务节点的个数。
   --heter_worker_num    在异构集群中启动单机模拟分布式任务, 指定参数服务器异构训练节点的个数。
   --http_port           参数服务器模式中，用 Gloo 启动时设置的连接端口。

Elastic 参数
~~~~~~~~~~~~~~~~~~~~~

.. code-block::
   
   --elastic_server      etcd 服务地址 host:port，例如 --elastic_server=127.0.0.1:2379。
   --job_id              任务唯一 ID，例如 --job_id=job1。
   --np                  任务 pod/node 编号，例如 --np=2。
   --host                绑定的主机，默认等于 POD_IP 环境变量。

