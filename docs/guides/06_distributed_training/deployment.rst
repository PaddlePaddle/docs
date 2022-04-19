
环境部署
---------------

环境支持列表
^^^^^^^^^^^^^^^^^^^^^^

PaddlePaddle 分布式对不同系统和硬件的支持情况如下表所示，

.. list-table::

   * - 
     - CPU
     - GPU
     - XPU
     - NPU
   * - Linux
     - PS/Collective
     - PS/Collective
     - Collective
     - Collective
   * - Windows
     - Single
     - Single
     - -
     - -


裸机及docker化部署
^^^^^^^^^^^^^^^^^^^^^^

本节针对使用多台裸机使用分布式的场景提供指导，当机器数量多于 5 台且长期使用时，建议部署 kubernetes 或其他类似集群管理工具使用。

无论开发调试还是长期部署训练任务，建议使用 docker 环境，PaddlePaddle 提供了官方镜像供 `下载 <https://www.paddlepaddle.org.cn/install/quick>`_ 使用。

paddle 环境安装
~~~~~~

根据 `安装 <https://www.paddlepaddle.org.cn/install/quick>`_ 部分选择合适的 paddle 版本进行安装或下载对应版本的镜像然后通过以下命令启动

.. code-block::

   $ docker run --name paddle -it --host=net -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8 /bin/bash

* 当使用 gpu 时请配置 nvidia docker runtime 或使用 nvidia-docker 启动容器，进入容器后使用 nvidia-smi 命令确认环境正确
* 使用分布式时需要添加 --host=net 参数让容器使用主机网络以实现跨机建立连接

运行以下命令

.. code-block::

   $ python -c "import paddle; paddle.utils.run_check()"

确保输出结果符合预期以保证 paddle 环境安装正确。 至此，可以进行单机的代码开发和调试工作。

分布式启动
~~~~~~

在多机中安装好环境，同步数据和代码，在任意节点上运行以下命令

.. code-block::

    $ python -m paddle.distributed.launch --nnodes=2 demo.py

nnodes 为本次分布式任务的节点个数，这时会在看到如下输出

.. code-block::

    $ Copy the following command to other nodes to run.
    $ --------------------------------------------------------------------------------
    $ python -m paddle.distributed.launch --master 123.45.67.89:25880 --nnodes=2 demo.py
    $ --------------------------------------------------------------------------------

按照提示复制命令到其他所有节点即可启动分布式训练任务。

* --nnodes 为分布式任务的节点个数，默认为 1 即启动单机任务
* --master 为分布式同步主节点，可以直接由用户设置，这是用户需要配置主节点的 ip 和任意可用端口

更多 launch 启动参数和用法请参考 `文档 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/launch_cn.html>`_ 或通过以下命令获得

.. code-block::

    $ python -m paddle.distributed.launch --help


Kubernetes 部署
^^^^^^^^^^^^^^^^^^^^^^

在 kubernetes 上部署分布式任务需要安装 `paddle-operator <https://github.com/PaddleFlow/paddle-operator>`_ 。
paddle-operator 通过添加自定义资源类型 (paddlejob) 以及部署 controller 和一系列 kubernetes 原生组件的方式实现简单定义即可运行 paddle 任务的需求。

目前支持运行 ParameterServer (PS) 和 Collective 两种分布式任务，当然也支持运行单节点任务。

paddle-operator 安装
~~~~~~

安装 paddle-operator 需要有已经安装的 kubernetes (v1.8+) 集群和 `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/>`_  (v1.8+) 工具。

本节所需配置文件和示例可以在 `这里 <https://github.com/PaddleFlow/paddle-operator/tree/main/deploy>`_ 找到，
可以通过 *git clone* 或者复制文件内容保存。

.. code-block::

    deploy
    |-- examples
    |   |-- resnet.yaml
    |   |-- wide_and_deep.yaml
    |   |-- wide_and_deep_podip.yaml
    |   |-- wide_and_deep_service.yaml
    |   `-- wide_and_deep_volcano.yaml
    |-- v1
    |   |-- crd.yaml
    |   `-- operator.yaml
    `-- v1beta1
        |-- crd.yaml
        `-- operator.yaml


*注意：kubernetes 1.15 及以下使用 v1beta1 目录，1.16 及以上使用目录 v1.*

执行以下命令，

.. code-block::

   $ kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/crd.yaml

或者

.. code-block::

   $ kubectl create -f deploy/v1/crd.yaml

*注意：v1beta1 请根据报错信息添加 --validate=false 选项*

通过以下命令查看是否成功，

.. code-block::

    $ kubectl get crd
    NAME                                    CREATED AT
    paddlejobs.batch.paddlepaddle.org       2021-02-08T07:43:24Z
 

*注意：默认部署的 namespace 为 paddle-system，如果希望在自定义的 namespace 中运行或者提交任务，
需要先在 operator.yaml 文件中对应更改 namespace 配置，其中*

* *namespace: paddle-system* 表示该资源部署的 namespace，可理解为系统 controller namespace；
* Deployment 资源中 containers.args 中 *--namespace=paddle-system* 表示 controller 监控资源所在 namespace，即任务提交 namespace。


执行以下部署命令，

.. code-block::

   $ kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/operator.yaml

或者

.. code-block::

   $ kubectl create -f deploy/v1/operator.yaml

通过以下命令查看部署结果和运行状态，

.. code-block::

    $ kubectl -n paddle-system get pods
    NAME                                         READY   STATUS    RESTARTS   AGE
    paddle-controller-manager-698dd7b855-n65jr   1/1     Running   0          1m

通过查看 controller 日志以确保运行正常，

.. code-block::

    $ kubectl -n paddle-system logs paddle-controller-manager-698dd7b855-n65jr

提交 demo 任务查看效果，

.. code-block::

   $ kubectl -n paddle-system create -f deploy/examples/wide_and_deep.yaml

查看 paddlejob 任务状态, pdj 为 paddlejob 的缩写，

.. code-block::

    $ kubectl -n paddle-system get pdj
    NAME                     STATUS      MODE   AGE
    wide-ande-deep-service   Completed   PS     4m4s

以上信息可以看出：训练任务已经正确完成，该任务为 ps 模式。
可通过 cleanPodPolicy 配置任务完成/失败后的 pod 删除策略，详见任务配置。

查看 pod 状态，

.. code-block::

   $ kubectl -n paddle-system get pods


paddlejob 任务提交
~~~~~~~~~~~~~~~~~~~~~~~~

在上述安装过程中，我们使用了 wide-and-deep 的例子作为提交任务演示，本节详细描述任务配置和提交流程供用户参考提交自己的任务，
镜像的制作过程可在 *docker 镜像* 章节找到。

示例 wide and deep

本示例采用 PS 模式，使用 cpu 进行训练，所以需要配置 ps 和 worker。

准备配置文件，

.. code-block::
    
    $ cat demo-wide-and-deep.yaml
    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: wide-ande-deep
    spec:
      withGloo: 1
      intranet: PodIP
      cleanPodPolicy: OnCompletion
      worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
      ps:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1

说明：

* 提交命名需要唯一，如果存在冲突请先删除原 paddlejob 确保已经删除再提交;
* ps 模式时需要同时配置 ps 和 worker，collective 模式时只需要配置 worker 即可；
* withGloo 可选配置为 0 不启用， 1 只启动 worker 端， 2 启动全部(worker端和Server端)， 建议设置 1；
* cleanPodPolicy 可选配置为 Always/Never/OnFailure/OnCompletion，表示任务终止（失败或成功）时，是否删除 pod，调试时建议 Never，生产时建议 OnCompletion；
* intranet 可选配置为 Service/PodIP，表示 pod 间的通信方式，用户可以不配置, 默认使用 PodIP；
* ps 和 worker 的内容为 podTemplateSpec，用户可根据需要遵从 kubernetes 规范添加更多内容, 如 GPU 的配置.


提交任务: 使用 kubectl 提交 yaml 配置文件以创建任务，

.. code-block::
    
    $ kubectl -n paddle-system create -f demo-wide-and-deep.yaml

示例 resnet
~~~~~~~~~~~~~~~~~~~~~~~~

本示例采用 Collective 模式，使用 gpu 进行训练，所以只需要配置 worker，且需要配置 gpu。

准备配置文件，

.. code-block::

    $ cat resnet.yaml
    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: resnet
    spec:
      cleanPodPolicy: Never
      worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-resnet:v1
                command:
                - python
                args:
                - "-m"
                - "paddle.distributed.launch"
                - "train_fleet.py"
                volumeMounts:
                - mountPath: /dev/shm
                  name: dshm
                resources:
                  limits:
                    nvidia.com/gpu: 1
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
        

注意：

* 这里需要添加 shared memory 挂载以防止缓存出错；
* 本示例采用内置 flower 数据集，程序启动后会进行下载，根据网络环境可能等待较长时间。

提交任务: 使用 kubectl 提交 yaml 配置文件以创建任务，

.. code-block::
    
    $ kubectl -n paddle-system create -f resnet.yaml

卸载
~~~~~~

通过以下命令卸载部署的组件，

.. code-block::

   $ kubectl delete -f deploy/v1/crd.yaml -f deploy/v1/operator.yaml

*注意：重新安装时，建议先卸载再安装*

公有云和私有云部署
^^^^^^^^^^^^^^^^^^^^^^

在公有云上运行 PaddlePaddle 分布式建议通过选购容器引擎服务的方式，各大云厂商都推出了基于标准 kubernetes 的云产品，然后根据上节中的教程安装使用即可。

.. list-table::
  
  * - 云厂商
    - 容器引擎
    - 链接
  * - 百度云
    - CCE
    - https://cloud.baidu.com/product/cce.html
  * - 阿里云
    - ACK
    - https://help.aliyun.com/product/85222.html
  * - 华为云
    - CCE
    - https://www.huaweicloud.com/product/cce.html


更为方便的是使用百度提供的全功能AI开发平台 `BML <https://cloud.baidu.com/product/bml>`_  来使用，详细的使用方式请参考 `这里 <https://ai.baidu.com/ai-doc/BML/pkhxhgo5v>`_ 。
