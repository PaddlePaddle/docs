.. _cn_doc_device:

资源配置
==================

飞桨框架提供命令行启动命令\ ``fleetrun``\ ，可以快速实现模型在GPU单机单卡训练与GPU单机多卡训练。

1. 单机单卡
------------------

单机单卡有两种方式：一种直接使用\ ``python``\ 执行， 另一种使用\ ``fleetrun``\ 执行。推荐使用\ ``fleetrun``\ 启动方法。

方法一：直接使用\ ``python``\ 执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    $ export CUDA_VISIBLE_DEVICES=0
    $ python train.py

方法二：使用\ ``fleetrun``\ 执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
    
    $ fleetrun --gpus=0 train.py

注：如果指定了\ ``export CUDA_VISIBLE_DEVICES=0``\ ,则可以直接使用：

.. code:: bash

    $ export CUDA_VISIBLE_DEVICES=0
    $ fleetrun train.py

2. 单机多卡
-----------------

对于高层API来实现单机多卡非常简单，整个训练代码和单机单卡没有差异。直接使用\ ``flettrun``\ 启动单机单卡的程序，通过\ ``--gpus``\ 指定空闲的显卡即可。如使用4张显卡，命令如下：

.. code:: bash

    $ fleetrun --gpus=0,1,2,3 train.py

注：如果指定了\ ``export CUDA_VISIBLE_DEVICES=0,1,2,3``\ ，则可以直接使用：
    
.. code:: bash

    $ export CUDA_VISIBLE_DEVICES=0,1,2,3
    $ fleetrun train.py

3. 多机多卡
-------------------

通过\ ``fleetrun``\ ，也可以快速的实现多机多卡训练，此时需要通过\ ``--ips``\ 制定每台机器的IP。如实现2机8卡(每台机器上4张显卡)，则可以使用:

.. code:: bash

    $ fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus=0,1,2,3 train.py

注：如果每台机器都指定了\ ``export CUDA_VISIBLE_DEVICES=0,1,2,3``\ ，则可以直接使用：

.. code:: bash

    $ export CUDA_VISIBLE_DEVICES=0,1,2,3
    $ fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py
