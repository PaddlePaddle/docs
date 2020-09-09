.. _cn_guides_migration:

Paddle 1 to Paddle 2
====================

飞桨框架v2.0-beta，最重要的变化为API体系的全面升级以及动态图能力的全面完善。下文将简要介绍Paddle 2的变化，

主要变化
--------

在飞桨框架v2.0中，我们做了许多的升级。首先，我们全面提升了Paddle 动态图的能力。相较于静态图而言，动态图每次执行一个运算，可以立即得到结果，能够使算法的开发变得更加高效。此外，本版本对API目录，进行了较大的调整。将API体系从1.X版本的
``paddle.fluid.*`` 迁移到了 ``paddle.*`` 下。原则上，Paddle
2仍支持Paddle 1下的所有语法。但是，我们会逐步废弃掉 ``paddle.fluid``
下的API，强烈建议您将Paddle 1的代码迁移到Paddle
2下，以避免后续带来不必要的麻烦。下文将介绍手动与自动两种方式，来完成Paddle
1到Paddle 2的迁移。

手动将Paddle 1 的代码迁移到 Paddle 2
------------------------------------

本节将介绍如何将您的代码手动的从Paddle 1迁移到Paddle 2。

1、API的变化
~~~~~~~~~~~~

对于Paddle
1下的API，您可以通过我们提供的API升级表（TODO），查看每个API的升级关系，从而手动完成修改。

2、句法的变化 
~~~~~~~~~~~~

在Paddle 1中，通过 ``with fluid.dygraph.guard():``
开启动态图模式，在Paddle 2.0-beta中，可以直接通过
``paddle.disable_static()``\ 开启动态图。

Paddle1to2 自动迁移您的代码到Paddle2
------------------------------------

Paddle 2 包含了许多API的变化，为了节约您将代码从Paddle 1迁移到Paddle
2的时间，我们提供了自动迁移工具–Paddle1to2，能够帮助您快速完成代码迁移。

安装方式
~~~~~~~~

paddle1to2 可以直接通过pip的方式安装，方式如下：

.. code:: ipython3

    ! pip install -U paddle1to2

基本用法
~~~~~~~~

.. code:: ipython3

    ! paddle1to2 --inpath /path/to/model.py

这将在命令行中，以\ ``diff``\ 的形式，展示model.py从Paddle 1转换为Paddle
2的变化。如果您确认上述变化没有问题，只需要再执行：

.. code:: ipython3

    ! paddle1to2 --inpath /path/to/model.py --write

就会原地改写model.py，将上述变化改写到您的源文件中。
注意：我们会默认备份源文件，到~/.paddle1to2/下。

参数说明如下：

-  **--inpath** 输入文件路径，可以为单个文件或文件夹。
-  **--write**
   是否原地修改输入的文件，默认值False，表示不修改。如果为True，表示对文件进行原地修改。添加此参数也表示对文件进行原地修改。
-  **--backup**
   可选，是否备份源文件，默认值为 ``~/.paddle1to2/`` ，在此路径下备份源文件。
-  **--no-log-file**
   可选，是否需要输出日志文件，默认值为False，即输出日志文件。
-  **--log-filepath**
   可选，输出日志的路径，默认值为 ``report.log`` ，输出日志文件的路径。
-  **--no-confirm** 可选，输入文件夹时，是否逐文件确认原地写入。只在 ``--write`` 为True时有效，默认值为False，表示不需要逐文件确认。
-  **--log-level** 可选，log级别，可为[‘DEBUG’,‘INFO’,‘WARNING’,‘ERROR’]
   默认值：``INFO``
-  **--refactor** 可选，debug时使用。
-  **--print-match** 可选，debug时使用。

使用教程
~~~~~~~~

开始
^^^^

在使用Paddle 1to2前，需要确保您已经安装了Paddle 2.0-beta版本。

.. code:: ipython3

    import paddle
    print (paddle.__version__)
    # TODO change to paddle 2.0-beta


.. parsed-literal::

    0.0.0


克隆\ `PaddlePaddle/models <https://github.com/PaddlePaddle/models>`__\ 来作为工具的测试。

.. code:: ipython3

    ! git clone https://github.com/PaddlePaddle/models


.. parsed-literal::

    Cloning into 'models'...
    remote: Enumerating objects: 8, done.
    remote: Counting objects: 100% (8/8), done.
    remote: Compressing objects: 100% (8/8), done.
    remote: Total 35011 (delta 1), reused 0 (delta 0), pack-reused 35003
    Receiving objects: 100% (35011/35011), 356.97 MiB | 1.53 MiB/s, done.
    Resolving deltas: 100% (23291/23291), done.


查看帮助文档
^^^^^^^^^^^^

安装后，您可以通过下面的方式，查看paddle1to2的帮助文档。

.. code:: ipython3

    ! paddle1to2 -h


.. parsed-literal::

    usage: paddle1to2 [-h] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                      [--no-log-file] [--log-filepath LOG_FILEPATH] --inpath
                      INPATH [--backup [BACKUP]] [--write]
                      [--refactor {refactor_import,norm_api_alias,args_to_kwargs,refactor_kwargs,api_rename,refactor_with,post_refactor}]
                      [--print-match]
    
    optional arguments:
      -h, --help            show this help message and exit
      --log-level {DEBUG,INFO,WARNING,ERROR}
                            Set log level, default is INFO
      --no-log-file         Don't log to file
      --log-filepath LOG_FILEPATH
                            Set log file path, default is "report.log"
      --inpath INPATH       The file or directory path you want to upgrade.
      --backup [BACKUP]     backup directory, default is the "~/.paddle1to2/".
      --write               Modify files in place.
      --no-confirm          write files in-place without confirm, ignored without
                            --write.
      --refactor {refactor_import,norm_api_alias,args_to_kwargs,refactor_kwargs,api_rename,refactor_with,post_refactor}
                            This is a debug option. Specify refactor you want to
                            run. If none, all refactors will be run.
      --print-match         This is a debug option. Print matched code and node
                            for each file.


Paddle 1的例子
^^^^^^^^^^^^^^

这里是一个基于Paddle 1实现的一个mnist分类，部分内容如下：

.. code:: ipython3

    ! head -n 198 models/dygraph/mnist/train.py | tail -n  20


.. parsed-literal::

        with fluid.dygraph.guard(place):
            if args.ce:
                print("ce mode")
                seed = 33
                np.random.seed(seed)
                fluid.default_startup_program().random_seed = seed
                fluid.default_main_program().random_seed = seed
    
            if args.use_data_parallel:
                strategy = fluid.dygraph.parallel.prepare_context()
            mnist = MNIST()
            adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
            if args.use_data_parallel:
                mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)
    
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
            if args.use_data_parallel:
                train_reader = fluid.contrib.reader.distributed_batch_reader(
                    train_reader)


使用Paddle1to2进行转化
^^^^^^^^^^^^^^^^^^^^^^

paddle1to2支持单文件的转化，您可以通过下方的命令直接转化单独的文件。

.. code:: ipython3

    !paddle1to2 --inpath models/dygraph/mnist/train.py


注意，对于参数的删除及一些特殊情况，我们都会打印WARNING信息，需要您仔细核对相关内容。
如果您觉得上述信息没有问题，可以直接对文件进行原地修改，方式如下：

.. code:: ipython3

    !paddle1to2 --inpath models/dygraph/mnist/train.py --write 

此时，命令行会弹出下方的提示：

.. code:: ipython3

    Files will be modified in-place, but don't worry, we will backup your files to your_path/.paddle1to2 automatically. do you want to continue? [y/N]:

输入\ ``y``
后即开始执行代码迁移。为了高效完成迁移，我们这里采用了原地写入的方式。此外，为了防止特殊情况，我们会备份转换前的代码到
``~/.paddle1to2`` 目录下，如果需要，您可以在备份目录下找到转换前的代码。

代码迁移完成后，会生成一个report.log文件，记录了迁移的详情。内容如下：

.. code:: ipython3

    ! cat report.log

注意事项
~~~~~~~~

-  本迁移工具不能完成所有API的迁移，有少量的API需要您手动完成迁移，具体信息可见WARNING。

使用Paddle 2
~~~~~~~~~~~~

完成迁移后，代码就从Paddle 1迁移到了Paddle 2，您就可以在Paddle
2下进行相关的开发。
