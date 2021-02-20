版本迁移工具
====================

在飞桨框架2.0中，我们API的位置、命名、参数、行为，进行了系统性的调整和规范, 将API体系从1.X版本的 ``paddle.fluid.*`` 迁移到了 ``paddle.*`` 下。paddle.fluid目录下暂时保留了1.8版本API，主要是兼容性考虑，未来会被删除。

使用版本迁移工具自动迁移您的Paddle 1.x的代码到Paddle 2.0的代码
------------------------------------

WARNING: 版本自动迁移工具并不能处理所有的情况，在使用本工具后，您仍然需要手工来进行检查并做相应的调整。

安装
~~~~

版本迁移工具可以通过pip的方式安装，方式如下:

.. code:: ipython3

    $ pip install paddle_upgrade_tool

基本用法
~~~~~~~~

paddle_upgrade_tool 可以使用下面的方式，快速使用:

.. code:: ipython3

    $ paddle_upgrade_tool --inpath /path/to/model.py

这将在命令行中，以\ ``diff``\ 的形式，展示model.py从Paddle 1.x转换为Paddle 2.0的变化。如果您确认上述变化没有问题，只需要再执行：

.. code:: ipython3

    $ paddle_upgrade_tool --inpath /path/to/model.py --write

就会原地改写model.py，将上述变化改写到您的源文件中。
注意：我们会默认备份源文件，到~/.paddle_upgrade_tool/下。

参数说明如下：

-  –inpath 输入文件路径，可以为单个文件或文件夹。
-  –write 是否原地修改输入的文件，默认值False，表示不修改。如果为True，表示对文件进行原地修改。添加此参数也表示对文件进行原地修改。
-  –backup 可选，是否备份源文件，默认值为\ ``~/.paddle_upgrade_tool/``\ ，在此路径下备份源文件。
-  –no-log-file 可选，是否需要输出日志文件，默认值为False，即输出日志文件。
-  –log-filepath 可选，输出日志的路径，默认值为\ ``report.log``\ ，输出日志文件的路径。
-  –no-confirm 可选，输入文件夹时，是否逐文件确认原地写入，只在\ ``--write``\ 为True时有效，默认值为False，表示需要逐文件确认。
-  –parallel 可选，控制转换文件的并发数，当 \ ``no-confirm`` 为True时不生效，默认值:\ ``None``\ 。
-  –log-level 可选，log级别，可为[‘DEBUG’,‘INFO’,‘WARNING’,‘ERROR’] 默认值：\ ``INFO``\ 。
-  –refactor 可选，debug时使用。
-  –print-match 可选，debug时使用。

使用教程
~~~~~~~~

开始
^^^^

在使用paddle_upgrade_tool前，需要确保您已经安装了Paddle 2.0版本。

.. code:: ipython3

    import paddle
    print (paddle.__version__)

.. parsed-literal::

    2.0.0


克隆\ `paddlePaddle/models <https://github.com/PaddlePaddle/models>`__\ 来作为工具的测试。

.. code:: ipython3

    $ git clone https://github.com/PaddlePaddle/models

.. parsed-literal::

    Cloning into 'models'...
    remote: Enumerating objects: 8, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 35011 (delta 1), reused 0 (delta 0), pack-reused 35003[K
    Receiving objects: 100% (35011/35011), 356.97 MiB | 1.53 MiB/s, done.
    Resolving deltas: 100% (23291/23291), done.


查看帮助文档
^^^^^^^^^^^^

您可以直接通过下面的方式，查看帮助文档。

.. code:: ipython3

    $ paddle_upgrade_tool -h


.. parsed-literal::

    usage: paddle_upgrade_tool [-h] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                               [--no-log-file] [--log-filepath LOG_FILEPATH] -i
                               INPATH [-b [BACKUP]] [-w] [--no-confirm]
                               [-p PARALLEL]
                               [-r {refactor_import,norm_api_alias,args_to_kwargs,refactor_kwargs,api_rename,refactor_with,post_refactor}]
                               [--print-match]

    optional arguments:
      -h, --help            show this help message and exit
      --log-level {DEBUG,INFO,WARNING,ERROR}
                            set log level, default is INFO
      --no-log-file         don't log to file
      --log-filepath LOG_FILEPATH
                            set log file path, default is "report.log"
      -i INPATH, --inpath INPATH
                            the file or directory path you want to upgrade.
      -b [BACKUP], --backup [BACKUP]
                            backup directory, default is the
                            "~/.paddle_upgrade_tool/".
      -w, --write           modify files in-place.
      --no-confirm          write files in-place without confirm, ignored without
                            --write.
      -p PARALLEL, --parallel PARALLEL
                            specify the maximum number of concurrent processes to
                            use when refactoring, ignored with --no-confirm.
      -r {refactor_import,norm_api_alias,args_to_kwargs,refactor_kwargs,api_rename,refactor_with,post_refactor}, --refactor {refactor_import,norm_api_alias,args_to_kwargs,refactor_kwargs,api_rename,refactor_with,post_refactor}
                            this is a debug option. Specify refactor you want to
                            run. If none, all refactors will be run.
      --print-match         this is a debug option. Print matched code and node
                            for each file.

Paddle 1.x的例子
^^^^^^^^^^^^^^

这里是一个基于Paddle 1.x实现的一个mnist分类，部分内容如下：

.. code:: ipython3

    $ head -n 198 models/dygraph/mnist/train.py | tail -n  20


.. code:: ipython3

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


使用paddle_upgrade_tool进行转化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

paddle_upgrade_tool支持单文件的转化，您可以通过下方的命令直接转化单独的文件。

.. code:: ipython3

    $ paddle_upgrade_tool --inpath models/dygraph/mnist/train.py

注意，对于参数的删除及一些特殊情况，我们都会打印WARNING信息，需要您仔细核对相关内容。
如果您觉得上述信息没有问题，可以直接对文件进行原地修改，方式如下：

.. code:: ipython3

    $ paddle_upgrade_tool --inpath models/dygraph/mnist/train.py --write 

此时，命令行会弹出下方的提示：

.. code:: ipython3

    "models/dygraph/mnist/train.py" will be modified in-place, and it has been backed up to "~/.paddle_upgrade_tool/train.py_backup_2020_09_09_20_35_15_037821". Do you want to continue? [Y/n]:

输入\ ``y``
后即开始执行代码迁移。为了高效完成迁移，我们这里采用了原地写入的方式。此外，为了防止特殊情况，我们会备份转换前的代码到
``~/.paddle_upgrade_tool`` 目录下，如果需要，您可以在备份目录下找到转换前的代码。

代码迁移完成后，会生成一个report.log文件，记录了迁移的详情。内容如下：

.. code:: ipython3

    $ cat report.log

注意事项
~~~~~~~~

-  本迁移工具不能完成所有API的迁移，有少量的API需要您手动完成迁移，具体信息可见WARNING。

使用Paddle 2.0
~~~~~~~~~~~~~~~~

完成迁移后，代码就从Paddle 1.x迁移到了Paddle 2.0，您就可以在Paddle 2.0下进行相关的开发。
