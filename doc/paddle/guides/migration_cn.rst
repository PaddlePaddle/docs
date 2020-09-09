.. _cn_guides_migration:

Paddle 1 to Paddle 2
====================

飞桨框架v2.0的测试版，最重要的变化为API体系的全面升级以及动态图能力的全面完善。本版本飞桨的默认开发模式为动态图模式，

主要变化
--------

在飞桨框架v2.0中，我们做了许多的升级。首先，将默认开发模式设为了动态图模式，相较于静态图而言，动态图每次执行一个运算，可以立即得到结果，能够使算法的开发变得更加高效。此外，本版本对API目录，进行了较大的调整。将API体系从1.X版本的
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
### 2、句法的变化 在Paddle 1中，通过 ``with fluid.dygraph.guard():``
开启动态图模式，在Paddle 2.0-beta中，可以直接通过
``paddle.disable_static()``\ 开启动态图。

Paddle1to2 自动迁移您的代码到Paddle2
------------------------------------

Paddle 2 包含了许多API的变化，为了节约您将代码从Paddle 1迁移到Paddle
2的时间，我们提供了自动迁移工具–Paddle1to2，能够帮助您快速完成代码迁移。

注意：Paddle1to2 工具随Paddle 2.0-beta安装，您无需额外安装，即可使用。

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

-  –inpath 输入文件路径，可以为单个文件或文件夹。
-  –write
   是否原地修改输入的文件，默认值False，表示不修改。如果为True，表示对文件进行原地修改。添加此参数也表示对文件进行原地修改。
-  –backup
   可选，是否备份源文件，默认值为’~/.paddle1to2/’，在此路径下备份源文件。
-  –no-log-file
   可选，是否需要输出日志文件，默认值为False，即输出日志文件。
-  –log-filepath
   可选，输出日志的路径，默认值为“report.log”，输出日志文件的路径。
-  –log-level 可选，log级别，可为[‘DEBUG’,‘INFO’,‘WARNING’,‘ERROR’]
   默认值：‘INFO’
-  –refactor 可选，debug时使用。
-  –print-match 可选，debug时使用。

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
    remote: Enumerating objects: 8, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 35011 (delta 1), reused 0 (delta 0), pack-reused 35003[K
    Receiving objects: 100% (35011/35011), 356.97 MiB | 1.53 MiB/s, done.
    Resolving deltas: 100% (23291/23291), done.


查看帮助文档
^^^^^^^^^^^^

paddle1to2 会随着 paddle
2.0-beta安装。所以您可以直接通过下面的方式，查看帮助文档。

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


.. parsed-literal::

    [33;21m2020-09-09 15:20:09,654 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:22 remove "import paddle.fluid as fluid"[0m
    [33;21m2020-09-09 15:20:09,656 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:23 remove "from paddle.fluid.optimizer import AdamOptimizer"[0m
    [33;21m2020-09-09 15:20:09,657 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:24 remove "from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear"[0m
    [33;21m2020-09-09 15:20:09,658 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:25 remove "from paddle.fluid.dygraph.base import to_variable"[0m
    [38;21m2020-09-09 15:20:09,659 - utils.py:23 - INFO - models/dygraph/mnist/train.py:42 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,661 - utils.py:23 - INFO - models/dygraph/mnist/train.py:62 Conv2D -> paddle.fluid.dygraph.nn.Conv2D[0m
    [38;21m2020-09-09 15:20:09,662 - utils.py:23 - INFO - models/dygraph/mnist/train.py:75 Pool2D -> paddle.fluid.dygraph.nn.Pool2D[0m
    [38;21m2020-09-09 15:20:09,663 - utils.py:23 - INFO - models/dygraph/mnist/train.py:89 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,665 - utils.py:23 - INFO - models/dygraph/mnist/train.py:102 Linear -> paddle.fluid.dygraph.nn.Linear[0m
    [38;21m2020-09-09 15:20:09,665 - utils.py:23 - INFO - models/dygraph/mnist/train.py:103 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,666 - utils.py:23 - INFO - models/dygraph/mnist/train.py:104 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,667 - utils.py:23 - INFO - models/dygraph/mnist/train.py:111 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,668 - utils.py:23 - INFO - models/dygraph/mnist/train.py:114 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,670 - utils.py:23 - INFO - models/dygraph/mnist/train.py:129 to_variable -> paddle.fluid.dygraph.base.to_variable[0m
    [38;21m2020-09-09 15:20:09,671 - utils.py:23 - INFO - models/dygraph/mnist/train.py:130 to_variable -> paddle.fluid.dygraph.base.to_variable[0m
    [38;21m2020-09-09 15:20:09,671 - utils.py:23 - INFO - models/dygraph/mnist/train.py:133 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,672 - utils.py:23 - INFO - models/dygraph/mnist/train.py:134 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,673 - utils.py:23 - INFO - models/dygraph/mnist/train.py:146 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,674 - utils.py:23 - INFO - models/dygraph/mnist/train.py:146 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,675 - utils.py:23 - INFO - models/dygraph/mnist/train.py:147 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,675 - utils.py:23 - INFO - models/dygraph/mnist/train.py:148 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,676 - utils.py:23 - INFO - models/dygraph/mnist/train.py:151 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,678 - utils.py:23 - INFO - models/dygraph/mnist/train.py:168 to_variable -> paddle.fluid.dygraph.base.to_variable[0m
    [38;21m2020-09-09 15:20:09,679 - utils.py:23 - INFO - models/dygraph/mnist/train.py:177 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,679 - utils.py:23 - INFO - models/dygraph/mnist/train.py:177 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,680 - utils.py:23 - INFO - models/dygraph/mnist/train.py:178 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,681 - utils.py:23 - INFO - models/dygraph/mnist/train.py:179 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,681 - utils.py:23 - INFO - models/dygraph/mnist/train.py:184 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,682 - utils.py:23 - INFO - models/dygraph/mnist/train.py:185 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,683 - utils.py:23 - INFO - models/dygraph/mnist/train.py:188 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,684 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 AdamOptimizer -> paddle.fluid.optimizer.AdamOptimizer[0m
    [38;21m2020-09-09 15:20:09,684 - utils.py:23 - INFO - models/dygraph/mnist/train.py:192 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,685 - utils.py:23 - INFO - models/dygraph/mnist/train.py:197 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,687 - utils.py:23 - INFO - models/dygraph/mnist/train.py:210 to_variable -> paddle.fluid.dygraph.base.to_variable[0m
    [38;21m2020-09-09 15:20:09,688 - utils.py:23 - INFO - models/dygraph/mnist/train.py:211 to_variable -> paddle.fluid.dygraph.base.to_variable[0m
    [38;21m2020-09-09 15:20:09,689 - utils.py:23 - INFO - models/dygraph/mnist/train.py:216 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,689 - utils.py:23 - INFO - models/dygraph/mnist/train.py:217 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,691 - utils.py:23 - INFO - models/dygraph/mnist/train.py:244 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,692 - utils.py:23 - INFO - models/dygraph/mnist/train.py:246 fluid -> paddle.fluid[0m
    [38;21m2020-09-09 15:20:09,694 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.nn.Conv2D -> paddle.fluid.dygraph.Conv2D[0m
    [38;21m2020-09-09 15:20:09,695 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.initializer.NormalInitializer -> paddle.fluid.initializer.Normal[0m
    [38;21m2020-09-09 15:20:09,696 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.param_attr.ParamAttr -> paddle.fluid.ParamAttr[0m
    [38;21m2020-09-09 15:20:09,696 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.nn.Linear -> paddle.fluid.dygraph.Linear[0m
    [38;21m2020-09-09 15:20:09,698 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable[0m
    [38;21m2020-09-09 15:20:09,698 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable[0m
    [38;21m2020-09-09 15:20:09,701 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable[0m
    [38;21m2020-09-09 15:20:09,703 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.optimizer.AdamOptimizer -> paddle.fluid.optimizer.Adam[0m
    [38;21m2020-09-09 15:20:09,705 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable[0m
    [38;21m2020-09-09 15:20:09,706 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable[0m
    [38;21m2020-09-09 15:20:09,708 - utils.py:23 - INFO - models/dygraph/mnist/train.py:63 args_list: "['num_channels', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'dtype']" is longer than positional arguments, redundant arguments will be skipped.[0m
    [38;21m2020-09-09 15:20:09,709 - utils.py:23 - INFO - models/dygraph/mnist/train.py:103 args_list: "['input_dim', 'output_dim', 'param_attr', 'bias_attr', 'act', 'dtype']" is longer than positional arguments, redundant arguments will be skipped.[0m
    [38;21m2020-09-09 15:20:09,712 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 args_list: "['learning_rate', 'beta1', 'beta2', 'epsilon', 'parameter_list', 'regularization', 'grad_clip', 'name', 'lazy_mode']" is longer than positional arguments, redundant arguments will be skipped.[0m
    [33;21m2020-09-09 15:20:09,717 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "num_channels" to "in_channels".[0m
    [33;21m2020-09-09 15:20:09,717 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "num_filters" to "out_channels".[0m
    [33;21m2020-09-09 15:20:09,718 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "filter_size" to "kernel_size".[0m
    [33;21m2020-09-09 15:20:09,719 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "param_attr" to "weight_attr".[0m
    [33;21m2020-09-09 15:20:09,719 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 argument "use_cudnn" is removed.[0m
    [33;21m2020-09-09 15:20:09,720 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 argument "act" is removed.[0m
    [33;21m2020-09-09 15:20:09,722 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:62 variable "act" may not be visible here.[0m
    [38;21m2020-09-09 15:20:09,723 - utils.py:23 - INFO - models/dygraph/mnist/train.py:63 argument "dtype" not found.[0m
    [33;21m2020-09-09 15:20:09,725 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "input_dim" to "in_features".[0m
    [33;21m2020-09-09 15:20:09,726 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "output_dim" to "out_features".[0m
    [33;21m2020-09-09 15:20:09,727 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "param_attr" to "weight_attr".[0m
    [33;21m2020-09-09 15:20:09,728 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 argument "act" is removed.[0m
    [38;21m2020-09-09 15:20:09,729 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "dtype" not found.[0m
    [33;21m2020-09-09 15:20:09,731 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".[0m
    [38;21m2020-09-09 15:20:09,731 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,732 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.[0m
    [33;21m2020-09-09 15:20:09,733 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"[0m
    [33;21m2020-09-09 15:20:09,733 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"[0m
    [33;21m2020-09-09 15:20:09,735 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"[0m
    [33;21m2020-09-09 15:20:09,735 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".[0m
    [38;21m2020-09-09 15:20:09,736 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,736 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.[0m
    [33;21m2020-09-09 15:20:09,736 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"[0m
    [33;21m2020-09-09 15:20:09,737 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"[0m
    [33;21m2020-09-09 15:20:09,737 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"[0m
    [33;21m2020-09-09 15:20:09,739 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".[0m
    [38;21m2020-09-09 15:20:09,739 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,739 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.[0m
    [33;21m2020-09-09 15:20:09,740 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"[0m
    [33;21m2020-09-09 15:20:09,740 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"[0m
    [33;21m2020-09-09 15:20:09,741 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"[0m
    [33;21m2020-09-09 15:20:09,742 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:190 rename argument "learning_rate" to "learning_rate".[0m
    [38;21m2020-09-09 15:20:09,742 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "beta1" not found.[0m
    [38;21m2020-09-09 15:20:09,743 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "beta2" not found.[0m
    [38;21m2020-09-09 15:20:09,743 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "epsilon" not found.[0m
    [33;21m2020-09-09 15:20:09,744 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:190 rename argument "parameter_list" to "parameters".[0m
    [38;21m2020-09-09 15:20:09,744 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "regularization" not found.[0m
    [38;21m2020-09-09 15:20:09,745 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "grad_clip" not found.[0m
    [38;21m2020-09-09 15:20:09,745 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,746 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "lazy_mode" not found.[0m
    [33;21m2020-09-09 15:20:09,747 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".[0m
    [38;21m2020-09-09 15:20:09,748 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,748 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.[0m
    [33;21m2020-09-09 15:20:09,749 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"[0m
    [33;21m2020-09-09 15:20:09,750 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"[0m
    [33;21m2020-09-09 15:20:09,750 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"[0m
    [33;21m2020-09-09 15:20:09,751 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".[0m
    [38;21m2020-09-09 15:20:09,751 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.[0m
    [38;21m2020-09-09 15:20:09,752 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.[0m
    [33;21m2020-09-09 15:20:09,753 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"[0m
    [33;21m2020-09-09 15:20:09,753 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"[0m
    [33;21m2020-09-09 15:20:09,754 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"[0m
    [31m[1m--- models/dygraph/mnist/train.py[0m
    [32m[1m+++ models/dygraph/mnist/train.py[0m
    @@ -19,10 +19,6 @@
     from PIL import Image
     import os
     import paddle
    [31m-import paddle.fluid as fluid[0m
    [31m-from paddle.fluid.optimizer import AdamOptimizer[0m
    [31m-from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear[0m
    [31m-from paddle.fluid.dygraph.base import to_variable[0m
     
     
     def parse_args():
    @@ -39,7 +35,7 @@
         return args
     
     
    [31m-class SimpleImgConvPool(fluid.dygraph.Layer):[0m
    [32m+class SimpleImgConvPool(paddle.nn.Layer):[0m
         def __init__(self,
                      num_channels,
                      num_filters,
    @@ -59,20 +55,19 @@
                      bias_attr=None):
             super(SimpleImgConvPool, self).__init__()
     
    [31m-        self._conv2d = Conv2D([0m
    [31m-            num_channels=num_channels,[0m
    [31m-            num_filters=num_filters,[0m
    [31m-            filter_size=filter_size,[0m
    [32m+        self._conv2d = paddle.nn.Conv2d([0m
    [32m+            in_channels=num_channels,[0m
    [32m+            out_channels=num_filters,[0m
    [32m+            kernel_size=filter_size,[0m
                 stride=conv_stride,
                 padding=conv_padding,
                 dilation=conv_dilation,
                 groups=conv_groups,
    [31m-            param_attr=None,[0m
    [31m-            bias_attr=None,[0m
    [31m-            act=act,[0m
    [31m-            use_cudnn=use_cudnn)[0m
    [31m-[0m
    [31m-        self._pool2d = Pool2D([0m
    [32m+            weight_attr=None,[0m
    [32m+            bias_attr=None)[0m
    [32m+        self._act = act[0m
    [32m+[0m
    [32m+        self._pool2d = paddle.fluid.dygraph.nn.Pool2D([0m
                 pool_size=pool_size,
                 pool_type=pool_type,
                 pool_stride=pool_stride,
    @@ -82,11 +77,12 @@
     
         def forward(self, inputs):
             x = self._conv2d(inputs)
    [32m+        x = getattr(paddle.nn.functional, self._act)(x) if self._act else x[0m
             x = self._pool2d(x)
             return x
     
     
    [31m-class MNIST(fluid.dygraph.Layer):[0m
    [32m+class MNIST(paddle.nn.Layer):[0m
         def __init__(self):
             super(MNIST, self).__init__()
     
    @@ -99,19 +95,19 @@
             self.pool_2_shape = 50 * 4 * 4
             SIZE = 10
             scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
    [31m-        self._fc = Linear(self.pool_2_shape, 10,[0m
    [31m-                      param_attr=fluid.param_attr.ParamAttr([0m
    [31m-                          initializer=fluid.initializer.NormalInitializer([0m
    [31m-                              loc=0.0, scale=scale)),[0m
    [31m-                      act="softmax")[0m
    [32m+        self._fc = paddle.nn.Linear(in_features=self.pool_2_shape, out_features=10,[0m
    [32m+                      weight_attr=paddle.ParamAttr([0m
    [32m+                          initializer=paddle.nn.initializer.Normal([0m
    [32m+                              loc=0.0, scale=scale)))[0m
     
         def forward(self, inputs, label=None):
             x = self._simple_img_conv_pool_1(inputs)
             x = self._simple_img_conv_pool_2(x)
    [31m-        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])[0m
    [32m+        x = paddle.fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])[0m
             x = self._fc(x)
    [32m+        x = paddle.nn.functional.softmax(x)[0m
             if label is not None:
    [31m-            acc = fluid.layers.accuracy(input=x, label=label)[0m
    [32m+            acc = paddle.metric.accuracy(input=x, label=label)[0m
                 return x, acc
             else:
                 return x
    @@ -126,12 +122,12 @@
             y_data = np.array(
                 [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
     
    [31m-        img = to_variable(dy_x_data)[0m
    [31m-        label = to_variable(y_data)[0m
    [32m+        img = paddle.to_tensor(data=dy_x_data, dtype=None, place=None, stop_gradient=True)[0m
    [32m+        label = paddle.to_tensor(data=y_data, dtype=None, place=None, stop_gradient=True)[0m
             label.stop_gradient = True
             prediction, acc = model(img, label)
    [31m-        loss = fluid.layers.cross_entropy(input=prediction, label=label)[0m
    [31m-        avg_loss = fluid.layers.mean(loss)[0m
    [32m+        loss = paddle.fluid.layers.cross_entropy(input=prediction, label=label)[0m
    [32m+        avg_loss = paddle.mean(loss)[0m
             acc_set.append(float(acc.numpy()))
             avg_loss_set.append(float(avg_loss.numpy()))
     
    @@ -143,111 +139,113 @@
     
     
     def inference_mnist():
    [31m-    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \[0m
    [31m-        if args.use_data_parallel else fluid.CUDAPlace(0)[0m
    [31m-    with fluid.dygraph.guard(place):[0m
    [31m-        mnist_infer = MNIST()[0m
    [32m+    place = paddle.CUDAPlace(paddle.fluid.dygraph.parallel.Env().dev_id) \[0m
    [32m+        if args.use_data_parallel else paddle.CUDAPlace(0)[0m
    [32m+    paddle.disable_static(place)[0m
    [32m+    mnist_infer = MNIST()[0m
             # load checkpoint
    [31m-        model_dict, _ = fluid.load_dygraph("save_temp")[0m
    [31m-        mnist_infer.set_dict(model_dict)[0m
    [31m-        print("checkpoint loaded")[0m
    [32m+    model_dict, _ = paddle.fluid.load_dygraph("save_temp")[0m
    [32m+    mnist_infer.set_dict(model_dict)[0m
    [32m+    print("checkpoint loaded")[0m
     
             # start evaluate mode
    [31m-        mnist_infer.eval()[0m
    [31m-[0m
    [31m-        def load_image(file):[0m
    [31m-            im = Image.open(file).convert('L')[0m
    [31m-            im = im.resize((28, 28), Image.ANTIALIAS)[0m
    [31m-            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)[0m
    [31m-            im = im / 255.0 * 2.0 - 1.0[0m
    [31m-            return im[0m
    [31m-[0m
    [31m-        cur_dir = os.path.dirname(os.path.realpath(__file__))[0m
    [31m-        tensor_img = load_image(cur_dir + '/image/infer_3.png')[0m
    [31m-[0m
    [31m-        results = mnist_infer(to_variable(tensor_img))[0m
    [31m-        lab = np.argsort(results.numpy())[0m
    [31m-        print("Inference result of image/infer_3.png is: %d" % lab[0][-1])[0m
    [32m+    mnist_infer.eval()[0m
    [32m+[0m
    [32m+    def load_image(file):[0m
    [32m+        im = Image.open(file).convert('L')[0m
    [32m+        im = im.resize((28, 28), Image.ANTIALIAS)[0m
    [32m+        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)[0m
    [32m+        im = im / 255.0 * 2.0 - 1.0[0m
    [32m+        return im[0m
    [32m+[0m
    [32m+    cur_dir = os.path.dirname(os.path.realpath(__file__))[0m
    [32m+    tensor_img = load_image(cur_dir + '/image/infer_3.png')[0m
    [32m+[0m
    [32m+    results = mnist_infer(paddle.to_tensor(data=tensor_img, dtype=None, place=None, stop_gradient=True))[0m
    [32m+    lab = np.argsort(results.numpy())[0m
    [32m+    print("Inference result of image/infer_3.png is: %d" % lab[0][-1])[0m
    [32m+    paddle.enable_static()[0m
     
     
     def train_mnist(args):
         epoch_num = args.epoch
         BATCH_SIZE = 64
     
    [31m-    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \[0m
    [31m-        if args.use_data_parallel else fluid.CUDAPlace(0)[0m
    [31m-    with fluid.dygraph.guard(place):[0m
    [32m+    place = paddle.CUDAPlace(paddle.fluid.dygraph.parallel.Env().dev_id) \[0m
    [32m+        if args.use_data_parallel else paddle.CUDAPlace(0)[0m
    [32m+    paddle.disable_static(place)[0m
    [32m+    if args.ce:[0m
    [32m+        print("ce mode")[0m
    [32m+        seed = 33[0m
    [32m+        np.random.seed(seed)[0m
    [32m+        paddle.static.default_startup_program().random_seed = seed[0m
    [32m+        paddle.static.default_main_program().random_seed = seed[0m
    [32m+[0m
    [32m+    if args.use_data_parallel:[0m
    [32m+        strategy = paddle.fluid.dygraph.parallel.prepare_context()[0m
    [32m+    mnist = MNIST()[0m
    [32m+    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=mnist.parameters())[0m
    [32m+    if args.use_data_parallel:[0m
    [32m+        mnist = paddle.fluid.dygraph.parallel.DataParallel(mnist, strategy)[0m
    [32m+[0m
    [32m+    train_reader = paddle.batch([0m
    [32m+        paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)[0m
    [32m+    if args.use_data_parallel:[0m
    [32m+        train_reader = paddle.fluid.contrib.reader.distributed_batch_reader([0m
    [32m+            train_reader)[0m
    [32m+[0m
    [32m+    test_reader = paddle.batch([0m
    [32m+        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)[0m
    [32m+[0m
    [32m+    for epoch in range(epoch_num):[0m
    [32m+        for batch_id, data in enumerate(train_reader()):[0m
    [32m+            dy_x_data = np.array([x[0].reshape(1, 28, 28)[0m
    [32m+                                  for x in data]).astype('float32')[0m
    [32m+            y_data = np.array([0m
    [32m+                [x[1] for x in data]).astype('int64').reshape(-1, 1)[0m
    [32m+[0m
    [32m+            img = paddle.to_tensor(data=dy_x_data, dtype=None, place=None, stop_gradient=True)[0m
    [32m+            label = paddle.to_tensor(data=y_data, dtype=None, place=None, stop_gradient=True)[0m
    [32m+            label.stop_gradient = True[0m
    [32m+[0m
    [32m+            cost, acc = mnist(img, label)[0m
    [32m+[0m
    [32m+            loss = paddle.fluid.layers.cross_entropy(cost, label)[0m
    [32m+            avg_loss = paddle.mean(loss)[0m
    [32m+[0m
    [32m+            if args.use_data_parallel:[0m
    [32m+                avg_loss = mnist.scale_loss(avg_loss)[0m
    [32m+                avg_loss.backward()[0m
    [32m+                mnist.apply_collective_grads()[0m
    [32m+            else:[0m
    [32m+                avg_loss.backward()[0m
    [32m+[0m
    [32m+            adam.minimize(avg_loss)[0m
    [32m+                # save checkpoint[0m
    [32m+            mnist.clear_gradients()[0m
    [32m+            if batch_id % 100 == 0:[0m
    [32m+                print("Loss at epoch {} step {}: {:}".format([0m
    [32m+                    epoch, batch_id, avg_loss.numpy()))[0m
    [32m+[0m
    [32m+        mnist.eval()[0m
    [32m+        test_cost, test_acc = test_mnist(test_reader, mnist, BATCH_SIZE)[0m
    [32m+        mnist.train()[0m
             if args.ce:
    [31m-            print("ce mode")[0m
    [31m-            seed = 33[0m
    [31m-            np.random.seed(seed)[0m
    [31m-            fluid.default_startup_program().random_seed = seed[0m
    [31m-            fluid.default_main_program().random_seed = seed[0m
    [31m-[0m
    [31m-        if args.use_data_parallel:[0m
    [31m-            strategy = fluid.dygraph.parallel.prepare_context()[0m
    [31m-        mnist = MNIST()[0m
    [31m-        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())[0m
    [31m-        if args.use_data_parallel:[0m
    [31m-            mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)[0m
    [31m-[0m
    [31m-        train_reader = paddle.batch([0m
    [31m-            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)[0m
    [31m-        if args.use_data_parallel:[0m
    [31m-            train_reader = fluid.contrib.reader.distributed_batch_reader([0m
    [31m-                train_reader)[0m
    [31m-[0m
    [31m-        test_reader = paddle.batch([0m
    [31m-            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)[0m
    [31m-[0m
    [31m-        for epoch in range(epoch_num):[0m
    [31m-            for batch_id, data in enumerate(train_reader()):[0m
    [31m-                dy_x_data = np.array([x[0].reshape(1, 28, 28)[0m
    [31m-                                      for x in data]).astype('float32')[0m
    [31m-                y_data = np.array([0m
    [31m-                    [x[1] for x in data]).astype('int64').reshape(-1, 1)[0m
    [31m-[0m
    [31m-                img = to_variable(dy_x_data)[0m
    [31m-                label = to_variable(y_data)[0m
    [31m-                label.stop_gradient = True[0m
    [31m-[0m
    [31m-                cost, acc = mnist(img, label)[0m
    [31m-[0m
    [31m-                loss = fluid.layers.cross_entropy(cost, label)[0m
    [31m-                avg_loss = fluid.layers.mean(loss)[0m
    [31m-[0m
    [31m-                if args.use_data_parallel:[0m
    [31m-                    avg_loss = mnist.scale_loss(avg_loss)[0m
    [31m-                    avg_loss.backward()[0m
    [31m-                    mnist.apply_collective_grads()[0m
    [31m-                else:[0m
    [31m-                    avg_loss.backward()[0m
    [31m-[0m
    [31m-                adam.minimize(avg_loss)[0m
    [31m-                # save checkpoint[0m
    [31m-                mnist.clear_gradients()[0m
    [31m-                if batch_id % 100 == 0:[0m
    [31m-                    print("Loss at epoch {} step {}: {:}".format([0m
    [31m-                        epoch, batch_id, avg_loss.numpy()))[0m
    [31m-[0m
    [31m-            mnist.eval()[0m
    [31m-            test_cost, test_acc = test_mnist(test_reader, mnist, BATCH_SIZE)[0m
    [31m-            mnist.train()[0m
    [31m-            if args.ce:[0m
    [31m-                print("kpis\ttest_acc\t%s" % test_acc)[0m
    [31m-                print("kpis\ttest_cost\t%s" % test_cost)[0m
    [31m-            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format([0m
    [31m-                epoch, test_cost, test_acc))[0m
    [31m-[0m
    [31m-        save_parameters = (not args.use_data_parallel) or ([0m
    [31m-            args.use_data_parallel and[0m
    [31m-            fluid.dygraph.parallel.Env().local_rank == 0)[0m
    [31m-        if save_parameters:[0m
    [31m-            fluid.save_dygraph(mnist.state_dict(), "save_temp")[0m
    [32m+            print("kpis\ttest_acc\t%s" % test_acc)[0m
    [32m+            print("kpis\ttest_cost\t%s" % test_cost)[0m
    [32m+        print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format([0m
    [32m+            epoch, test_cost, test_acc))[0m
    [32m+[0m
    [32m+    save_parameters = (not args.use_data_parallel) or ([0m
    [32m+        args.use_data_parallel and[0m
    [32m+        paddle.fluid.dygraph.parallel.Env().local_rank == 0)[0m
    [32m+    if save_parameters:[0m
    [32m+        paddle.fluid.save_dygraph(mnist.state_dict(), "save_temp")[0m
                 
    [31m-            print("checkpoint saved")[0m
    [31m-[0m
    [31m-            inference_mnist()[0m
    [32m+        print("checkpoint saved")[0m
    [32m+[0m
    [32m+        inference_mnist()[0m
    [32m+    paddle.enable_static()[0m
     
     
     if __name__ == '__main__':
    [33;21m2020-09-09 15:20:09,886 - main.py:80 - WARNING - Refactor finished without touching source files, add "--write" to modify source files in-place if everything is ok.[0m


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


.. parsed-literal::

    2020-09-09 15:02:54 - utils.py:341 - ERROR - /path/to/model.py doesn't exist.
    2020-09-09 15:02:54 - main.py:52 - ERROR - convert abort!
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:22 remove "import paddle.fluid as fluid"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:23 remove "from paddle.fluid.optimizer import AdamOptimizer"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:24 remove "from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:25 remove "from paddle.fluid.dygraph.base import to_variable"
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:42 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:62 Conv2D -> paddle.fluid.dygraph.nn.Conv2D
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:75 Pool2D -> paddle.fluid.dygraph.nn.Pool2D
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:89 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:102 Linear -> paddle.fluid.dygraph.nn.Linear
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:103 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:104 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:111 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:114 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:129 to_variable -> paddle.fluid.dygraph.base.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:130 to_variable -> paddle.fluid.dygraph.base.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:133 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:134 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:146 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:146 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:147 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:148 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:151 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:168 to_variable -> paddle.fluid.dygraph.base.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:177 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:177 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:178 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:179 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:184 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:185 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:188 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 AdamOptimizer -> paddle.fluid.optimizer.AdamOptimizer
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:192 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:197 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:210 to_variable -> paddle.fluid.dygraph.base.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:211 to_variable -> paddle.fluid.dygraph.base.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:216 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:217 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:244 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:246 fluid -> paddle.fluid
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.nn.Conv2D -> paddle.fluid.dygraph.Conv2D
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.initializer.NormalInitializer -> paddle.fluid.initializer.Normal
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.param_attr.ParamAttr -> paddle.fluid.ParamAttr
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.nn.Linear -> paddle.fluid.dygraph.Linear
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.optimizer.AdamOptimizer -> paddle.fluid.optimizer.Adam
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:1 paddle.fluid.dygraph.base.to_variable -> paddle.fluid.dygraph.to_variable
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:63 args_list: "['num_channels', 'num_filters', 'filter_size', 'stride', 'padding', 'dilation', 'groups', 'param_attr', 'bias_attr', 'use_cudnn', 'act', 'dtype']" is longer than positional arguments, redundant arguments will be skipped.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:103 args_list: "['input_dim', 'output_dim', 'param_attr', 'bias_attr', 'act', 'dtype']" is longer than positional arguments, redundant arguments will be skipped.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 args_list: "['learning_rate', 'beta1', 'beta2', 'epsilon', 'parameter_list', 'regularization', 'grad_clip', 'name', 'lazy_mode']" is longer than positional arguments, redundant arguments will be skipped.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "num_channels" to "in_channels".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "num_filters" to "out_channels".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "filter_size" to "kernel_size".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 rename argument "param_attr" to "weight_attr".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 argument "use_cudnn" is removed.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:63 argument "act" is removed.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:62 variable "act" may not be visible here.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:63 argument "dtype" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "input_dim" to "in_features".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "output_dim" to "out_features".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "param_attr" to "weight_attr".
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 argument "act" is removed.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "dtype" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:190 rename argument "learning_rate" to "learning_rate".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "beta1" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "beta2" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "epsilon" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:190 rename argument "parameter_list" to "parameters".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "regularization" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "grad_clip" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:190 argument "lazy_mode" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 rename argument "value" to "data".
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "name" not found.
    2020-09-09 15:20:09 - utils.py:23 - INFO - models/dygraph/mnist/train.py:0 argument "zero_copy" not found.
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "dtype=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "place=None"
    2020-09-09 15:20:09 - utils.py:27 - WARNING - models/dygraph/mnist/train.py:0 add argument "stop_gradient=True"
    2020-09-09 15:20:09 - main.py:80 - WARNING - Refactor finished without touching source files, add "--write" to modify source files in-place if everything is ok.


注意事项
~~~~~~~~

-  本迁移工具不能完成所有API的迁移，有少量的API需要您手动完成迁移，具体信息可见WARNING。

使用Paddle 2
~~~~~~~~~~~~

完成迁移后，代码就从Paddle 1迁移到了Paddle 2，您就可以在Paddle
2下进行相关的开发。
