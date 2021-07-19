.. _cn_guides_migration:

版本迁移工具
====================

在飞桨框架2.0中，Paddle API的位置、命名、参数、行为，进行了系统性的调整和规范, 将API体系从1.X版本的 ``paddle.fluid.*`` 迁移到了 ``paddle.*`` 下。``paddle.fluid`` 目录下暂时保留了1.8版本API，主要是兼容性考虑，未来会被删除。

使用版本迁移工具自动迁移Paddle 1.X的代码到Paddle 2.0
------------------------------------

飞桨提供了版本迁移工具，该工具按 Paddle 2.0 对于 Paddle 1.X的变化，能够自动实现以下功能：

- 按照 :ref:`API映射表 <cn_guides_api_mapping>` ，将转换工具能否转换这列为True的API由Paddle 1.X 转为 Paddle 2.0，为False的API打印WARNING，提示手动升级。
- 因为Paddle 2.0.0 默认开启动态图，所以删除用于开启动态图上下文的 ``with paddle.fluid.dygraph.guard(place)`` ，并修改该上下文的代码缩进；
- 删除组网API中的 ``act`` 参数，并自动添加相关的激活函数；

目前，版本迁移工具能够处理的API数量为X个，如果你有代码迁移的需求，使用转换工具能够节省你部分时间，帮助你快速完成代码迁移。

.. warning::

    版本迁移工具并不能处理所有的情况，对于API的处理只能按照 :ref:`API映射表 <cn_guides_api_mapping>` 中的关系完成API的变化。如代码中包含有转换工具能否转换这列为False的API或不在此表中的API，在使用本工具后，仍然需要手工来进行检查并做相应的调整。

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

这将在命令行中，以\ ``diff``\ 的形式，展示model.py从Paddle 1.x转换为Paddle 2.0的变化。如果你确认上述变化没有问题，只需要再执行：

.. code:: ipython3

    $ paddle_upgrade_tool --inpath /path/to/model.py --write

就会原地改写model.py，将上述变化改写到你的源文件中。
注意：版本转换工具会默认备份源文件，到~/.paddle_upgrade_tool/下。

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

在使用paddle_upgrade_tool前，需要确保已经安装了Paddle 2.0.0版本。

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
    remote: Enumerating objects: 8, done.
    remote: Counting objects: 100% (8/8), done.
    remote: Compressing objects: 100% (8/8), done.
    remote: Total 35011 (delta 1), reused 0 (delta 0), pack-reused 35003
    Receiving objects: 100% (35011/35011), 356.97 MiB | 1.53 MiB/s, done.
    Resolving deltas: 100% (23291/23291), done.


查看帮助文档
^^^^^^^^^^^^

你可以直接通过下面的方式，查看帮助文档。

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

paddle_upgrade_tool支持单文件的转化，你可以通过下方的命令直接转化单独的文件。

.. code:: ipython3

    $ paddle_upgrade_tool --inpath models/dygraph/mnist/train.py

注意，对于参数的删除及一些特殊情况，迁移工具都会打印WARNING信息，需要你仔细核对相关内容。
如果你觉得上述信息没有问题，可以直接对文件进行原地修改，方式如下：

.. code:: ipython3

    $ paddle_upgrade_tool --inpath models/dygraph/mnist/train.py --write 

此时，命令行会弹出下方的提示：

.. code:: ipython3

    "models/dygraph/mnist/train.py" will be modified in-place, and it has been backed up to "~/.paddle_upgrade_tool/train.py_backup_2020_09_09_20_35_15_037821". Do you want to continue? [Y/n]:

输入\ ``y``
后即开始执行代码迁移。为了高效完成迁移，工具这里采用了原地写入的方式。此外，为了防止特殊情况，工具会备份转换前的代码到
``~/.paddle_upgrade_tool`` 目录下，如果需要，你可以在备份目录下找到转换前的代码。

代码迁移完成后，会生成一个report.log文件，记录了迁移的详情。内容如下：

.. code:: ipython3

    $ cat report.log

注意事项
~~~~~~~~

-  本迁移工具不能完成所有API的迁移，有少量的API需要你手动完成迁移，具体信息可见WARNING。

使用Paddle 2.0
~~~~~~~~~~~~~~~~

完成迁移后，代码就从Paddle 1.x迁移到了Paddle 2.0，你就可以在Paddle 2.0下进行相关的开发。



旧保存格式兼容载入
~~~~~~~~~~~~~~~

如果你是从飞桨框架1.x切换到2.1，曾经使用飞桨框架1.x的fluid相关接口保存模型或者参数，飞桨框架2.1也对这种情况进行了兼容性支持，包括以下几种情况。

飞桨1.x模型准备及训练示例，该示例为后续所有示例的前序逻辑：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid
    import paddle.nn as nn
    import paddle.optimizer as opt

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    # enable static mode
    paddle.enable_static()

    # define a random dataset
    class RandomDataset(paddle.io.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    image = fluid.data(name='image', shape=[None, 784], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    pred = fluid.layers.fc(input=image, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=pred, label=label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        feed_list=[image, label],
        places=place,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train model
    for data in loader():
        exe.run(
            fluid.default_main_program(),
            feed=data, 
            fetch_list=[avg_loss])


1 从 ``paddle.fluid.io.save_inference_model`` 保存结果中载入模型&参数
------------------------------------------------------------------

(1) 同时载入模型和参数

使用 ``paddle.jit.load`` 配合 ``**configs`` 载入模型和参数。

如果你是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

.. code-block:: python

    # save default
    model_path = "fc.example.model"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe)

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path)

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

如果你指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    # save with model_filename
    model_path = "fc.example.model.with_model_filename"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe, model_filename="__simplenet__")

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path, model_filename="__simplenet__")

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

如果你指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    # save with params_filename
    model_path = "fc.example.model.with_params_filename"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe, params_filename="__params__")

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path, params_filename="__params__")

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

(2) 仅载入参数

如果你仅需要从 ``paddle.fluid.io.save_inference_model`` 的存储结果中载入参数，以state_dict的形式配置到已有代码的模型中，可以使用 ``paddle.load`` 配合 ``**configs`` 载入。

如果你是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model"

    load_param_dict = paddle.load(model_path)

如果你指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_model_filename"

    load_param_dict = paddle.load(model_path, model_filename="__simplenet__")

如果你指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_params_filename"

    load_param_dict = paddle.load(model_path, params_filename="__params__")

.. note::
    一般预测模型不会存储优化器Optimizer的参数，因此此处载入的仅包括模型本身的参数。

.. note::
    由于 ``structured_name`` 是动态图下独有的变量命名方式，因此从静态图存储结果载入的state_dict在配置到动态图的Layer中时，需要配置 ``Layer.set_state_dict(use_structured_name=False)`` 。


2 从 ``paddle.fluid.save`` 存储结果中载入参数
----------------------------------------------

 ``paddle.fluid.save`` 的存储格式与2.x动态图接口 ``paddle.save`` 存储格式是类似的，同样存储了dict格式的参数，因此可以直接使用 ``paddle.load`` 载入state_dict，但需要注意不能仅传入保存的路径，而要传入保存参数的文件名，示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.save
    model_path = "fc.example.model.save"
    program = fluid.default_main_program()
    fluid.save(program, model_path)

    # enable dynamic mode
    paddle.disable_static(place)

    load_param_dict = paddle.load("fc.example.model.save.pdparams")


.. note::
    由于 ``paddle.fluid.save`` 接口原先在静态图模式下的定位是存储训练时参数，或者说存储Checkpoint，故尽管其同时存储了模型结构，目前也暂不支持从 ``paddle.fluid.save`` 的存储结果中同时载入模型和参数，后续如有需求再考虑支持。


3 从 ``paddle.fluid.io.save_params/save_persistables`` 保存结果中载入参数
-------------------------------------------------------------------------

这两个接口在飞桨1.x版本时，已经不再推荐作为存储模型参数的接口使用，故并未继承至飞桨2.x，之后也不会再推荐使用这两个接口存储参数。

对于使用这两个接口存储参数兼容载入的支持，分为两种情况，下面以 ``paddle.fluid.io.save_params`` 接口为例介绍相关使用方法：

(1) 使用默认方式存储，各参数分散存储为单独的文件，文件名为参数名

这种存储方式仍然可以使用 ``paddle.load`` 接口兼容载入，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params"
    fluid.io.save_params(exe, model_path)

    # load 
    state_dict = paddle.load(model_path)
    print(state_dict)

(2) 指定了参数存储的文件，将所有参数存储至单个文件中

将所有参数存储至单个文件中会导致存储结果中丢失Tensor名和Tensor数据之间的映射关系，因此这部分丢失的信息需要用户传入进行补足。为了确保正确性，这里不仅要传入Tensor的name列表，同时要传入Tensor的shape和dtype等描述信息，通过检查和存储数据的匹配性确保严格的正确性，这导致载入数据的恢复过程变得比较复杂，仍然需要一些飞桨1.x的概念支持。后续如果此项需求较为普遍，飞桨将会考虑将该项功能兼容支持到 ``paddle.load`` 中，但由于信息丢失而导致的使用复杂性仍然是存在的，因此建议你避免仅使用这两个接口存储参数。

目前暂时推荐你使用 ``paddle.static.load_program_state`` 接口解决此处的载入问题，需要获取原Program中的参数列表传入该方法，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params_with_filename"
    fluid.io.save_params(exe, model_path, filename="__params__")

    # load 
    import os
    params_file_path = os.path.join(model_path, "__params__")
    var_list = fluid.default_main_program().all_parameters()
    state_dict = paddle.io.load_program_state(params_file_path, var_list)


4 从 ``paddle.static.save`` 保存结果中载入参数
-------------------------------------------
``paddle.static.save`` 接口生成三个文件： ``*.pdparams' 、``*.pdopt`` 、``*.pdmodel``，分别保存了组网的参数、优化器的参数、静态图的Program。推荐您使用 ``paddle.load`` 分别加载这三个文件，然后使用 ``set_state_dict`` 接口将参数设置到 ``Program`` 中 。如果您已经在代码中定义了 ``Program`` ，您可以不加载 ``*.pdmodel`` 文件；如果您不需要恢复优化器中的参数，您可以不加载 ``*.pdopt`` 文件。使用示例如下：


.. code-block:: python

    import os
    import paddle

    paddle.enable_static()
    x = paddle.static.data(
                    name="static_x", shape=[None, 224], dtype='float32')
    z = paddle.static.nn.fc(x, 10)
    z = paddle.static.nn.fc(z, 10, bias_attr=False)

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    prog = paddle.static.default_main_program()

    path = os.path.join("test_static_save_load", "model")
    paddle.static.save(prog, path)

    # load program 
    program=paddle.load(path + '.pdmodel')

    state_dict_param = paddle.load(path + '.pdparams')
    program.set_state_dict(state_dict_param)

    state_dict_opt = paddle.load(path + '.pdopt')
    program.set_state_dict(state_dict_opt)