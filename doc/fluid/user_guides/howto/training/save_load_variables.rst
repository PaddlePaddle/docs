.. _user_guide_save_load_vars:

################################
模型/变量的保存、载入与增量训练
################################

模型变量分类
############

在PaddlePaddle Fluid中，所有的模型变量都用 :code:`fluid.framework.Variable()` 作为基类。
在该基类之下，模型变量主要可以分为以下几种类别：

1. 模型参数
  模型参数是深度学习模型中被训练和学习的变量，在训练过程中，训练框架根据反向传播(backpropagation)算法计算出每一个模型参数当前的梯度，
  并用优化器(optimizer)根据梯度对参数进行更新。模型的训练过程本质上可以看做是模型参数不断迭代更新的过程。
  在PaddlePaddle Fluid中，模型参数用 :code:`fluid.framework.Parameter` 来表示，
  这是一个 :code:`fluid.framework.Variable()` 的派生类，除了具有 :code:`fluid.framework.Variable()` 的各项性质以外，
  :code:`fluid.framework.Parameter` 还可以配置自身的初始化方法、更新率等属性。

2. 长期变量
  长期变量指的是在整个训练过程中持续存在、不会因为一个迭代的结束而被销毁的变量，例如动态调节的全局学习率等。
  在PaddlePaddle Fluid中，长期变量通过将 :code:`fluid.framework.Variable()` 的 :code:`persistable`
  属性设置为 :code:`True` 来表示。所有的模型参数都是长期变量，但并非所有的长期变量都是模型参数。

3. 临时变量
  不属于上面两个类别的所有模型变量都是临时变量，这种类型的变量只在一个训练迭代中存在，在每一个迭代结束后，
  所有的临时变量都会被销毁，然后在下一个迭代开始之前，又会先构造出新的临时变量供本轮迭代使用。
  一般情况下模型中的大部分变量都属于这一类别，例如输入的训练数据、一个普通的layer的输出等等。



如何保存模型变量
################

根据用途的不同，我们需要保存的模型变量也是不同的。例如，如果我们只是想保存模型用来进行以后的预测，
那么只保存模型参数就够用了。但如果我们需要保存一个checkpoint（检查点，类似于存档，存有复现目前模型的必要信息）以备将来恢复训练，
那么我们应该将各种长期变量都保存下来，甚至还需要记录一下当前的epoch和step的id。
因为一些模型变量虽然不是参数，但对于模型的训练依然必不可少。

save_vars、save_params、save_persistables 以及 save_inference_model的区别
##########################################################################
1. :code:`save_inference_model` 会根据用户配置的 :code:`feeded_var_names` 和 :code:`target_vars` 进行网络裁剪，保存下裁剪后的网络结构的 ``__model__`` 以及裁剪后网络中的长期变量

2. :code:`save_persistables` 不会保存网络结构，会保存网络中的全部长期变量到指定位置。

3. :code:`save_params` 不会保存网络结构，会保存网络中的全部模型参数到指定位置。

4. :code:`save_vars` 不会保存网络结构，会根据用户指定的 :code:`fluid.framework.Parameter` 列表进行保存。

 :code:`save_persistables` 保存的网络参数是最全面的，如果是增量训练或者恢复训练， 请选择 :code:`save_persistables` 进行变量保存。
 :code:`save_inference_model` 会保存网络参数及裁剪后的模型，如果后续要做预测相关的工作， 请选择 :code:`save_inference_model` 进行变量和网络的保存。
 :code:`save_vars 和 save_params` 仅在用户了解清楚用途及特殊目的情况下使用， 一般不建议使用。


保存模型用于对新样本的预测
==========================

如果我们保存模型的目的是用于对新样本的预测，那么只保存模型参数就足够了。我们可以使用
:code:`fluid.io.save_params()` 接口来进行模型参数的保存。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path, main_program=None)

上面的例子中，通过调用 :code:`fluid.io.save_params` 函数，PaddlePaddle Fluid会对默认
:code:`fluid.Program` 也就是 :code:`prog` 中的所有模型变量进行扫描，
筛选出其中所有的模型参数，并将这些模型参数保存到指定的 :code:`param_path` 之中。



如何载入模型变量
################

与模型变量的保存相对应，我们提供了两套API来分别载入模型的参数和载入模型的长期变量，分别为保存、加载模型参数的 ``save_params()`` 、 ``load_params()`` 和
保存、加载长期变量的 ``save_persistables`` 、 ``load_persistables`` 。

载入模型用于对新样本的预测
==========================

对于通过 :code:`fluid.io.save_params` 保存的模型，可以使用 :code:`fluid.io.load_params`
来进行载入。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                         main_program=prog)

上面的例子中，通过调用 :code:`fluid.io.load_params` 函数，PaddlePaddle Fluid会对
:code:`prog` 中的所有模型变量进行扫描，筛选出其中所有的模型参数，
并尝试从 :code:`param_path` 之中读取加载它们。

需要格外注意的是，这里的 :code:`prog` 必须和调用 :code:`fluid.io.save_params`
时所用的 :code:`prog` 中的前向部分完全一致，且不能包含任何参数更新的操作。如果两者存在不一致，
那么可能会导致一些变量未被正确加载；如果错误地包含了参数更新操作，那可能会导致正常预测过程中参数被更改。
这两个 :code:`fluid.Program` 之间的关系类似于训练 :code:`fluid.Program`
和测试 :code:`fluid.Program` 之间的关系，详见： :ref:`user_guide_test_while_training`。

另外，需特别注意运行 :code:`fluid.default_startup_program()` 必须在调用 :code:`fluid.io.load_params`
之前。如果在之后运行，可能会覆盖已加载的模型参数导致错误。

预测模型的保存和加载
##############################

预测引擎提供了存储预测模型 :code:`fluid.io.save_inference_model` 和加载预测模型 :code:`fluid.io.load_inference_model` 两个接口。

- :code:`fluid.io.save_inference_model`：请参考  :ref:`api_guide_inference`。
- :code:`fluid.io.load_inference_model`：请参考  :ref:`api_guide_inference`。



增量训练
############

增量训练指一个学习系统能不断地从新样本中学习新的知识，并能保存大部分以前已经学习到的知识。因此增量学习涉及到两点：在上一次训练结束的时候保存需要的长期变量， 在下一次训练开始的时候加载上一次保存的这些长期变量。 因此增量训练涉及到如下几个API:
:code:`fluid.io.save_persistables`、:code:`fluid.io.load_persistables` 。

单机增量训练
==========================
单机的增量训练的一般步骤如下：

1. 在训练的最后调用 :code:`fluid.io.save_persistables` 保存持久性参数到指定的位置。
2. 在训练的startup_program通过执行器 :code:`Executor` 执行成功之后调用 :code:`fluid.io.load_persistables` 加载之前保存的持久性参数。
3. 通过执行器 :code:`Executor` 或者 :code:`ParallelExecutor` 继续训练。


例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    prog = fluid.default_main_program()
    fluid.io.save_persistables(exe, path, prog)

上面的例子中，通过调用 :code:`fluid.io.save_persistables` 函数，PaddlePaddle Fluid会从默认 :code:`fluid.Program` 也就是 :code:`prog` 的所有模型变量中找出长期变量，并将他们保存到指定的 :code:`path` 目录下。


.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    startup_prog = fluid.default_startup_program()
    exe.run(startup_prog)
    fluid.io.load_persistables(exe, path, startup_prog)
    main_prog = fluid.default_main_program()
    exe.run(main_prog)

上面的例子中，通过调用 :code:`fluid.io.load_persistables` 函数，PaddlePaddle Fluid会从默认
:code:`fluid.Program` 也就是 :code:`prog` 的所有模型变量中找出长期变量，从指定的 :code:`path` 目录中将它们一一加载， 然后再继续进行训练。



多机增量（不带分布式大规模稀疏矩阵）训练的一般步骤为
==========================

多机增量训练和单机增量训练有若干不同点：

1. 在训练的最后调用 :code:`fluid.io.save_persistables` 保存长期变量时，不必要所有的trainer都调用这个方法来保存，一般0号trainer来保存即可。
2. 多机增量训练的参数加载在PServer端，trainer端不用加载参数。在PServer全部启动后，trainer会从PServer端同步参数。
3. 在确认需要使用增量的情况下， 多机在调用 :code:`fluid.DistributeTranspiler.transpile` 时需要指定 ``current_endpoint`` 参数。

多机增量（不带分布式大规模稀疏矩阵）训练的一般步骤为：

1. 0号trainer在训练的最后调用 :code:`fluid.io.save_persistables` 保存持久性参数到指定的 :code:`path` 下。
2. 通过HDFS等方式将0号trainer保存下来的所有的参数共享给所有的PServer(每个PServer都需要有完整的参数)。
3. PServer在训练的startup_program通过执行器（:code:`Executor`）执行成功之后调用 :code:`fluid.io.load_persistables` 加载0号trainer保存的持久性参数。
4. PServer通过执行器 :code:`Executor` 继续启动PServer_program.
5. 所有的训练节点trainer通过执行器 :code:`Executor` 或者 :code:`ParallelExecutor` 正常训练。


对于训练过程中待保存参数的trainer， 例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    trainer_id = 0
    if trainer_id == 0:
        prog = fluid.default_main_program()
        fluid.io.save_persistables(exe, path, prog)


.. code-block:: bash
    hadoop fs -mkdir /remote/$path
    hadoop fs -put $path /remote/$path

上面的例子中，0号trainer通过调用 :code:`fluid.io.save_persistables` 函数，PaddlePaddle Fluid会从默认
:code:`fluid.Program` 也就是 :code:`prog` 的所有模型变量中找出长期变量，并将他们保存到指定的 :code:`path` 目录下。然后通过调用第三方的文件系统（如HDFS）将存储的模型进行上传到所有PServer都可访问的位置。

对于训练过程中待载入参数的PServer， 例如：


.. code-block:: bash
    hadoop fs -get /remote/$path $path


.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./models"
    pserver_endpoints = "127.0.0.1:1001,127.0.0.1:1002"
    trainers = 4
    training_role == "PSERVER"
    config = fluid.DistributeTranspilerConfig()
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers, sync_mode=True, current_endpoint=current_endpoint)

    if training_role == "PSERVER":
        current_endpoint = "127.0.0.1:1001"
        pserver_prog = t.get_pserver_program(current_endpoint)
        pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)

        exe.run(pserver_startup)
        fluid.io.load_persistables(exe, path, pserver_startup)
        exe.run(pserver_prog)
    if training_role == "TRAINER":
        main_program = t.get_trainer_program()
                exe.run(main_program)

上面的例子中，每个PServer通过调用HDFS的命令获取到0号trainer保存的参数，通过配置获取到PServer的 :code:`fluid.Program` ，PaddlePaddle Fluid会从此
:code:`fluid.Program` 也就是 :code:`pserver_startup` 的所有模型变量中找出长期变量，并通过指定的 :code:`path` 目录下一一加载。


