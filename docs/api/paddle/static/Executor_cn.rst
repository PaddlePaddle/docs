.. _cn_api_paddle_static_Executor:

Executor
-------------------------------



.. py:class:: paddle.static.Executor (place=None)




Executor 支持单 GPU、多 GPU 以及 CPU 运行。

参数
::::::::::::

    - **place** (paddle.CPUPlace()|paddle.CUDAPlace(N)|None) – 该参数表示 Executor 执行所在的设备，这里的 N 为 GPU 对应的 ID。当该参数为 `None` 时，PaddlePaddle 会根据其安装版本设置默认的运行设备。当安装的 Paddle 为 CPU 版时，默认运行设置会设置成 `CPUPlace()`，而当 Paddle 为 GPU 版时，默认运行设备会设置成 `CUDAPlace(0)`。默认值为 None。多卡训练初始化 Executor 时也只用传入一个 Place 或 None，其他 API 会处理使用的多卡，见 `多卡使用方式 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/update_cn.html#danjiduokaqidong>`_

返回
::::::::::::
初始化后的 ``Executor`` 对象。

代码示例
::::::::::::

COPY-FROM: paddle.static.Executor

方法
::::::::::::
close()
'''''''''


关闭执行器。该接口主要用于对于分布式训练，调用该接口后不可以再使用该执行器。该接口会释放在 PServers 上和目前 Trainer 有关联的资源。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.Executor.close

run(program=None, feed=None, fetch_list=None, feed_var_name='feed', fetch_var_name='fetch', scope=None, return_numpy=True, use_program_cache=False, use_prune=False)
'''''''''

执行指定的 Program 或者 CompiledProgram。需要注意的是，执行器会执行 Program 或 CompiledProgram 中的所有算子，而不会根据 fetch_list 对 Program 或 CompiledProgram 中的算子进行裁剪。同时，需要传入运行该模型用到的 scope，如果没有指定 scope，执行器将使用全局 scope，即 paddle.static.global_scope()。

**参数**

  - **program** (Program|CompiledProgram，可选) – 该参数为被执行的 Program 或 CompiledProgram，如果未提供该参数，即该参数为 None，在该接口内，main_program 将被设置为 paddle.static.default_main_program()。默认为：None。
  - **feed** (list|dict，可选) – 该参数表示模型的输入变量。如果是单卡训练，``feed`` 为 ``dict`` 类型，如果是多卡训练，参数 ``feed`` 可以是 ``dict`` 或者 ``list`` 类型变量，如果该参数类型为 ``dict`` ，feed 中的数据将会被分割(split)并分送给多个设备（CPU/GPU），即输入数据被均匀分配到不同设备上；如果该参数类型为 ``list``，则列表中的各个元素都会直接分别被拷贝到各设备中。默认为：None。
  - **fetch_list** (list，可选) – 该参数表示模型运行之后需要返回的变量。默认为：None。
  - **feed_var_name** (str，可选) – 该参数表示数据输入算子(feed operator)的输入变量名称。默认为："feed"。
  - **fetch_var_name** (str，可选) – 该参数表示结果获取算子(fetch operator)的输出变量名称。默认为："fetch"。
  - **scope** (Scope，可选) – 该参数表示执行当前 program 所使用的作用域，用户可以为不同的 program 指定不同的作用域。默认值：paddle.static.global_scope()。
  - **return_numpy** (bool，可选) – 该参数表示是否将返回的计算结果（fetch list 中指定的变量）转化为 numpy；如果为 False，则每个变量返回的类型为 Tensor，否则返回变量的类型为 numpy.ndarray。默认为：True。
  - **use_program_cache** (bool，可选) – 该参数表示是否对输入的 Program 进行缓存。如果该参数为 True，在以下情况时，模型运行速度可能会更快：输入的 program 为 ``paddle.static.Program``，并且模型运行过程中，调用该接口的参数（program、 feed 变量名和 fetch_list 变量）名始终不变。默认为：False。
  - **use_prune** (bool，可选) - 该参数表示输入 Program 是否会被裁剪。如果该参数为 True，会根据 feed 和 fetch_list 裁剪 Program，这意味着对生成 fetch_list 没有必要的算子和变量会被裁剪掉。默认为 False，即算子和变量在运行过程不会被裁剪。注意如果 Optimizer.minimize()返回的 tuple 被作为 fetch_list 参数，那么 use_prune 会被重载为 True 并且 Program 会被裁剪。

**返回**

返回 fetch_list 中指定的变量值。

.. note::
     1. 如果是多卡训练，并且 feed 参数为 dict 类型，输入数据将被均匀分配到不同的卡上，例如：使用 2 块 GPU 训练，输入样本数为 3，即[0, 1, 2]，经过拆分之后，GPU0 上的样本数为 1，即[0]，GPU1 上的样本数为 2，即[1, 2]。如果样本数少于设备数，程序会报错，因此运行模型时，应额外注意数据集的最后一个 batch 的样本数是否少于当前可用的 CPU 核数或 GPU 卡数，如果是少于，建议丢弃该 batch。
     2. 如果可用的 CPU 核数或 GPU 卡数大于 1，则 fetch 出来的结果为不同设备上的相同变量值（fetch_list 中的变量）在第 0 维拼接在一起。


**代码示例 1**

COPY-FROM: paddle.static.Executor.run:code-example-1

**代码示例 2**

COPY-FROM: paddle.static.Executor.run:code-example-2

infer_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100, fetch_handler=None)
'''''''''

infer_from_dataset 的文档与 train_from_dataset 几乎完全相同，只是在分布式训练中，推进梯度将在 infer_from_dataset 中禁用。infer_from_dataset（）可以非常容易地用于多线程中的评估。

**参数**

  - **program** (Program|CompiledProgram，可选) – 需要执行的 program，如果没有给定那么默认使用 default_main_program (未编译的)。
  - **dataset** (paddle.fluid.Dataset，可选) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查 Dataset 文件。默认为 None。
  - **scope** (Scope，可选) – 执行这个 program 的域，用户可以指定不同的域。默认为全局域。
  - **thread** (int，可选) – 用户想要在这个函数中运行的线程数量。线程的实际数量为 min(Dataset.thread_num, thread)，如果 thread > 0，默认为 0。
  - **debug** (bool，可选) – 是否开启 debug 模式，默认为 False。
  - **fetch_list** (Tensor List，可选) – 返回变量列表，每个变量都会在预测过程中被打印出来，默认为 None。
  - **fetch_info** (String List，可选) – 每个变量的打印信息，默认为 None。
  - **print_period** (int，可选) – 每两次打印之间间隔的 mini-batches 的数量，默认为 100。
  - **fetch_handler** (FetchHandler，可选) - 获取用户定义的输出类。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.Executor.infer_from_dataset

train_from_dataset(program=None, dataset=None, scope=None, thread=0, debug=False, fetch_list=None, fetch_info=None, print_period=100, fetch_handler=None)
'''''''''

从预定义的数据集中训练。数据集在 paddle.fluid.dataset 中定义。给定程序（或编译程序），train_from_dataset 将使用数据集中的所有数据样本。输入范围可由用户给出。默认情况下，范围是 global_scope()。训练中的线程总数是 thread。训练中使用的线程数将是数据集中 threadnum 的最小值，同时也是此接口中线程的值。可以设置 debug，以便执行器显示所有算子的运行时间和当前训练任务的吞吐量。

.. note::
train_from_dataset 将销毁每次运行在 executor 中创建的所有资源。

**参数**

  - **program** (Program|CompiledProgram，可选) – 需要执行的 program，如果没有给定那么默认使用 default_main_program (未编译的)。
  - **dataset** (paddle.fluid.Dataset，可选) – 在此函数外创建的数据集，用户应当在调用函数前提供完整定义的数据集。必要时请检查 Dataset 文件。默认为 None。
  - **scope** (Scope，可选) – 执行这个 program 的域，用户可以指定不同的域。默认为全局域。
  - **thread** (int，可选) – 用户想要在这个函数中运行的线程数量。线程的实际数量为 min(Dataset.thread_num, thread)，如果 thread > 0，默认为 0。
  - **debug** (bool，可选) – 是否开启 debug 模式，默认为 False。
  - **fetch_list** (Tensor List，可选) – 返回变量列表，每个变量都会在训练过程中被打印出来，默认为 None。
  - **fetch_info** (String List，可选) – 每个变量的打印信息，默认为 None。
  - **print_period** (int，可选) – 每两次打印之间间隔的 mini-batches 的数量，默认为 100。
  - **fetch_handler** (FetchHandler，可选) - 获取用户定义的输出类。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.Executor.train_from_dataset
