.. _cn_api_fluid_ParallelExecutor:

ParallelExecutor
-------------------------------


.. py:class:: paddle.static.ParallelExecutor(use_cuda, loss_name=None, main_program=None, share_vars_from=None, exec_strategy=None, build_strategy=None, num_trainers=1, trainer_id=0, scope=None)




``ParallelExecutor`` 是 ``Executor`` 的一个升级版本，可以支持基于数据并行的多节点模型训练和测试。如果采用数据并行模式，``ParallelExecutor`` 在构造时会将参数分发到不同的节点上，并将输入的 ``Program`` 拷贝到不同的节点，在执行过程中，各个节点独立运行模型，将模型反向计算得到的参数梯度在多个节点之间进行聚合，之后各个节点独立的进行参数的更新。

- 如果使用 GPU 运行模型，即 ``use_cuda=True``，节点指代 GPU， ``ParallelExecutor`` 将自动获取在当前机器上可用的 GPU 资源，用户也可以通过在环境变量设置可用的 GPU 资源，例如：希望使用 GPU0、GPU1 计算，export CUDA_VISIBLEDEVICES=0,1；
- 如果在 CPU 上进行操作，即 ``use_cuda=False``，节点指代 CPU，**注意：此时需要用户在环境变量中手动添加 CPU_NUM，并将该值设置为 CPU 设备的个数，例如：export CPU_NUM=4，如果没有设置该环境变量，执行器会在环境变量中添加该变量，并将其值设为 1**。

参数
::::::::::::

    - **use_cuda** (bool) – 该参数表示是否使用 GPU 执行。
    - **loss_name** （str） - 该参数为模型最后得到的损失变量的名字。**注意：如果是数据并行模型训练，必须设置 loss_name，否则计算结果可能会有问题。** 默认为：None。
    - **main_program** (Program) – 需要被执行的 Program。如果未提供该参数，即该参数为 None，在该接口内，main_program 将被设置为 paddle.static.default_main_program()。默认为：None。
    - **share_vars_from** (ParallelExecutor) - 如果设置了 share_vars_from，当前的 ParallelExecutor 将与 share_vars_from 指定的 ParallelExecutor 共享参数值。
    需要设置该参数的情况：模型训练过程中需要进行模型测试，并且训练和测试都是采用数据并行模式，那么测试对应的 ParallelExecutor 在调用 with_data_parallel 时，需要将 share_vars_from 设置为训练所对应的 ParallelExecutor。
    由于 ParallelExecutor 只有在第一次执行时才会将参数变量分发到其他设备上，因此 share_vars_from 指定的 ParallelExecutor 必须在当前 ParallelExecutor 之前运行。默认为：None。
    - **exec_strategy** (ExecutionStrategy) -  通过 exec_strategy 指定执行计算图过程可以调整的选项，例如线程池大小等。关于 exec_strategy 更多信息，请参阅 ``paddle.static.ExecutionStrategy``。默认为：None。
    - **build_strategy** (BuildStrategy)：通过配置 build_strategy，对计算图进行转换和优化，例如：计算图中算子融合、计算图执行过程中开启内存/显存优化等。关于 build_strategy 更多的信息，请参阅  ``paddle.static.BuildStrategy``。默认为：None。
    - **num_trainers** (int) – 进行 GPU 分布式训练时需要设置该参数。如果该参数值大于 1，NCCL 将会通过多层级节点的方式来初始化。每个节点应有相同的 GPU 数目。默认为：1。
    - **trainer_id** (int) –  进行 GPU 分布式训练时需要设置该参数。该参数必须与 num_trainers 参数同时使用。trainer_id 指明是当前所在节点的 “rank”（层级）。trainer_id 从 0 开始计数。默认为：0。
    - **scope** (Scope) – 指定执行 Program 所在的作用域。默认为：paddle.static.global_scope()。

返回
::::::::::::
初始化后的 ``ParallelExecutor`` 对象。

.. note::
     1. 如果只是进行多卡测试，不需要设置 loss_name 以及 share_vars_from。
     2. 如果程序中既有模型训练又有模型测试，则构建模型测试所对应的 ParallelExecutor 时必须设置 share_vars_from，否则模型测试和模型训练所使用的参数是不一致。

代码示例
::::::::::::

COPY-FROM: paddle.static.ParallelExecutor

方法
::::::::::::
run(fetch_list, feed=None, feed_dict=None, return_numpy=True)
'''''''''

运行当前模型，需要注意的是，执行器会执行 Program 中的所有算子，而不会根据 fetch_list 对 Program 中的算子进行裁剪。

**参数**

    - **fetch_list** (list) – 该变量表示模型运行之后需要返回的变量。
    - **feed** (list|dict) – 该变量表示模型的输入变量。如果该参数类型为 ``dict`` ，feed 中的数据将会被分割(split)并分送给多个设备（CPU/GPU）；如果该参数类型为 ``list``，则列表中的各个元素都会直接分别被拷贝到各设备中。默认为：None。
    - **feed_dict** – 该参数已经停止使用。默认为：None。
    - **return_numpy** (bool) – 该变量表示是否将 fetched tensor 转换为 numpy。默认为：True。

**返回**

返回 fetch_list 中指定的变量值。

.. note::
     1. 如果 feed 参数为 dict 类型，输入数据将被均匀分配到不同的卡上，例如：使用 2 块 GPU 训练，输入样本数为 3，即[0, 1, 2]，经过拆分之后，GPU0 上的样本数为 1，即[0]，GPU1 上的样本数为 2，即[1, 2]。如果样本数少于设备数，程序会报错，因此运行模型时，应额外注意数据集的最后一个 batch 的样本数是否少于当前可用的 CPU 核数或 GPU 卡数，如果是少于，建议丢弃该 batch。
     2. 如果可用的 CPU 核数或 GPU 卡数大于 1，则 fetch 出来的结果为不同设备上的相同变量值（fetch_list 中的变量）在第 0 维拼接在一起。

**代码示例**

COPY-FROM: paddle.static.ParallelExecutor.run

drop_local_exe_scopes()
'''''''''

立即清除 scope 中的临时变量。模型运行过程中，生成的中间临时变量将被放到 local execution scope 中，为了避免对临时变量频繁的申请与释放，ParallelExecutor 中采取的策略是间隔若干次迭代之后清理一次临时变量。ParallelExecutor 在 ExecutionStrategy 中提供了 num_iteration_per_drop_scope 选项，该选项表示间隔多少次迭代之后清理一次临时变量。如果 num_iteration_per_drop_scope 值为 100，但是希望在迭代 50 次之后清理一次临时变量，可以通过手动调用该接口。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.ParallelExecutor.drop_local_exe_scopes
