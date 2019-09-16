.. _best_practice_dist_training_gpu:

#####################
分布式GPU训练最佳实践
#####################

开始优化您的GPU分布式训练任务
---------------------------

PaddlePaddle Fluid可以支持在现代GPU [#]_ 服务器集群上完成高性能分布式训练。
通常可以通过以下方法优化在多机多卡环境训练性能，建议在进行性能优化时，
检查每项优化点并验证对应提升，最终获得最优性能。

一个简单的验证当前的训练程序是否需要进一步优化性能的方法，
是查看GPU的计算利用率 [#]_ ，通常用 :code:`nvidia-smi` 命令查看。
如果GPU利用率较低，则可能存在较大的优化空间。
下面主要从环境变量设置、训练策略设置、数据准备和训练方式四个方向介绍GPU分布式训练中常用的方法。

1、环境变量设置
=============

环境变量设置表

..  csv-table:: 
    :header: "调节项", "可选值", "说明"
    :widths: 3, 3, 5

    ":code:`FLAGS_sync_nccl_allreduce`", "0,1", "是否同步AllReduce操作。1表示开启,每次调用等待AllReduce同步"
    ":code:`FLAGS_fraction_of_gpu_memory_to_use`", "0~1之间的float值",  "预先分配显存的占比"
    ":code:`NCCL_IB_DISABLE` ", "0,1", "是否启用RDMA多机通信。如果机器硬件支持，可以设置1，开启RDMA支持"

说明：

- 关于 :code:`FLAGS_sync_nccl_allreduce` ，配置 :code:`FLAGS_sync_nccl_allreduce=1` 让每次allreduce操作都等待完成，可以提升性能，详细原因和分析可以参考：https://github.com/PaddlePaddle/Paddle/issues/15049。
- 关于  :code:`FLAGS_fraction_of_gpu_memory_to_use` ，配置 :code:`FLAGS_fraction_of_gpu_memory_to_use=0.95` ，0.95是指95%的显存会预先分配。设置的范围是0.0~1.0。注意，设置成0.0会让每次显存分配都调用 :code:`cudaMalloc` 这样会极大的降低训练性能。
- 关于 :code:`NCCL_IB_DISABLE` ，在使用NCCL2模式训练时，其会默认尝试开启RDMA通信，如果系统不支持，则会自动降级为使用TCP通信。可以通过打开环境变量 :code:`NCCL_DEBUG=INFO` 查看NCCL是否选择了开启RDMA通信。如果需要强制使用TCP方式通信，可以设置 :code:`NCCL_IB_DISABLE=1` 。


2、训练策略设置
===========

训练参数设置表

..  csv-table:: 
    :header: "选项", "类型", "默认值", "说明"
    :widths: 3, 3, 3, 5

    ":code:`num_threads`", "int", "1", "CPU线程数"
    ":code:`nccl_comm_num`", "int", "1", "nccl通信器数量"
    ":code:`fuse_all_reduce_ops`", "bool", "False", "多卡训练时，将AllReduce操纵进行融合"
    ":code:`use_hierarchical_allreduce` ", "bool", "False","分级式reduce"
    ":code:`num_iteration_per_drop_scope`", "int", "1", "scope drop频率，设置每隔几个batch的迭代之后执行一次清理scope"
    ":code:`fetch_frequency`", "int", "1", "fetch的刷新频率"

说明：

- 关于设置合适的CPU线程数 :code:`num_threads` 和nccl通信器数量 :code:`nccl_comm_num` 。PaddlePaddle Fluid使用“线程池” [#]_ 模型调度并执行Op，Op在启动GPU计算之前，通常需要CPU的协助，然而如果Op本身占用时间很小，“线程池”模型下又会带来额外的调度开销。使用多进程模式时，如果神经网络的计算图 [#]_ 节点间有较高的并发度，即使每个进程只在一个GPU上运行，使用多个线程可以更大限度的提升GPU利用率。nccl通信器数量 :code:`nccl_comm_num` 可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。针对CPU线程数 :code:`num_threads` ，建议单机设置为1，多机设置为 :code:`nccl_comm_num` +1。
- 关于AllReduce融合 :code:`fuse_all_reduce_ops` ，默认情况下会将同一layer中参数的梯度的AllReduce操作合并成一个，比如对于 :code:`fluid.layers.fc` 中有Weight和Bias两个参数，打开该选项之后，原本需要两次AllReduce操作，现在只用一次AllReduce 操作。此外，为支持更大粒度的参数梯度融合，Paddle提供了 :code:`FLAGS_fuse_parameter_memory_size` 和 :code:`FLAGS_fuse_parameter_groups_size` 两个环境变量选项。用户可以指定融合AllReduce操作之后，每个AllReduce操作的梯度字节数，比如希望每次AllReduce调用传输16MB的梯度，:code:`export FLAGS_fuse_parameter_memory_size=16` ，经验值为总通信量的十分之一。可以指定每次AllReduce操作的最大层数，即到达该层数就进行AllReduce，如指定50层 :code:`export FLAGS_fuse_parameter_groups_size=50` 。注意：目前不支持sparse参数梯度。
- 关于使用分级式reduce :code:`use_hierarchical_allreduce` 。对于多机模式，针对小数据量的通信，Ring AllReduce通信效率低，采用Hierarchical AllReduce可以解决该问题。
- 关于降低scope drop频率 :code:`num_iteration_per_drop_scope` 和fetch频率 :code:`fetch_frequency` 。减少scope drop和fetch频率，可以减少频繁的变量内存申请、释放和拷贝，从而提升性能。

设置这些参数可以参考：

.. code-block:: python
   :linenos:

   dist_strategy = DistributedStrategy()
   dist_strategy.nccl_comm_num = 2                    #建议多机设置为2，单机设置为1
   exec_strategy = fluid.ExecutionStrategy()
   exe_st.num_threads = 3                             #建议多机设置为nccl_comm_num+1，单机设置为1
   exec_strategy.num_iteration_per_drop_scope = 30    #scope drop频率
   dist_strategy.exec_strategy = exec_strategy
   dist_strategy.fuse_all_reduce_ops = True           #AllReduce是否融合
                ...
   with fluid.program_guard(main_prog, startup_prog): #组网
       params = model.params
       optimizer = optimizer_setting(params)
       dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
       dist_optimizer.minimize(avg_cost)
                ...
   for pass_id in range(PASS_NUM):
       batch_id = 0
       while True:
           if batch_id % fetch_frequency == 0:        #fetch频率
               fetched = exe.run(main_prog, fetch_list)
           else:
               exe.run([])


3、数据准备
===========

1、使用GPU完成部分图片预处理

如果可能，使用GPU完成可以部分数据预处理，比如图片Tensor的归一化：

.. code-block:: python
   :linenos:

   image = fluid.layers.data()
   img_mean = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_mean", persistable=True)
   img_std = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_std", persistable=True)
   t1 = fluid.layers.elementwise_sub(image / 255.0, img_mean, axis=1)
   image = fluid.layers.elementwise_div(t1, img_std, axis=1)

对输入的图片Tensor，使用 :code:`fluid.layers` 完成图片数据归一化预处理，
这样可以减轻CPU预处理数据的负担，提升总体训练速度。

2、优化reader性能

数据读取的优化在GPU训练中至关重要，尤其在不断增加batch_size提升吞吐时，计算对reader性能会有更高对要求，
优化reader性能需要考虑的点包括：

 - 使用 :code:`pyreader` 。参考 `这里 <../../user_guides/howto/prepare_data/use_py_reader.html>`_ 使用pyreader，并开启 :code:`use_double_buffer` 。
 - reader返回uint8类型数据。图片在解码后一般会以uint8类型存储，如果在reader中转换成float类型数据，会将数据体积扩大4倍。直接返回uint8数据，然后在GPU上转化成float类型进行训练
 - 减少reader初始化时间 (infinite read)
   在训练任务开始执行第一轮训练时，reader开始异步的，不断的从磁盘或其他存储中读取数据并执行预处理，然后将处理好的数据
   填充到队列中供计算使用。从0开始填充这个队列直到数据可以源源不断供给计算，需要一定时间的预热。所以，如果每轮训练
   都重新填充队列，会产生一些时间的开销。所以，在使用pyreader时，可以让reader函数不断的产生数据，直到训练循环手动break：

   .. code-block:: python
      :linenos:

      def infinite_reader(file_path):
          while True:
              with open(file_path) as fn:
                  for line in fn:
                      yield process(line)

      def train():
          ...
          for pass_id in xrange(NUM_PASSES):
              if pass_id == 0:
                  pyreader.start()
              for batch_id in (iters_per_pass):
                  exe.run()
          pyreader.reset()

4、训练方式
===========

1、Local SGD

GPU多机多卡同步训练过程中存在慢trainer现象，
即每步中训练快的trainer的同步通信需要等待训练慢的trainer。
由于每步中慢trainer的rank具有随机性，
因此我们使用局部异步训练的方式——LocalSGD，
通过多步异步训练（无通信阻塞）实现慢trainer时间均摊，
从而提升同步训练性能。
Local SGD训练方式主要有三个参数，分别是：

..  csv-table:: 
    :header: "选项", "类型", "可选值", "说明"
    :widths: 3, 3, 3, 5

    ":code:`use_local_sgd`", "bool", "False/True", "是否开启Local SGD，默认不开启"
    ":code:`local_sgd_is_warm_steps`", "int", "大于0", "训练多少轮之后才使用Local SGD方式训练"
    ":code:`local_sgd_steps`", "int", "大于0", "Local SGD的步长"

说明：

- Local SGD的warmup步长 :code:`local_sgd_is_warm_steps` 影响最终模型的泛化能力，一般需要等到模型参数稳定之后在进行Local SGD训练，经验值可以将学习率第一次下降时的epoch作为warmup步长，之后再进行Local SGD训练。
- Local SGD步长 :code:`local_sgd_steps` ，一般该值越大，通信次数越少，训练速度越快，但随之而来的时模型精度下降。经验值设置为2或者4。

具体的Local SGD的训练代码可以参考：
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet


2、使用混合精度训练

V100 GPU提供了 `Tensor Core <https://www.nvidia.com/en-us/data-center/tensorcore/>`_ 可以在混合精度计算
场景极大的提升性能。使用混合精度计算的例子可以参考：
https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#using-mixed-precision-training

目前Paddle只提供在两个模型（ResNet, BERT）的混合精度计算实现并支持static loss scaling，其他模型使用混合精度也
可以参考以上的实现完成验证。

附录
----

.. [#] 现代GPU：指至少支持运行 `CUDA <https://developer.nvidia.com/cuda-downloads>`_ 版本7.5以上的GPU
.. [#] GPU利用率：这里指GPU计算能力被使用部分所占的百分比
.. [#] https://en.wikipedia.org/wiki/Thread_pool
.. [#] https://en.wikipedia.org/wiki/Data-flow_diagram
