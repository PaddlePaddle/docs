.. _best_practice_dist_training_gpu:

性能优化最佳实践之：GPU分布式训练
============================

开始优化您的GPU分布式训练任务
-------------------------

PaddlePaddle Fluid可以支持在现代GPU服务器集群上完成高性能分布式训练。通常可以通过以下手段
优化在多机多卡环境训练性能，建议在进行性能优化时，检查每项优化点并验证对应提升，最终获得最优性能。

一个简单的验证当前的训练程序是否需要进一步优化性能的方法，是查看GPU的计算利用率，通常用 :code:`nvidia-smi`
命令查看。如果GPU利用率较低，则可能存在较大的优化空间。

下列表格中列出本文将介绍的所有可优化点的概述：

可配置项一览
++++++++++

..  csv-table:: GPU分布式训练性能调节项
    :header: "调节项", "可选值说明", "配置方法"
    :widths: 3, 3, 5

    "通信模式", "pserver模式；NCCL2模式（collective）", "配置方法参考： `这里 <http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2->`_ "
    "执行模式", "单进程；单进程ParallelGraph；多进程", "配置方法参考： `这里 <http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/training/cluster_howto.html#permalink-9--nccl2->`_ "
    "同步AllReduce操作", "开启则使每次调用等待AllReduce同步", "设置环境变量 :code:`FLAGS_sync_nccl_allreduce`"
    "CPU线程数", "int值，配置使用的CPU线程数", "参考本片后续说明"
    "预先分配足够的显存", "0~1之间的float值，预先分配显存的占比", "设置环境变量 :code:`FLAGS_fraction_of_gpu_memory_to_use`"
    "scope drop频率", "int值，设置每隔N个batch的迭代之后执行一次清理scope", "设置 :code:`fluid.ExecutionStrategy().num_iteration_per_drop_scope`"
    "fetch频率", "代码配置", "参考本片后续说明"
    "启用RDMA多机通信", "如果机器硬件支持，可以选择开启RDMA支持", "配置环境变量 :code:`NCCL_IB_DISABLE` "
    "使用GPU完成部分图片预处理", "代码配置", "参考本片后续说明"
    "设置通信频率（batch merge）", "代码配置", "参考本片后续说明"
    "优化reader性能", "代码优化", "参考本片后续说明"
    "混合精度", "FP32训练；混合FP32,FP16训练（在V100下启用TensorCore）", "参考项目：`图像分类 <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification>`_ "


通信模式，执行模式
++++++++++++++++

GPU分布式训练场景，使用多进程+NCCL2模式（collective）通常可以获得最好的性能。参考 `这里 <http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2->`_ 配置您的程序
使用多进程NCCL2模式训练。

在进程模式下，每台服务器的每个GPU卡都会对应启动一个训练进程，集群中的所有进程之间会互相通信完成训练。以此方式最大
限度的降低进程内部资源抢占的开销。对比在单进程开启ParallelGraph方法，多进程模式不但可以获得更高性能，而且无需考虑
reader在多卡下io性能不足的问题，直接使用多进程提升数据读取IO效率。

使用ParallelGraph模式相对而言会减少多进程管理，并提升性能，而且可以无需修改代码，只需要开启下列开关即可：

.. code-block:: bash
   :linenos:

   export FLAGS_enable_parallel_graph=1

如果是单机多卡模式，同样可以通过开启ParallelGraph来提升性能：

.. code-block:: bash
   :linenos:

   export FLAGS_enable_parallel_graph=1
   export FLAGS_sync_nccl_allreduce=1

注：在单机多卡ParallelGraph模式下，配置 :code:`FLAGS_sync_nccl_allreduce=1` 让每次allreduce操作都等待完成，可以提升性能，
详细原因和分析可以参考：https://github.com/PaddlePaddle/Paddle/issues/15049


设置合适的CPU线程数
+++++++++++++++++

PaddlePaddle Fluid使用“线程池”模型调度并执行Op，Op在启动GPU计算之前，通常需要CPU的协助，然而如果Op本身占用时间很小，
“线程池”模型下又回带来额外的调度开销。使用多进程模式时，如果神经网络的计算图节点间有较高的并发度，即使每个进程只在一个GPU上
运行，使用多个线程可以更大限度的提升GPU利用率。这项配置需要根据运行模型的情况来配置，通常在多进程模式，设置线程数为1和4，
然后观察是否存在提升，然后逐步调整此项配置。设置CPU线程数的方法参考：

.. code-block:: python
   :linenos:

   exe_st = fluid.ExecutionStrategy()
   exe_st.num_threads = 1
   exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=exe_st,
        num_trainers=num_trainers,
        trainer_id=trainer_id)


预先分配足够的显存
+++++++++++++++

通过设置环境变量 :code:`FLAGS_fraction_of_gpu_memory_to_use=0.95` 设置预先分配的显存占比，比如0.95是指95%的
显存会预先分配。设置的范围是0.0~1.0。注意，设置成0.0会让每次显存分配都调用 :code:`cudaMalloc` 这样会极大的降低训练
性能。

降低scope drop频率和fetch频率
+++++++++++++++++++++++++++

减少scope drop和fetch频率，可以减少频繁的变量内存申请、释放和拷贝，从而提升性能。配置这两项的方法参考下面代码：

.. code-block:: python
   :linenos:

   exe_st = fluid.ExecutionStrategy()
   strategy.num_iteration_per_drop_scope = 30
   exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=exe_st,
        num_trainers=num_trainers,
        trainer_id=trainer_id)
   for pass_id in xrange(PASS_NUM):
       batch_id = 0
       while True:
           if batch_id % 30 == 0:
               fetched = exe.run(fetch_list)
           else:
               exe.run([])


启用RDMA多机通信
++++++++++++++

在使用NCCL2模式训练时，其会默认尝试开启RDMA通信，如果系统不支持，则会自动降级为使用TCP通信。可以通过打开
环境变量 :code:`NCCL_DEBUG=INFO` 查看NCCL是否选择了开启RDMA通信。如果需要强制使用TCP方式通信，可以设置
:code:`NCCL_IB_DISABLE=1` 。


使用GPU完成部分图片预处理
++++++++++++++++++++++

如果可能，使用GPU完成可以部分数据预处理，比如图片Tensor的归一化：

.. code-block:: python
   :linenos:

   image = fluid.layers.data()
   img_mean = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_mean", persistable=True)
   img_std = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_std", persistable=True)
   t1 = fluid.layers.elementwise_sub(image / 255.0, img_mean, axis=1)
   image = fluid.layers.elementwise_div(t1, img_std, axis=1)

对输入的图片Tensor，使用 :code:`fluid.layers` 完成图片数据归一化预处理，这样可以减轻CPU预处理数据的负担，提升总体训练速度。

增大batch_size或使用设置通信频率（batch merge）
++++++++++++++++++++++++++++++++++++++++++

分布式同步训练，跨界点通信或多或少会带来性能影响，增大训练的batch_size，可以保持通信开销不变的情况下，增大计算吞吐
从而降低通信在整个训练过程中的占比来提升总体的训练吞吐。

然而增大batch_size会带来同等比例的显存消耗提升，为了进一步的增大batch_size，Fluid提供“batch merge”功能，通过
在一个GPU上串行计算多个小的batch并积累梯度，然后再执行多机多卡之间的通信，此模式同样也可以被称为“可变通信频率“。使用
batch merge功能，在同样的模型，可以极大的增加batch size，提升多机训练的总吞吐。
使用方法可以参考实例：https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/dist_train


优化reader性能
+++++++++++++

数据读取的优化在GPU训练中至关重要，尤其在不断增加batch_size提升吞吐时，计算对reader性能会有更高对要求，
优化reader性能需要考虑的点包括：

1. 使用 :code:`pyreader` 
   参考 `这里 <http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/prepare_data/use_py_reader.html>`_
   使用pyreader，并开启 :code:`use_double_buffer`
2. reader返回uint8类型数据
   图片在解码后一般会以uint8类型存储，如果在reader中转换成float类型数据，会将数据体积扩大4倍。直接返回uint8数据，然后在GPU
   上转化成float类型进行训练
3. reader pin memory
   reader读取的数据会在训练时组成batch，并从CPU拷贝到GPU上，如果在CPU上分配pin memory内存，这个拷贝过程可以通过硬件
   DMA完成拷贝提升性能。在使用pyreader的方式下，可以使用下面的实例代码开启pin memory batch reader：

   .. code-block:: python
      :linenos:

      def batch_feeder(batch_reader, pin_memory=True, img_dtype="uint8"):
          # batch((sample, label)) => batch(sample), batch(label)
          def _feeder():
              for batch_data in batch_reader():
                  sample_batch = []
                  label_batch = []
                  for sample, label in batch_data:
                      sample_batch.append(sample)
                      label_batch.append([label])
                  tensor = core.LoDTensor()
                  label = core.LoDTensor()
                  place = core.CUDAPinnedPlace() if pin_memory else core.CPUPlace()
                  tensor.set(np.array(sample_batch, dtype=img_dtype, copy=False), place)
                  label.set(np.array(label_batch, dtype="int64", copy=False), place)
                  yield [tensor, label]
          return _feeder

      pyreader.decorate_tensor_provider(
        batch_feeder(
            paddle.batch(rd, batch_size=batch_size_per_gpu),
            pin_memory=True,
            img_dtype='uint8'
        )
    )

4. 减少reader初始化时间 (infinite read）
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
   

使用混合精度训练
++++++++++++++

V100 GPU提供了 `Tensor Core <https://www.nvidia.com/en-us/data-center/tensorcore/>`_ 可以在混合精度计算
场景极大的提升性能。使用混合精度计算的例子可以参考：
https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#using-mixed-precision-training

目前Paddle只提供在两个模型（ResNet, BERT）的混合精度计算实现并支持static loss scaling，其他模型使用混合精度也
可以参考以上的实现完成验证。
