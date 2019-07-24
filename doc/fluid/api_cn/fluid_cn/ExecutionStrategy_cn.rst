.. _cn_api_fluid_ExecutionStrategy:

ExecutionStrategy
-------------------------------

.. py:class:: paddle.fluid.ExecutionStrategy

``ExecutionStrategy`` 允许用户更加精准地控制program在 ``ParallelExecutor`` 中的运行方式。可以通过在 ``ParallelExecutor`` 中设置本成员来实现。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
     
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)
     
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4

    train_exe = fluid.ParallelExecutor(use_cuda=False,
                                       loss_name=avg_loss.name,
                                     exec_strategy=exec_strategy)



.. py:attribute:: allow_op_delay

这是一个bool类型成员，表示是否推迟communication operators(交流运算)的执行，这样做会使整体执行过程更快一些。但是在一些模型中，allow_op_delay会导致程序中断。默认为False。



.. py:attribute:: num_iteration_per_drop_scope

int型成员。它表明了清空执行时产生的临时变量需要的程序执行迭代次数。因为临时变量的形状可能在两次重复过程中保持一致，所以它会使整体执行过程更快。默认值为1。

.. note::
  1. 如果在调用 ``run`` 方法时获取结果数据，``ParallelExecutor`` 会在当前程序重复执行尾部清空临时变量

  2. 在一些NLP模型里，该成员会致使GPU内存不足。此时，你应减少 ``num_iteration_per_drop_scope`` 的值

.. py:attribute:: num_iteration_per_run
它配置了当用户在python脚本中调用pe.run()时执行器会执行的迭代次数。

.. py:attribute:: num_threads

int型成员。它代表了线程池(thread pool)的大小。这些线程会被用来执行当前 ``ParallelExecutor`` 的program中的operator（算子，运算）。如果 :math:`num\_threads=1` ，则所有的operator将一个接一个地执行，但在不同的程序重复周期(iterations)中执行顺序可能不同。如果该成员没有被设置，则在 ``ParallelExecutor`` 中，它会依据设备类型(device type)、设备数目(device count)而设置为相应值。对GPU，:math:`num\_threads=device\_count∗4` ；对CPU， :math:`num\_threads=CPU\_NUM∗4` 。在 ``ParallelExecutor`` 中有关于 :math:`CPU\_NUM` 的详细解释。如果没有设置 :math:`CPU\_NUM` ， ``ParallelExecutor`` 可以通过调用 ``multiprocessing.cpu_count()`` 获取CPU数目(cpu count)。默认值为0。












