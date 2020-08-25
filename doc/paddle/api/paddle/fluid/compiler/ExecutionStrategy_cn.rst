.. _cn_api_fluid_ExecutionStrategy:

ExecutionStrategy
-------------------------------


.. py:class:: paddle.fluid.ExecutionStrategy

:api_attr: 声明式编程模式（静态图)



通过设置 ``ExecutionStrategy`` 中的选项，用户可以对执行器的执行配置进行调整，比如设置执行器中线程池的大小等。

返回：初始化后的ExecutionStrategy的实例

返回类型：ExecutionStrategy

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


.. py:attribute:: num_iteration_per_drop_scope

int型成员。该选项表示间隔多少次迭代之后清理一次临时变量。模型运行过程中，生成的中间临时变量将被放到local execution scope中，为了避免对临时变量频繁的申请与释放，通常将其设为较大的值（比如10或者100）。默认值为100。


.. py:attribute:: num_iteration_per_run

int型成员。它配置了当用户在python脚本中调用pe.run()时执行器会执行的迭代次数。Executor每次调用，会进行num_iteration_per_run次训练，它会使整体执行过程更快。

.. py:attribute:: num_threads

int型成员。该选项表示当前 ``Executor`` 的线程池(thread pool)的大小, 此线程池可用来并发执行program中的operator（算子，运算）。如果 :math:`num\_threads=1` ，则所有的operator将一个接一个地执行，但在不同的program重复周期(iterations)中执行顺序可能不同。如果该选项没有被设置，则在 ``Executor`` 中，它会依据设备类型(device type)、设备数目(device count)而设置为相应值。对GPU，:math:`num\_threads=device\_count∗4` ；对CPU， :math:`num\_threads=CPU\_NUM∗4` 。在 ``Executor`` 中有关于 :math:`CPU\_NUM` 的详细解释。如果没有设置 :math:`CPU\_NUM` ，则设置默认值为1， 并提示用户进行 :math:`CPU\_NUM` 的设置。












