.. _cn_api_fluid_dygraph_DataParallel:

DataParallel
-------------------------------

.. py:class:: paddle.fluid.dygraph.DataParallel(layers, strategy)

该接口用于构建 ``DataParallel`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。用于实现在数据并行模式下运行模型。

当前， ``DataParallel`` 仅支持使用多进程来运行动态图程序，具体用法如下（其中 ``dynamic_graph_test.py`` 是包含示例代码的文件）：

``python -m paddle.distributed.launch --selected_gpus=0,1 dynamic_graph_test.py``

参数：
    - **layers** (Layer) - 需要在数据并行模式下运行的模型。
    - **strategy** (ParallelStrategy) - 数据并行化策略。由 :ref:`cn_api_fluid_dygraph_prepare_context` 产生的对象。

返回：
    None

.. code-block:: python

   import numpy as np
   import paddle.fluid as fluid
   import paddle.fluid.dygraph as dygraph
   from paddle.fluid.optimizer import AdamOptimizer
   from paddle.fluid.dygraph.nn import Linear
   from paddle.fluid.dygraph.base import to_variable

   place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
   with fluid.dygraph.guard(place=place):

       # 准备数据并行模式下的环境配置
       strategy=dygraph.parallel.prepare_context()

       linear = Linear(1, 10, act="softmax")
       adam = fluid.optimizer.AdamOptimizer(parameter_list=linear.parameters())

       # 使用户的模型linear变成数据并行模式下的模型
       linear = dygraph.parallel.DataParallel(linear, strategy)

       x_data = np.random.random(size=[10, 1]).astype(np.float32)
       data = to_variable(x_data)

       hidden = linear(data)
       avg_loss = fluid.layers.mean(hidden)

       # 根据trainers的数量来损失值进行缩放
       avg_loss = linear.scale_loss(avg_loss)

       avg_loss.backward()

       # 对多个trainers下模型的参数梯度进行平均 
       linear.apply_collective_grads()

       adam.minimize(avg_loss)
       linear.clear_gradients()

.. py:method:: scale_loss(loss)

对损失值进行缩放。在数据并行模式下，损失值根据 ``trainers`` 的数量缩放一定的比例；反之，返回原始的损失值。在 ``backward`` 前调用，示例如上。

参数：
    - **loss** (Variable) - 当前模型的损失值

返回：缩放的损失值

返回类型：Variable

.. py:method:: apply_collective_grads()

使用AllReduce模式来计算数据并行模式下多个 ``trainers`` 模型之间参数梯度的均值。在 ``backward`` 之后调用，示例如上。

