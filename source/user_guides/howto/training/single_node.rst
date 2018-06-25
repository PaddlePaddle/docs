########
单机训练
########

准备工作
########

要进行PaddlePaddle Fluid单机训练，需要先 :ref:`user_guide_prepare_data` 和
:ref:`user_guide_config_neural_network` 。当
:ref:`user_guide_config_neural_network` 完毕后，可以得到两个
:ref:`api_fluid_Program`， :code:`startup_program` 和 :code:`main_program`。
默认情况下，可以使用 :ref:`api_fluid_default_startup_program` 与 :ref:`api_fluid_default_main_program` 获得全局的 :ref:`api_fluid_Program`。

例如:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[784])
   label = fluid.layers.data(name="label", shape=[1])
   hidden = fluid.layers.fc(input=image, size=100, act='relu')
   prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
   loss = fluid.layers.mean(
       fluid.layers.cross_entropy(
           input=prediction,
           label=label
       )
   )

   sgd = fluid.optimizer.SGD(learning_rate=0.001)
   sgd.minimize(loss)

   # Here the fluid.default_startup_program() and fluid.default_main_program()
   # has been constructed.

在上述模型配置执行完毕后， :code:`fluid.default_startup_program()` 与
:code:`fluid.default_main_program()` 配置完毕了。

初始化参数
##########

参数随机初始化
==============

载入预定义参数
==============

在神经网络训练过程中，经常会需要载入预定义模型，进而继续进行训练。
如何载入预定义参数，请参考 :ref:`user_guide_save_load_vars`。


单卡训练
########


多卡训练
########


边训练边测试
############


进阶使用
########

