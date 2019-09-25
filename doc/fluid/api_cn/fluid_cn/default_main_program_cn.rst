.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()


此接口可以获取当前用于存储op和variable描述信息的 ``default main program``

``fluid.layers`` 接口中添加的op和variable会存储在 ``default main program`` 中

``default main program`` 是fluid的许多编程接口中Program参数的默认值。例如对于 ``Executor.run()`` 如果用户没有传入Program参数，会默认使用 ``default main program`` 

可以使用 :ref:`cn_api_fluid_program_guard` 来替换 ``default main program`` 

参数: 
    - 无

返回： 当前默认用于存储op和variable描述的Program

返回类型： :ref:`cn_api_fluid_Program`

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
     
        #示例网络:
        data = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    
        conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None)
        bn1 = fluid.layers.batch_norm(conv1, act='relu')
        pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
        conv2 = fluid.layers.conv2d(pool1, 16, 5, 1, act=None)
        bn2 = fluid.layers.batch_norm(conv2, act='relu')
        pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)
    
        fc1 = fluid.layers.fc(pool2, size=50, act='relu')
        fc2 = fluid.layers.fc(fc1, size=102, act='softmax')
     
        loss = fluid.layers.cross_entropy(input=fc2, label=label)
        loss = fluid.layers.mean(loss)
        opt = fluid.optimizer.Momentum(
            learning_rate=0.1,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
        opt.minimize(loss)
     
        print(fluid.default_main_program().num_blocks)
        print(fluid.default_main_program().blocks[0].var('image'))



