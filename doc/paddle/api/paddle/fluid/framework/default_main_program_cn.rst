.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()





此接口可以获取当前用于存储OP和Tensor描述信息的 ``default main program``。

``default main program`` 是许多编程接口中Program参数的默认值。例如对于 ``Executor.run()`` 如果用户没有传入Program参数，会默认使用 ``default main program`` 。

可以使用 :ref:`cn_api_fluid_program_guard` 来切换 ``default main program``。 

参数: 
    - 无

返回： 当前默认用于存储OP和Tensor描述的Program。

返回类型： :ref:`cn_api_fluid_Program`

**代码示例**

.. code-block:: python

        import paddle
        
        paddle.enable_static()
        # Sample Network:
        data = paddle.data(name='image', shape=[None, 3, 224, 224], dtype='float32')
        label = paddle.data(name='label', shape=[None, 1], dtype='int64')
        
        conv1 = paddle.static.nn.conv2d(data, 4, 5, 1, act=None)
        bn1 = paddle.static.nn.batch_norm(conv1, act='relu')
        pool1 = paddle.nn.functional.pool2d(bn1, 2, 'max', 2)
        conv2 = paddle.static.nn.conv2d(pool1, 16, 5, 1, act=None)
        bn2 = paddle.static.nn.batch_norm(conv2, act='relu')
        pool2 = paddle.nn.functional.pool2d(bn2, 2, 'max', 2)
        
        fc1 = paddle.static.nn.fc(x=pool2, size=50, activation='relu')
        fc2 = paddle.static.nn.fc(x=fc1, size=102, activation='softmax')
        
        loss = paddle.nn.functional.loss.cross_entropy(input=fc2, label=label)
        loss = paddle.mean(loss)
        opt = paddle.optimizer.Momentum(
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=paddle.regularizer.L2Decay(1e-4))
        opt.minimize(loss)
        
        #print the number of blocks in the program, 1 in this case
        print(paddle.static.default_main_program().num_blocks) # 1
        #print the default_main_program
        print(paddle.static.default_main_program())



