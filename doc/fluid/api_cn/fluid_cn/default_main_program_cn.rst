.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.fluid.default_main_program()





此函数用于获取默认或全局main program(主程序)。该主程序用于训练和测试模型。

``fluid.layers`` 中的所有layer函数可以向 ``default_main_program`` 中添加operators（算子）和variables（变量）。

``default_main_program`` 是fluid的许多编程接口（API）的Program参数的缺省值。例如,当用户program没有传入的时候，
``Executor.run()`` 会默认执行 ``default_main_program`` 。


返回： main program

返回类型: Program

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    # Sample Network:
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
     
    print(fluid.default_main_program())







