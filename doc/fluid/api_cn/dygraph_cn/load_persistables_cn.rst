.. _cn_api_fluid_dygraph_load_persistables:

load_persistables
-------------------------------

.. py:function:: paddle.fluid.dygraph.load_persistables(dirname='save_dir')

该接口从指定目录加载模型参数以及优化器学习率状态。其中加载优化器学习率，仅加载带衰减的学习率状态，例如学习率类型为`LearningRateDecay`的状态。


参数:
    - **dirname**  (str) – 指定加载的目录路径。默认值为"save_dir"。


返回:
    - load_var_map  – 加载的模型参数字典。该字典键为str类型，值为Parameter类型，其中包含的数据类型可以为float32，float64的多维Tensor。
    - load_optimizer_map  – 加载的优化器的带衰减的学习率字典。该字典键为str类型，值为`LearningRateDecay`类型。

返回类型:   
    - dict of Variable – 模型参数字典。字典键为str类型，值为Variable类型，数据类型可以为float16，float32, .... , int16，int32，....的多维Tensor。
    - dict of LearningRateDecay – 优化器带衰减的学习率对象字典。字典键为str类型，值为LearningRateDecay类型。

  
**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid.optimizer as optimizer
    import paddle.fluid as fluid
    import numpy as np
    from paddle.fluid.dygraph.base import to_variable
    from paddle.fluid.dygraph.nn import FC
    from paddle.fluid.optimizer import SGDOptimizer

    class MLP(fluid.Layer):
        def __init__(self, name_scope):
            super(MLP, self).__init__(name_scope)
            self._fc1 = FC(self.full_name(), 10)
            self._fc2 = FC(self.full_name(), 10)

        def forward(self, inputs):
            y = self._fc1(inputs)
            y = self._fc2(y)
            return y

    train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

    with fluid.dygraph.guard():
        mlp = MLP("mlp")
        sgd = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
            learning_rate=0.1,
            decay_steps=1,
            decay_rate=0.5,
            staircase=True))

        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array(
                [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                128, 1)

            img = to_variable(dy_x_data)
            label = to_variable(y_data)
            label._stop_gradient = True

            cost = mlp(img)
            avg_loss = fluid.layers.reduce_mean(cost)
            avg_loss.backward()

            sgd.minimize(avg_loss)
            sgd.clear_gradient()
            break
        
        save_dir="save_dir"
        fluid.dygraph.save_persistables(mlp.state_dict(), dirname=save_dir, optimizers=sgd)
        para_dict, opt_dict = fluid.dygraph.load_persistables(dirname=save_dir)
