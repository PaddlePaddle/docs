.. _cn_api_fluid_dygraph_save_persistables:

save_persistables
-------------------------------

.. py:function:: paddle.fluid.dygraph.save_persistables(model_dict, dirname='save_dir', optimizers=None)

该接口将模型所有参数以及优化器状态保存到指定目录。优化器状态目前仅支持保存带衰减的学习率状态，即`LearningRateDecay`的状态。

参数:
 - **model_dict**  (dict of Parameter) – 需要保存的模型参数字典，该字典键为str类型，值为Parameter类型，其中包含了数据类型可以为float32，float64的多维Tensor。
 - **dirname**  (str) – 指定保存的目录名。默认值为"save_dir"。
 - **optimizers**  (fluid.Optimizer|list(fluid.Optimizer)，可选) –  可以设定为单个优化器，也可以设定为由多个优化器构成的列表。保存优化器中带衰减的学习率参数。默认值为None。 
 
返回:  无
  
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
            sgd.minimize(avg_loss)该函数把传入的层中所有参数以及优化器进行保存。

