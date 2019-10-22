.. _cn_user_guide_Program:

=======
Program
=======


飞桨（PaddlePaddle，以下简称Paddle）用Program的形式动态描述整个计算过程。这种描述方式，兼具网络结构修改的灵活性和模型搭建的便捷性，在保证性能的同时极大地提高了框架对模型的表达能力。

用户定义Operator会被顺序的放入Program中，在网络搭建过程中，由于不能使用python 的控制流，Paddle通过同时提供分支和循环两类控制流op结构的支持，让用户可以通过组合描述任意复杂的模型。

**顺序执行：**

用户可以使用顺序执行的方式搭建网络：

.. code-block:: python

    x = fluid.data(name='x',shape=[None, 13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)


**条件分支——switch、if else：**

Fluid 中有 switch 和 if-else 类来实现条件选择，用户可以使用这一执行结构在学习率调节器中调整学习率或其他希望的操作：

.. code-block:: python
    lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")

    one_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.0)
    two_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.0)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.tensor.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.tensor.assign(input=two_var, output=lr)



关于 Padldle 中 Program 的详细设计思想，可以参考阅读 `Fluid设计思想 <../../advanced_usage/design_idea/fluid_design_idea.html>` 。

更多 Paddle 中的控制流，可以参考阅读 `API文档 <../../api_cn/layers_cn/layers_cn.html#control-flow>` 。
