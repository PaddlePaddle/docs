.. _cn_api_fluid_layers_IfElse:

IfElse
-------------------------------

.. py:class:: paddle.fluid.layers.IfElse(cond, name=None)

if-else控制流。

参数：
    - **cond** (Variable)-用于比较的条件
    - **Name** (str,默认为空（None）)-该层名称

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    image = fluid.layers.data(name="X", shape=[2, 5, 5], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    limit = fluid.layers.fill_constant_batch_size_like(
         input=label, dtype='int64', shape=[1], value=5.0)
    cond = fluid.layers.less_than(x=label, y=limit)
    ie = fluid.layers.IfElse(cond)
    with ie.true_block():
        true_image = ie.input(image)
        hidden = fluid.layers.fc(input=true_image, size=100, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)

    with ie.false_block():
        false_image = ie.input(image)
        hidden = fluid.layers.fc(
            input=false_image, size=200, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)
    prob = ie()









