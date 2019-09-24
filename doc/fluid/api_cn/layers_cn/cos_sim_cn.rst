.. _cn_api_fluid_layers_cos_sim:

cos_sim
-------------------------------

.. py:function:: paddle.fluid.layers.cos_sim(X, Y)

余弦相似度算子（Cosine Similarity Operator）

.. math::

        Out = \frac{X^{T}*Y}{\sqrt{X^{T}*X}*\sqrt{Y^{T}*Y}}

输入X和Y必须具有相同的shape。但是有一个例外：如果输入Y的第一维为1(不同于输入X的第一维度)，在计算它们的余弦相似度之前，Y的第一维度会自动进行广播(broadcast)，以便于匹配输入X的shape。

输入X和Y可以都携带或者都不携带LoD(Level of Detail)信息。但输出和输入X的LoD信息保持一致。

参数：
    - **X** (Variable) - cos_sim操作函数的第一个输入。
    - **Y** (Variable) - cos_sim操作函数的第二个输入。

返回：LoDTensor。输出两个输入的余弦相似度。

返回类型：Variable

**代码示例**：

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
    y = fluid.layers.data(name='y', shape=[1, 7], dtype='float32', append_batch_size=False)
    out = fluid.layers.cos_sim(x, y)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    np_x = np.random.random(size=(3, 7)).astype('float32')
    np_y = np.random.random(size=(1, 7)).astype('float32')
    output = exe.run(feed={"x": np_x, "y": np_y}, fetch_list = [out])
    print(output)



