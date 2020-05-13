.. _cn_api_fluid_layers_dropout:

dropout
-------------------------------

.. py:function:: paddle.fluid.layers.dropout(x,dropout_prob,is_test=False,seed=None,name=None,dropout_implementation='downgrade_in_infer')

:alias_main: paddle.nn.functional.dropout
:alias: paddle.nn.functional.dropout,paddle.nn.functional.common.dropout
:old_api: paddle.fluid.layers.dropout



dropout操作

丢弃或者保持x的每个元素独立。Dropout是一种正则化手段，通过在训练过程中阻止神经元节点间的相关性来减少过拟合。根据给定的丢弃概率，dropout操作符按丢弃概率随机将一些神经元输出设置为0，其他的仍保持不变。

dropout op可以从Program中删除，提高执行效率。

参数：
    - **x** (Variable) - 输入，多维Tensor。数据类型：float32和float64。
    - **dropout_prob** (float32) - 输入单元的丢弃概率，即输入单元设置为0的概率。
    - **is_test** (bool) - 标记是否是测试阶段。默认：False。
    - **seed** (int) - 整型数据，用于创建随机种子。如果该参数设为None，则使用随机种子。注：如果给定一个整型种子，始终丢弃相同的输出单元。训练过程中勿用固定不变的种子。
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。
    - **dropout_implementation** (str) - 丢弃单元的方式，有两种'downgrade_in_infer'和'upscale_in_train'两种选择，默认：'downgrade_in_infer'。具体作用可以参考一下描述。

      1. downgrade_in_infer(default), 在预测时减小输出结果

         - train: out = input * mask

         - inference: out = input * (1.0 - dropout_prob)

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` )

      2. upscale_in_train, 增加训练时的结果

         - train: out = input * mask / ( 1.0 - dropout_prob )

         - inference: out = input

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` ）

dropout操作符可以从程序中移除，使程序变得高效。

返回：Tensor。经过丢弃部分数据之后的结果，与输入X形状相同的张量。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    x = fluid.layers.data(name="x", shape=[32, 32], dtype="float32")
    droped = fluid.layers.dropout(x, dropout_prob=0.5)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    np_x = np.random.random(size=(32, 32)).astype('float32')
    output = exe.run(feed={"x": np_x}, fetch_list = [droped])
    print(output)

