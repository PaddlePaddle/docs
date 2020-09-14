.. _cn_api_fluid_dygraph_Dropout:

Dropout
-------------------------------

.. py:class:: paddle.fluid.dygraph.Dropout(p=0.5, seed=None, dropout_implementation='downgrade_in_infer', is_test=False)

丢弃或者保持输入的每个元素独立。Dropout是一种正则化手段，通过在训练过程中阻止神经元节点间的相关性来减少过拟合。根据给定的丢弃概率，dropout操作符按丢弃概率随机将一些神经元输出设置为0，其他的仍保持不变。

Dropout层可以删除，提高执行效率。

参数：
    - **p** (float32，可选) - 输入单元的丢弃概率，即输入单元设置为0的概率。默认值：0.5
    - **seed** (int，可选) - 整型数据，用于创建随机种子。如果该参数设为None，则使用随机种子。注：如果给定一个整型种子，始终丢弃相同的输出单元。训练过程中勿用固定不变的种子。默认值：None。
    - **dropout_implementation** (str，可选) - 丢弃单元的方式，有两种'downgrade_in_infer'和'upscale_in_train'两种选择，默认：'downgrade_in_infer'。具体作用可以参考一下描述。

      1. downgrade_in_infer(default), 在预测时减小输出结果

         - train: out = input * mask

         - inference: out = input * (1.0 - p)

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``p`` )

      2. upscale_in_train, 增加训练时的结果

         - train: out = input * mask / ( 1.0 - p )

         - inference: out = input

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``p`` ）

    - **is_test** (bool，可选) - 标记是否是测试阶段。此标志仅对静态图模式有效。对于动态图模式，请使用 ``eval()`` 接口。默认：False。

返回：无

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np

    x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
    with fluid.dygraph.guard():
        x = to_variable(x)
        m = fluid.dygraph.Dropout(p=0.5)
        droped_train = m(x)
        # 切换到 eval 模式
        m.eval()
        droped_eval = m(x)

