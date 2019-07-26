.. _cn_api_fluid_layers_dropout:

dropout
-------------------------------

.. py:function:: paddle.fluid.layers.dropout(x,dropout_prob,is_test=False,seed=None,name=None,dropout_implementation='downgrade_in_infer')

dropout操作

丢弃或者保持x的每个元素独立。Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。根据给定的丢弃概率dropout操作符随机将一些神经元输出设置为0，其他的仍保持不变。

dropout op可以从Program中删除，提高执行效率。

参数：
    - **x** (Variable)-输入张量
    - **dropout_prob** (float)-设置为0的单元的概率
    - **is_test** (bool)-显示是否进行测试用语的标记
    - **seed** (int)-Python整型，用于创建随机种子。如果该参数设为None，则使用随机种子。注：如果给定一个整型种子，始终丢弃相同的输出单元。训练过程中勿用固定不变的种子。
    - **name** (str|None)-该层名称（可选）。如果设置为None,则自动为该层命名
    - **dropout_implementation** (string) -

      [‘downgrade_in_infer’(default)|’upscale_in_train’] 其中:

      1. downgrade_in_infer(default), 在预测时减小输出结果

         - train: out = input * mask

         - inference: out = input * (1.0 - dropout_prob)

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` )

      2. upscale_in_train, 增加训练时的结果

         - train: out = input * mask / ( 1.0 - dropout_prob )

         - inference: out = input

         (mask是一个张量，维度和输入维度相同，值为0或1，值为0的比例即为 ``dropout_prob`` ）

dropout操作符可以从程序中移除，程序变得高效。

返回：与输入X，shape相同的张量

返回类型：变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    droped = fluid.layers.dropout(x, dropout_prob=0.5)









