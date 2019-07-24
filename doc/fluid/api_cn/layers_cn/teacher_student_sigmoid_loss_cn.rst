.. _cn_api_fluid_layers_teacher_student_sigmoid_loss:

teacher_student_sigmoid_loss
-----------------------------------

.. py:function:: paddle.fluid.layers.teacher_student_sigmoid_loss(input, label, soft_max_up_bound=15.0, soft_max_lower_bound=-15.0)

**Teacher Student Log Loss Layer（教师--学生对数损失层）**

此图层接受输入预测和目标标签，并返回teacher_student损失。

.. math::

    loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))


参数：
  - **input**  (Variable|list) – 形状为[N x 1]的二维张量，其中N是批大小batch size。 该输入是由前一个运算计算而得的概率。
  - **label**  (Variable|list) – 具有形状[N x 1]的二维张量的真实值，其中N是批大小batch_size。
  - **soft_max_up_bound**  (float) – 若input > soft_max_up_bound, 输入会被向下限制。默认为15.0
  - **soft_max_lower_bound**  (float) – 若input < soft_max_lower_bound, 输入将会被向上限制。默认为-15.0

返回：具有形状[N x 1]的2-D张量，teacher_student_sigmoid_loss。

返回类型：变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    batch_size = 64
    label = fluid.layers.data(
              name="label", shape=[batch_size, 1], dtype="int64", append_batch_size=False)
    similarity = fluid.layers.data(
              name="similarity", shape=[batch_size, 1], dtype="float32", append_batch_size=False)
    cost = fluid.layers.teacher_student_sigmoid_loss(input=similarity, label=label)


