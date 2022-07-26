.. _cn_api_fluid_layers_teacher_student_sigmoid_loss:

teacher_student_sigmoid_loss
-----------------------------------

.. py:function:: paddle.fluid.layers.teacher_student_sigmoid_loss(input, label, soft_max_up_bound=15.0, soft_max_lower_bound=-15.0)




**Teacher Student Log Loss Layer（教师--学生对数损失层）**

定制化需求，用于student萃取teacher的值。此图层接受输入预测和目标标签，并返回teacher_student损失。
z表示是否点击，z'表示teacher q值。label取值范围{-2，-1，[0, 2]}
teacher q值不存在时，点击时label为-1，否则为-2。
teacher q值存在时，点击时label为z'，否则为1 + z'。

.. math::

    loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))

其中：
    - :math:`x`：预测输入值。
    - :math:`z`：是否点击。
    - :math:`z'` ：teacher q值。


参数
::::::::::::

  - **input**  (Variable) – 形状为[N x 1]的2-d Tensor，其中N是批大小batch size。该输入是由前一个运算计算而得的概率，数据类型为float32或者float64。
  - **label**  (Variable) – 具有形状[N x 1]的2-d Tensor的真实值，其中N是批大小batch_size，数据类型为float32或者float64。
  - **soft_max_up_bound**  (float) – 若input > soft_max_up_bound，输入会被向下限制。默认为15.0 。
  - **soft_max_lower_bound**  (float) – 若input < soft_max_lower_bound，输入将会被向上限制。默认为-15.0 。

返回
::::::::::::
具有形状[N x 1]的2-D Tensor，teacher_student_sigmoid_loss。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.teacher_student_sigmoid_loss