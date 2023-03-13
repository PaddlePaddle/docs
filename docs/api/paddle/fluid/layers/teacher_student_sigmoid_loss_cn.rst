.. _cn_api_fluid_layers_teacher_student_sigmoid_loss:

teacher_student_sigmoid_loss
-----------------------------------

.. py:function:: paddle.fluid.layers.teacher_student_sigmoid_loss(input, label, soft_max_up_bound=15.0, soft_max_lower_bound=-15.0)




**Teacher Student Log Loss Layer（教师--学生对数损失层）**

定制化需求，用于 student 萃取 teacher 的值。此图层接受输入预测和目标标签，并返回 teacher_student 损失。
z 表示是否点击，z'表示 teacher q 值。label 取值范围{-2，-1，[0, 2]}
teacher q 值不存在时，点击时 label 为-1，否则为-2。
teacher q 值存在时，点击时 label 为 z'，否则为 1 + z'。

.. math::

    loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))

其中：
    - :math:`x`：预测输入值。
    - :math:`z`：是否点击。
    - :math:`z'` ：teacher q 值。


参数
::::::::::::

  - **input**  (Variable) – 形状为[N x 1]的 2-d Tensor，其中 N 是批大小 batch size。该输入是由前一个运算计算而得的概率，数据类型为 float32 或者 float64。
  - **label**  (Variable) – 具有形状[N x 1]的 2-d Tensor 的真实值，其中 N 是批大小 batch_size，数据类型为 float32 或者 float64。
  - **soft_max_up_bound**  (float) – 若 input > soft_max_up_bound，输入会被向下限制。默认为 15.0 。
  - **soft_max_lower_bound**  (float) – 若 input < soft_max_lower_bound，输入将会被向上限制。默认为-15.0 。

返回
::::::::::::
具有形状[N x 1]的 2-D Tensor，teacher_student_sigmoid_loss。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.teacher_student_sigmoid_loss
