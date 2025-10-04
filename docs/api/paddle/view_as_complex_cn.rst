.. _cn_api_paddle_view_as_complex:

view_as_complex
-------------------------------

.. py:function:: paddle.view_as_complex(input)


获取实数张量 ``input`` 的一个视图，将其视为一个复数张量。

输入 Tensor 的数据类型是 'float32' 或者 'float64'，输出 Tensor 的数据类型相应为 'complex64' 或者 'complex128'。

输入 Tensor 的形状是 ``(*, 2)`` (其中 ``*`` 表示任意形状)，亦即，输入的最后一维的大小必须是 2，这对应着复数的实部和虚部。输出 Tensor 的形状是 ``(*,)``。

下图展示了一个 view_as_complex 简单的使用情形。一个形状为[2, 3, 2]的实数 Tensor 经过 view_as_complex 转换，最后一个长度为 2 的维度被合并为复数，形状变为[2, 3]。

返回的复数张量是输入实数张量的一个视图，这意味着二者共享同一块内存区域。

.. figure:: ../../images/api_legend/as_complex.png
   :alt: view_as_complex 图示
   :width: 500
   :align: center
参数
:::::::::
    - **input** (Tensor) - 输入 Tensor，数据类型为：float32 或 float64。

返回
:::::::::
输出 Tensor，数据类型是 'complex64' 或 'complex128'，与 ``input`` 的数值精度一致。

代码示例
:::::::::

COPY-FROM: paddle.view_as_complex
