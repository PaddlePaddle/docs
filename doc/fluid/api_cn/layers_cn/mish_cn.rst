.. _cn_api_fluid_layers_mish:

mish
-------------------------------

.. py:function:: paddle.fluid.layers.mish(x, threshold=20, name=None)


Mish激活函数。
更多详情请参考 : `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

当 ``threshold`` 为 ``None`` 或负值时：

.. math::

		out = x * \tanh(\ln(1 + e^{x}))

The formula is as follows if :attr:`threshold` is set:
当 ``threshold`` 设置为正值时:

.. math::

		out = \begin{cases}
						x \ast \tanh(x), \text{if } x > \text{threshold} \\
						x \ast \tanh(e^{x}), \text{if } x < -\text{threshold} \\
						x \ast \tanh(\ln(1 + e^{x})),  \text{otherwise}
					\end{cases}

参数:
    - **x** (Variable)- 多维Tensor，数据类型为float32或float64。
    - **threshold** (float|None)- softplus计算中的阈值，当输入值绝对值大于该阈值时，将使用近似值代替softplus计算结果， ``threshold`` 为 ``None`` 或负值时不生效，默认值为20。
    - **name** (str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：表示绝对值结果的Tensor，数据类型与x相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

		import paddle.fluid as fluid
		import numpy as np

		DATATYPE='float32'

		x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

		x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
		y = fluid.layers.mish(x)

		place = fluid.CPUPlace()
		# place = fluid.CUDAPlace(0)
		exe = fluid.Executor(place)
		out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
		print(out)  # [[0.66666667, 1.66666667, 3., 4.]]
