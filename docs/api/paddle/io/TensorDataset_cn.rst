.. _cn_api_io_cn_TensorDataset:

TensorDataset
-------------------------------

.. py:class:: paddle.io.TensorDataset

由张量列表定义的数据集。

每个张量的形状应为[N，...]，而N是样本数，每个张量表示样本中一个字段，TensorDataset中通过在第一维索引张量来获取每个样本。

参数:
    - **tensors** (list of Tensors) - Tensor列表，这些Tensor的第一维形状相同

返回：由张量列表定义的数据集

返回类型: Dataset

**代码示例**

.. code-block:: python

		import numpy as np
		import paddle
		from paddle.io import TensorDataset


		input_np = np.random.random([2, 3, 4]).astype('float32')
		input = paddle.to_tensor(input_np)
		label_np = np.random.random([2, 1]).astype('int32')
		label = paddle.to_tensor(label_np)

		dataset = TensorDataset([input, label])

		for i in range(len(dataset)):
				input, label = dataset[i]
				print(input, label)

