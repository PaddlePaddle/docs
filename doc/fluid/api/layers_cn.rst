

.. _cn_api_fluid_layers_create_array:

create_array
>>>>>>>>>>>>

paddle.fluid.layers.create_array(dtype)
""""""""""""""""""""""""""""""""""""""""""

create_array

paddle.fluid.layers.create_array(dtype)

创建LoDTensorArray数组。它主要用于实现RNN与array_write, array_read和While。

  参数:dtype(int |float)——lod_tensor_array中元素的数据类型。

  返回: lod_tensor_array，数组元素类型为dtype。

  返回类型: Variable。

::
	
	Given:

	array = [0.6, 0.1, 0.3, 0.1]

	And:

	i = 2

	Then:

	output = 0.3
	

参数：  
		- array (Variable|list)：待读取的输入张量（Tensor）
		- i (Variable|list)：待读取的输入数据索引

返回：	张量（Tensor）类型的变量，储存事前写入的数据

返回类型:	变量（variable）


**代码示例**

..  code-block:: python
  
  data = fluid.layers.create_array(dtype='float32')
  
  
