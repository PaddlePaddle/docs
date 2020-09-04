Quick Start
=================

**前提准备**
接下来我们会通过几段Python代码的方式对Paddle Inference使用进行介绍，
为了能够成功运行代码，请您在环境中（Mac， Windows，Linux）安装不低于1.7版本的Paddle，
安装Paddle 请参考 `飞桨官网主页 <https://www.paddlepaddle.org.cn/>`_。

导出预测模型文件
----------------

在模型训练期间，我们通常使用Python来构建模型结构，比如：

.. code:: python

	import paddle.fluid as fluid
	res = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)

在模型部署时，我们需要提前将这种Python表示的结构以及参数序列化到磁盘中。那是如何做到的呢？

在模型训练过程中或者模型训练结束后，我们可以通过save_inference_model接口来导出标准化的模型文件。    

我们用一个简单的代码例子来展示下导出模型文件的这一过程。


.. code:: python

	import paddle
	import paddle.fluid as fluid
	# 建立一个简单的网络，网络的输入的shape为[batch, 3, 28, 28]
	image_shape = [3, 28, 28]

	img = fluid.layers.data(name='image', shape=image_shape, dtype='float32', append_batch_size=True)
	# 模型包含两个Conv层
	conv1 = fluid.layers.conv2d(
		input=img,
		num_filters=8,
		filter_size=3,
		stride=2,
		padding=1,
		groups=1,
		act=None,
		bias_attr=True)

	out = fluid.layers.conv2d(
		input=conv1,
		num_filters=8,
		filter_size=3,
		stride=2,
		padding=1,
		groups=1,
		act=None,
		bias_attr=True)

	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	# 创建网络中的参数变量，并初始化参数变量
	exe.run(fluid.default_startup_program())

	# 如果存在预训练模型
	# def if_exist(var):
	#            return os.path.exists(os.path.join("./ShuffleNet", var.name))
	#    fluid.io.load_vars(exe, "./pretrained_model", predicate=if_exist)
	# 保存模型到model目录中，只保存与输入image和输出与推理相关的部分网络
	fluid.io.save_inference_model(dirname='./sample_model', feeded_var_names=['image'], target_vars = [out], executor=exe, model_filename='model', params_filename='params')

该程序运行结束后，会在本目录中生成一个sample_model目录，目录中包含model, params 两个文件，model文件表示模型的结构文件，params表示所有参数的融合文件。 


飞桨提供了 **两种标准** 的模型文件，一种为Combined方式， 一种为No-Combined的方式。

- Combined的方式

.. code:: python

	fluid.io.save_inference_model(dirname='./sample_model', feeded_var_names=['image'], target_vars = [out], executor=exe, model_filename='model', params_filename='params')

model_filename，params_filename表示要生成的模型结构文件、融合参数文件的名字。


* No-Combined的方式  

.. code:: python

	fluid.io.save_inference_model(dirname='./sample_model', feeded_var_names=['image'], target_vars = [out], executor=exe)

如果不指定model_filename，params_filename，会在sample_model目录下生成__model__ 模型结构文件，以及一系列的参数文件。


在模型部署期间，**我们更推荐使用Combined的方式**，因为涉及模型上线加密的场景时，这种方式会更友好一些。



加载模型预测
----------------

1）使用load_inference方式

我们可以使用load_inference_model接口加载训练好的模型（以sample_model模型举例），并复用训练框架的前向计算，直接完成推理。
示例程序如下所示：

.. code:: python

	import paddle.fluid as fluid
	import numpy as np

	data = np.ones((1, 3, 28, 28)).astype(np.float32)
	exe = fluid.Executor(fluid.CPUPlace())

	# 加载Combined的模型需要指定model_filename, params_filename
	# 加载No-Combined的模型不需要指定model_filename, params_filename
	[inference_program, feed_target_names, fetch_targets] = \
		fluid.io.load_inference_model(dirname='sample_model', executor=exe, model_filename='model', params_filename='params')

	with fluid.program_guard(inference_program):
	results = exe.run(inference_program,
		feed={feed_target_names[0]: data},
		fetch_list=fetch_targets, return_numpy=False)

	print (np.array(results[0]).shape)
	# (1, 8, 7, 7)

在上述方式中，在模型加载后会按照执行顺序将所有的OP进行拓扑排序，在运行期间Op会按照排序一一运行，整个过程中运行的为训练中前向的OP，期间不会有任何的优化（OP融合，显存优化，预测Kernel针对优化）。 因此，load_inference_model的方式预测期间很可能不会有很好的性能表现，此方式比较适合用来做实验（测试模型的效果、正确性等）使用，并不适用于真正的部署上线。接下来我们会重点介绍Paddle Inference的使用。

2）使用Paddle Inference API方式

不同于 load_inference_model方式，Paddle Inference 在模型加载后会进行一系列的优化，包括： Kernel优化，OP横向，纵向融合，显存/内存优化，以及MKLDNN，TensorRT的集成等，性能和吞吐会得到大幅度的提升。这些优化会在之后的文档中进行详细的介绍。

那我们先用一个简单的代码例子来介绍Paddle Inference 的使用。

.. code::

	from paddle.fluid.core import AnalysisConfig
	from paddle.fluid.core import create_paddle_predictor

	import numpy as np

	# 配置运行信息
	# config = AnalysisConfig("./sample_model") # 加载non-combined 模型格式
	config = AnalysisConfig("./sample_model/model", "./sample_model/params") # 加载combine的模型格式

	config.switch_use_feed_fetch_ops(False)
	config.enable_memory_optim()
	config.enable_use_gpu(1000, 0)

	# 根据config创建predictor
	predictor = create_paddle_predictor(config)

	img = np.ones((1, 3, 28, 28)).astype(np.float32)

	# 准备输入
	input_names = predictor.get_input_names()
	input_tensor = predictor.get_input_tensor(input_names[0])
	input_tensor.reshape(img.shape)   
	input_tensor.copy_from_cpu(img.copy())

	# 运行
	predictor.zero_copy_run()

	# 获取输出
	output_names = predictor.get_output_names()
	output_tensor = predictor.get_output_tensor(output_names[0])
	output_data = output_tensor.copy_to_cpu()

	print (output_data)

上述的代码例子，我们通过加载一个简答模型以及随机输入的方式，展示了如何使用Paddle Inference进行模型预测。可能对于刚接触Paddle Inferenece同学来说，代码中会有一些陌生名词出现，比如AnalysisConfig, Predictor 等。先不要着急，接下来的文章中会对这些概念进行详细的介绍。 


**相关链接**

`Python API 使用介绍 <../user_guides/inference_python_api.html>`_
`C++ API使用介绍 <../user_guides/cxx_api.html>`_
`Python 使用样例 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python>`_
`C++ 使用样例 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B>`_

