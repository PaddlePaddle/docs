快速开始
===========

快速安装
----------

PaddlePaddle支持使用pip快速安装， 执行下面的命令完成CPU版本的快速安装：

.. code-block:: bash

	pip install -U paddlepaddle

如需安装GPU版本的PaddlePaddle，执行下面的命令完成GPU版本的快速安装:

.. code-block:: bash

	pip install -U paddlepaddle-gpu

同时请保证您参考NVIDIA官网，已经正确配置和安装了显卡驱动，`CUDA 9 <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_ ，`cuDNN 7.3 <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，`NCCL2 <https://developer.nvidia.com/nccl/nccl-download/>`_ 等依赖，其他更加详细的安装信息请参考：`安装说明 <https://www.paddlepaddle.org.cn/install/doc/index>`_


快速使用
-------------

首先，您需要导入fluid库

.. code-block:: python

	import paddle.fluid as fluid

* Tensor操作


下面几个简单的案例，可以帮助您快速了解Fluid：

1.使用Fluid创建5个元素的一维数组，其中每个元素都为1

.. code-block:: python
    
	# 定义数组维度及数据类型，可以修改shape参数定义任意大小的数组
	data = fluid.layers.ones(shape=[5], dtype='int64')
	# 在CPU上执行运算
	place = fluid.CPUPlace()
	# 创建执行器
	exe = fluid.Executor(place)
	# 执行计算
	ones_result = exe.run(fluid.default_main_program(),
	                        # 获取数据data
				fetch_list=[data], 
				return_numpy=True)
	# 输出结果
	print(ones_result[0])

可以得到结果：

.. code-block:: text

	[1 1 1 1 1]

2.使用Fluid将两个数组按位相加

.. code-block:: python

	# 调用 elementwise_op 将生成的一维数组按位相加
	add = fluid.layers.elementwise_add(data,data)
	# 定义运算场所
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	# 执行计算
	add_result = exe.run(fluid.default_main_program(),
	                 fetch_list=[add],
	                 return_numpy=True)
	# 输出结果
	print (add_result[0])

可以得到结果：

.. code-block:: text

	[2 2 2 2 2]

3.使用Fluid转换数据类型

.. code-block:: python

	# 将一维整型数组，转换成float64类型
	cast = fluid.layers.cast(x=data, dtype='float64')
	# 定义运算场所执行计算
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	cast_result = exe.run(fluid.default_main_program(),
	                 fetch_list=[cast],
	                 return_numpy=True)
	# 输出结果
	print(cast_result[0])

可以得到结果：

.. code-block:: text

	[1. 1. 1. 1. 1.]


运行线性回归模型
-----------------

通过上面的小例子，相信您已经对如何使用Fluid操作数据有了一定的了解，那么试着创建一个test.py，并粘贴下面的代码吧。

这是一个简单的线性回归模型，来帮助我们快速求解4元一次方程。

.. code-block:: python

	#加载库
	import paddle.fluid as fluid
	import numpy as np
	#生成数据
	np.random.seed(0)
	outputs = np.random.randint(5, size=(10, 4))
	res = []
	for i in range(10):
		# 假设方程式为 y=4a+6b+7c+2d
		y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
		res.append([y])
	# 定义数据
	train_data=np.array(outputs).astype('float32')
	y_true = np.array(res).astype('float32')

	#定义网络
	x = fluid.layers.data(name="x",shape=[4],dtype='float32')
	y = fluid.layers.data(name="y",shape=[1],dtype='float32')
	y_predict = fluid.layers.fc(input=x,size=1,act=None)
	#定义损失函数
	cost = fluid.layers.square_error_cost(input=y_predict,label=y)
	avg_cost = fluid.layers.mean(cost)
	#定义优化方法
	sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
	sgd_optimizer.minimize(avg_cost)
	#参数初始化
	cpu = fluid.CPUPlace()
	exe = fluid.Executor(cpu)
	exe.run(fluid.default_startup_program())
	##开始训练，迭代500次
	for i in range(500):
		outs = exe.run(
			feed={'x':train_data,'y':y_true},
			fetch_list=[y_predict.name,avg_cost.name])
		if i%50==0:
			print ('iter={:.0f},cost={}'.format(i,outs[1][0]))
	#存储训练结果
	params_dirname = "result"
	fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

	# 开始预测
	infer_exe = fluid.Executor(cpu)
	inference_scope = fluid.Scope()
	# 加载训练好的模型
	with fluid.scope_guard(inference_scope):
		[inference_program, feed_target_names,
		 fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)

	# 生成测试数据
	test = np.array([[[9],[5],[2],[10]]]).astype('float32')
	# 进行预测
	results = infer_exe.run(inference_program,
							feed={"x": test},
							fetch_list=fetch_targets) 
	# 给出题目为 【9,5,2,10】 输出y=4*9+6*5+7*2+10*2的值
	print ("9a+5b+2c+10d={}".format(results[0][0]))

.. code-block:: text

    得到结果：
	
	9a+5b+2c+10d=[99.946]
	
输出结果应是一个近似等于100的值，每次计算结果略有不同。
	
    
	

