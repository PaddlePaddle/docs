Quick Start
=============

Quick Installation
--------------------

PaddlePaddle supports quick installation by pip. Execute the following commands to finish quick installation of the CPU version:

.. code-block:: bash

	pip install paddlepaddle

If you need to install the GPU version, or look up more specific installation methods, please refer to `Installation Instructions <../beginners_guide/install/index_en.html>`_


Quick Usage
-------------

First, you need to import the fluid library

.. code-block:: python

	import paddle.fluid as fluid

* Tensor Operations


The following simple examples may help you quickly know about Fluid:

1.use Fluid to create a one-dimensional array with five elements, and each element is 1

.. code-block:: python
    
	# define the dimension of an array and the data type, and the parameter 'shape' can be modified to define an array of any size
	data = fluid.layers.ones(shape=[5], dtype='int64')
	# compute on the CPU
	place = fluid.CPUPlace()
	# create executors
	exe = fluid.Executor(place)
	# execute computation
	ones_result = exe.run(fluid.default_main_program(),
	                        # get data
				fetch_list=[data], 
				return_numpy=True)
	# output the results
	print(ones_result[0])

you can get the results:

.. code-block:: text

	[1 1 1 1 1]

2.use Fluid to add two arrays by bits

.. code-block:: python

	# call elementwise_op to add the generative arrays by bits
	add = fluid.layers.elementwise_add(data,data)
	# define computation place
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	# execute computation
	add_result = exe.run(fluid.default_main_program(),
	                 fetch_list=[add],
	                 return_numpy=True)
	# output the results
	print (add_result[0])

you can get the results:

.. code-block:: text

	[2 2 2 2 2]

3.use Fluid to transform the data type

.. code-block:: python

	# transform a one-dimentional array of int to float64
	cast = fluid.layers.cast(x=data, dtype='float64')
	# define computation place to execute computation
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	cast_result = exe.run(fluid.default_main_program(),
	                 fetch_list=[cast],
	                 return_numpy=True)
	# output the results
	print(cast_result[0])

you can get the results:

.. code-block:: text

	[1. 1. 1. 1. 1.]


Operate the Linear Regression Model
-------------------------------------

By the simple example above, you may have known how to operate data with Fluid to some extent, so please try to create a test.py, and copy the following codes.

This a a simple linear regression model to help us quickly solve the quaternary linear equation.

.. code-block:: python

	#load the library
	import paddle.fluid as fluid
	import numpy as np
	#generate data
	np.random.seed(0)
	outputs = np.random.randint(5, size=(10, 4))
	res = []
	for i in range(10):
		# assume the equation is y=4a+6b+7c+2d
		y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
		res.append([y])
	# define data
	train_data=np.array(outputs).astype('float32')
	y_true = np.array(res).astype('float32')

	#define the network
	x = fluid.layers.data(name="x",shape=[4],dtype='float32')
	y = fluid.layers.data(name="y",shape=[1],dtype='float32')
	y_predict = fluid.layers.fc(input=x,size=1,act=None)
	#define loss function
	cost = fluid.layers.square_error_cost(input=y_predict,label=y)
	avg_cost = fluid.layers.mean(cost)
	#define optimization methods
	sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
	sgd_optimizer.minimize(avg_cost)
	#initialize parameters
	cpu = fluid.CPUPlace()
	exe = fluid.Executor(cpu)
	exe.run(fluid.default_startup_program())
	##start training and iterate for 500 times
	for i in range(500):
		outs = exe.run(
			feed={'x':train_data,'y':y_true},
			fetch_list=[y_predict.name,avg_cost.name])
		if i%50==0:
			print ('iter={:.0f},cost={}'.format(i,outs[1][0]))
	#save the training result
	params_dirname = "result"
	fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

	# start inference
	infer_exe = fluid.Executor(cpu)
	inference_scope = fluid.Scope()
	# load the trained model
	with fluid.scope_guard(inference_scope):
		[inference_program, feed_target_names,
		 fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)

	# generate test data
	test = np.array([[[9],[5],[2],[10]]]).astype('float32')
	# inference
	results = infer_exe.run(inference_program,
							feed={"x": test},
							fetch_list=fetch_targets) 
	# give the problem 【9,5,2,10】 and output the value of y=4*9+6*5+7*2+10*2
	print ("9a+5b+2c+10d={}".format(results[0][0]))

.. code-block:: text

    get the result:
	
	9a+5b+2c+10d=[99.946]
	
The output result should be a value close to 100, which may have a few errors every time.
	
    
	

