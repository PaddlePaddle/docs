# DyGraph

DyGraph mode of PaddlePaddle is a dynamic mechanism for the execution of graphs, which can execuate the result immediately without constructing the whole graph. Meanwhile, different from the static execution of graphs before, in the DyGraph mode, you can get the execution result once you operate rather than construct the fixed computation graph to execute.Under these conditions, you can build a deep learning assignment and debug the model with  PaddlePaddle more intuitively. What's more, mass codes for constructing the static computation graph are reduced, so that the process of establishing networks more convenient and debugging easier.

PaddlePaddle DyGraph is a more flexible and easily accessible mode, which can provide:      

*	More flexible and convenient organizational structure of codes: use the execution and control process and the object-oriented model design of python.


* 	More convenient debugging function:call operations directly to examine the running model and test the changes


*   Common model codes with static execution graph: you can debug and execute the same model codes with the more convenient DyGraph or the original static graph mode.


*   Support layer realized by pure Python and Numpy grammers: support to construct the model computation part directly with Numpy's related operations

## Installation and Basic Usage

1. Upgrade to the newest PaddlePaddle 1.4:		
		
		pip install -q --upgrade paddlepaddle==1.4

2. Use `fluid.dygraph.guard(place=None)` contexts:
		
		import paddle.fluid as fluid
		with fluid.dygraph.guard():
			# write your executable dygraph code here             
			 
	Now you can run the network with DyGraph mode in the contexts of `fluid.dygraph.guard()` , and DyGraph will change the previous executiong way of PaddlePaddle: now they will execute immediately and return the computation result to Python.


	It's very appropriate to use Dygraph with Numpy. You can transform ndarray to `fluid.Variable` by `fluid.dygraph.base.to_variable(x)` and transform the computation result gotten at any time to Numpy `ndarray` by `fluid.Variable.numpy()` :       
	
			x = np.ones([2, 2], np.float32)
			with fluid.dygraph.guard():
		        inputs = []
		        for _ in range(10):
		            inputs.append(fluid.dygraph.base.to_variable(x))
		        ret = fluid.layers.sums(inputs)
		        print(ret.numpy())
					
			                        
			[[10. 10.]
			[10. 10.]]

			Process finished with exit code 0


	>	A sequence of `ndarray` inputs are created here,and we can print the running results
	
	Then execute the back propagation by `Variable.backward()` method after calling `reduce_sum`. And you can get the gradient value in the form of `ndarray` after the back network's execution by `Variable.gradient()` method:

		loss = fluid.layers.reduce_sum(ret)
		loss.backward()
		print(loss.gradient())
		
		
		
		[1.]

		Process finished with exit code 0



<!--3. 使用Python和Numpy的操作来构建一个网络：

	首先定义了一个继承自`fluid.PyLayer`的`MyPyLayer`:

	这个类需要实现：

	1. 一个调用基类方法的用于初始化网络参数和结构的`__init__`方法
	2. 一个用于在实际运行时实现前向逻辑的静态方法`forward(*inputs)`
	3. 一个用于在实际运行时实现反向逻辑的静态方法`backward(*douts)`


			class MyPyLayer(fluid.PyLayer):
			    def __init__(self):
			        super(MyPyLayer, self).__init__()
			
			    @staticmethod
			    def forward(inputs):
			        return np.tanh(inputs[0])
			
			    @staticmethod
			    def backward(inputs):
			        inp, out, dout = inputs
			        return np.array(dout) * (1 - np.square(np.array(out))

	然后在`fluid.dygraph.guard()`中使用类似`2`中的方法调用`my_py_layer`执行这个callable object来执行:

			np_inp = np.ones([2, 2], np.float32)
        	with fluid.dygraph.guard():
	            my_py_layer = MyPyLayer()
	            var_inp = fluid.dygraph.base.to_variable(np_inp)
	            outs = my_py_layer(var_inp)
	            dy_out = np.sum(outs[0].numpy())
	            outs[0].backward()
	            dy_grad = var_inp.gradient()

	>  请注意，继承自`fluid.PyLayer`的网络和继承自`fluid.Layer`的网络暂时不可混用	
-->
## Bulid a Network based on DyGraph
		
1. Write a Object-Oriented-Designed for the execution of DyGraph. PaddlePaddle model codes are mainly constituted by the following **Three Parts** : **Please pay attention to the fact that if the designed layer includes parameters,you have to describe the behaviors of the layer by Object-Oriented-Designed classes inherited from `fluid.Layer`**

	
	1. When building an Object-Oriented network that can execute in DyGraph mode, inheritance from `fluid.Layer` are needed. During this period, you should call the `__init__` method of base classes and realize the `__init__` construction function of `name_scope` (for identifying the name of this layer) with parameters. In construction functions, we usually execute some operations such as parameter initialization and sub-network initialization, which do not depend on the input dynamic information:
		
			class MyLayer(fluid.Layer):
			    def __init__(self, name_scope):
			        super(MyLayer, self).__init__(name_scope)    
			        
	2. Realize a `forward(self, *inputs)` execution function, which is responsible for executing the network's execution logic in the real running process. This function will be called every training/prediction turn. We execute a simple `relu` -> `elementwise add` -> `reduce sum` here:
	
			def forward(self, inputs):
		        x = fluid.layers.relu(inputs)
		        self._x_for_debug = x
		        x = fluid.layers.elementwise_mul(x, x)
		        x = fluid.layers.reduce_sum(x)
		        return [x]       
		        
	3. (Optional)Realize a `build_once(self, *inputs)` method, which executes only once to initialize some parameters and network information that depend on the inputs. For example, in `FC`（fully connected layer), we need to depend on the input `shape` initialization parameter. This operation is unnecessary and only for display, which can be skipped directly:
		
			def build_once(self, input):
		        pass

2. Execute in `fluid.dygraph.guard()`:

	1. Build the inputs by Numpy:
		
			np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)

	2. Transform the inputs and execute the forward network to get the returned value:transform the Numpy input to the input received by DyGraph with `fluid.dygraph.base.to_variable(np_inp)`, call callable object and get `x` as a returned value with `l(var_inp)[0]` , and then get the `ndarray` returned value directly with `x.numpy()` method.


			with fluid.dygraph.guard():
			    var_inp = fluid.dygraph.base.to_variable(np_inp)
			    l = MyLayer("my_layer")
			    x = l(var_inp)[0]
			    dy_out = x.numpy()

	3. Calculate the gradient: the automatic differentiation is very useful for realizing machine learning algorithms(such as back propagation for training the neural network). With `x.backward()` method,the back network can be executed from certain `fluid.Varaible` and the `ndarray` returned value of `x` gradient can be gotten by `l._x_for_debug.gradient()` .
		
			    x.backward()
			    dy_grad = l._x_for_debug.gradient()



## Train model by DyGraph

Next we will take the basic "Handwritten Digit Recognition" model for example to show how we can build and train a model by DyGraph:

Please refer to contents in [PaddleBook](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits)for related knowledge about handwritten digit recognition. We assume that you have masterd the theoretical knowledge of deep learning that the model needs.


1.	Prepare the data. We take `paddle.dataset.mnist` as the data set that our traning needs:
		
		train_reader = paddle.batch(
		paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

2. Build the network. Although you can define all the network structures yourself according to the previous introduction, you can also use the basic network structures we have customized for you in `fluid.Layer.nn` directly. We build basic `SimpleImgConvPool` by `fluid.Layer.nn.Conv2d` and `fluid.Layer.nn.Pool2d` here:

		class SimpleImgConvPool(fluid.dygraph.Layer):
		    def __init__(self,
		                 name_scope,
		                 num_channels,
		                 num_filters,
		                 filter_size,
		                 pool_size,
		                 pool_stride,
		                 pool_padding=0,
		                 pool_type='max',
		                 global_pooling=False,
		                 conv_stride=1,
		                 conv_padding=0,
		                 conv_dilation=1,
		                 conv_groups=1,
		                 act=None,
		                 use_cudnn=False,
		                 param_attr=None,
		                 bias_attr=None):
		        super(SimpleImgConvPool, self).__init__(name_scope)
		
		        self._conv2d = Conv2D(
		            self.full_name(),
		            num_channels=num_channels,
		            num_filters=num_filters,
		            filter_size=filter_size,
		            stride=conv_stride,
		            padding=conv_padding,
		            dilation=conv_dilation,
		            groups=conv_groups,
		            param_attr=None,
		            bias_attr=None,
		            use_cudnn=use_cudnn)
		
		        self._pool2d = Pool2D(
		            self.full_name(),
		            pool_size=pool_size,
		            pool_type=pool_type,
		            pool_stride=pool_stride,
		            pool_padding=pool_padding,
		            global_pooling=global_pooling,
		            use_cudnn=use_cudnn)
		
		    def forward(self, inputs):
		        x = self._conv2d(inputs)
		        x = self._pool2d(x)
		        return x



	> Attention: When building networks, the definition and usage of the sub-network proceeds in `__init__`, while the call of the sub-network proceeds in `forward`


		       

3. Form the final `MNIST` network by built `SimpleImgConvPool`:

		class MNIST(fluid.dygraph.Layer):
		    def __init__(self, name_scope):
		        super(MNIST, self).__init__(name_scope)
		
		        self._simple_img_conv_pool_1 = SimpleImgConvPool(
		            self.full_name(), 1, 20, 5, 2, 2, act="relu")
		
		        self._simple_img_conv_pool_2 = SimpleImgConvPool(
		            self.full_name(), 20, 50, 5, 2, 2, act="relu")
		
		        pool_2_shape = 50 * 4 * 4
		        SIZE = 10
		        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
		        self._fc = FC(self.full_name(),
		                      10,
		                      param_attr=fluid.param_attr.ParamAttr(
		                          initializer=fluid.initializer.NormalInitializer(
		                              loc=0.0, scale=scale)),
		                      act="softmax")
		
		    def forward(self, inputs):
		        x = self._simple_img_conv_pool_1(inputs)
		        x = self._simple_img_conv_pool_2(x)
		        x = self._fc(x)
		        return x
				  


			
4. Define configured `MNIST` network structure in `fluid.dygraph.guard()` . Then we can call models and examine outputs in `fluid.dygraph.guard()` without training:
	
			with fluid.dygraph.guard():
				mnist = MNIST("mnist")
				id, data = list(enumerate(train_reader()))[0]
				dy_x_data = np.array(
				    [x[0].reshape(1, 28, 28)
				     for x in data]).astype('float32')
				img = to_variable(dy_x_data)
				print("cost is: {}".format(mnist(img).numpy()))
				
				
				
				cost is: [[0.10135901 0.1051138  0.1027941  ... 0.0972859  0.10221873 0.10165327]
				[0.09735426 0.09970362 0.10198303 ... 0.10134517 0.10179105 0.10025002]
				[0.09539858 0.10213123 0.09543551 ... 0.10613529 0.10535969 0.097991  ]
				...
				[0.10120598 0.0996111  0.10512722 ... 0.10067689 0.10088114 0.10071224]
				[0.09889644 0.10033772 0.10151272 ... 0.10245881 0.09878646 0.101483  ]
				[0.09097178 0.10078511 0.10198414 ... 0.10317434 0.10087223 0.09816764]]
					
				Process finished with exit code 0

5. Construct the training circulations, which means we call `mnist.clear_gradients()` to reset the gradient every turn the parameters finish update:

		for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    dy_x_data = np.array(
                        [x[0].reshape(1, 28, 28)
                         for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                    img = to_variable(dy_x_data)
                    label = to_variable(y_data)
                    label.stop_gradient = True

                    cost = mnist(img)
                    loss = fluid.layers.cross_entropy(cost, label)
                    avg_loss = fluid.layers.mean(loss)

                    dy_out = avg_loss.numpy()
                    avg_loss.backward()
                    sgd.minimize(avg_loss)
                    mnist.clear_gradients()




6. Variable and Optimizer

	Parameters of the model or any value you hope to detect can be packaged in a class as variable and can get and use the `ndarray` output by `numpy()` method through objects. In the process of training, you can use `mnist.parameters()` to get all parameters in the network, appoint a parameter of certain `Layer` or use `parameters()` to get all parameters of the layer, and use `numpy()` method to examine parameters' value at any time.

	After back running, call the `minimize` method of the previous defined `SGD` optimizer object to upgrade parameters:
		
		with fluid.dygraph.guard():
		        fluid.default_startup_program().random_seed = seed
		        fluid.default_main_program().random_seed = seed
		
		        mnist = MNIST("mnist")
		        sgd = SGDOptimizer(learning_rate=1e-3)
		        train_reader = paddle.batch(
		            paddle.dataset.mnist.train(), batch_size= BATCH_SIZE, drop_last=True)
		
		        dy_param_init_value = {}
		        np.set_printoptions(precision=3, suppress=True)
		        for epoch in range(epoch_num):
		            for batch_id, data in enumerate(train_reader()):
		                dy_x_data = np.array(
		                    [x[0].reshape(1, 28, 28)
		                     for x in data]).astype('float32')
		                y_data = np.array(
		                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
		
		                img = to_variable(dy_x_data)
		                label = to_variable(y_data)
		                label.stop_gradient = True
		
		                cost = mnist(img)
		                loss = fluid.layers.cross_entropy(cost, label)
		                avg_loss = fluid.layers.mean(loss)
		
		                dy_out = avg_loss.numpy()
		
		                if epoch == 0 and batch_id == 0:
		                    for param in mnist.parameters():
		                        dy_param_init_value[param.name] = param.numpy()
		
		                avg_loss.backward()
		                sgd.minimize(avg_loss)
		                mnist.clear_gradients()
		
		                dy_param_value = {}
		                for param in mnist.parameters():
		                    dy_param_value[param.name] = param.numpy()
		
		                if batch_id % 20 == 0:
		                    print("Loss at step {}: {:.7}".format(batch_id, avg_loss.numpy()))
		        print("Final loss: {:.7}".format(avg_loss.numpy()))
		        print("_simple_img_conv_pool_1_conv2d W's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._filter_param.numpy().mean()))
		        print("_simple_img_conv_pool_1_conv2d Bias's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._bias_param.numpy().mean()))




			Loss at step 0: [2.302]
			Loss at step 20: [1.616]
			Loss at step 40: [1.244]
			Loss at step 60: [1.142]
			Loss at step 80: [0.911]
			Loss at step 100: [0.824]
			Loss at step 120: [0.774]
			Loss at step 140: [0.626]
			Loss at step 160: [0.609]
			Loss at step 180: [0.627]
			Loss at step 200: [0.466]
			Loss at step 220: [0.499]
			Loss at step 240: [0.614]
			Loss at step 260: [0.585]
			Loss at step 280: [0.503]
			Loss at step 300: [0.423]
			Loss at step 320: [0.509]
			Loss at step 340: [0.348]
			Loss at step 360: [0.452]
			Loss at step 380: [0.397]
			Loss at step 400: [0.54]
			Loss at step 420: [0.341]
			Loss at step 440: [0.337]
			Loss at step 460: [0.155]
			Final loss: [0.164]
			_simple_img_conv_pool_1_conv2d W's mean is: 0.00606656912714
			_simple_img_conv_pool_1_conv2d Bias's mean is: -3.4576318285e-05

7.	Performance

	When using `fluid.dygraph.guard()`, you can choose the device for executing DyGraph by introducting `fluid.CUDAPlace(0)` or `fluid.CPUPlace()` . Usually, your device can get matched without disposal.

## Save the Model Parameters

 In model traning, you can use `                    fluid.dygraph.save_persistables(your_model_object.state_dict(), "save_dir")` to save all model parameters in `your_model_object`. And you can define Python Dictionary introduction of "parameter name" - "parameter object" that needs to be saved yourself.

Or use `your_modle_object.load_dict(
                        fluid.dygraph.load_persistables(your_model_object.state_dict(), "save_dir"))` interface to recover saved model parameters to continue training.

The following codes show how to save parameters and read saved parameters to continue training in the "Handwriting Digit Recognition" task.
	
	for epoch in range(epoch_num):
	    for batch_id, data in enumerate(train_reader()):
	        dy_x_data = np.array(
	            [x[0].reshape(1, 28, 28)
	             for x in data]).astype('float32')
	        y_data = np.array(
	            [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
	
	        img = to_variable(dy_x_data)
	        label = to_variable(y_data)
	        label.stop_gradient = True
	
	        cost = mnist(img)
	        loss = fluid.layers.cross_entropy(cost, label)
	        avg_loss = fluid.layers.mean(loss)
	
	        dy_out = avg_loss.numpy()
	
	        avg_loss.backward()
	        sgd.minimize(avg_loss)
	        fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
	        mnist.clear_gradients()
	
	        for param in mnist.parameters():
	            dy_param_init_value[param.name] = param.numpy()
	
	        mnist.load_dict(fluid.dygraph.load_persistables(mnist.state_dict(), "save_dir"))
	        restore = mnist.parameters()
	# check save and load
	success = True
	for value in restore:
	    if (not np.allclose(value.numpy(), dy_param_init_value[value.name])) or (not np.isfinite(value.numpy().all())) or (np.isnan(value.numpy().any())):
	        success = False
	print("model save and load success? {}".format(success))


        

## Model Evaluation

When we need to do the inference task by constructed model in the DyGraph mode, we can use `YourModel.eval()` interface. In the before "Handwriting Digit Recognition" model, we use `mnist.eval()` to activate inference mode(training mode in the contexts of `fluid.dygraph.guard()` by default), in which DyGraph will only execute the forward inference network rather than do the derivation automatically and execute the back network:

The following codes show how to train and save a model for executing "Handwriting Digit Recognition" and infer with saved models by DyGraph.

In the first `fluid.dygraph.guard()` context, we train and save the models. It's noticeable that if we need to infer during training process, we can switch to the inference mode by `YourModel.eval()` and switch back to the training mode to continue training afer inference by `YourModel.train()`.

In the second `fluid.dygraph.guard()` context we can use previously saved `checkpoint` to do the inference. Also, we need to switch to the inference mode by `YourModel.eval()`.
			
	with fluid.dygraph.guard():
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
	
        mnist = MNIST("mnist")
        adam = AdamOptimizer(learning_rate=0.001)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
	
                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True
	
                cost, acc = mnist(img, label)
	
                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
            mnist.eval()
            test_cost, test_acc = self._test_train(test_reader, mnist, BATCH_SIZE)
            mnist.train()
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))
	
        fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
        print("checkpoint saved")

    with fluid.dygraph.guard():
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist_infer = MNIST("mnist")
        # load checkpoint
        mnist_infer.load_dict(
            fluid.dygraph.load_persistables(mnist.state_dict(), "save_dir"))
        print("checkpoint loaded")

        # start evaluate mode
        mnist_infer.eval()
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/infer_3.png')

        results = mnist_infer(to_variable(tensor_img))
        lab = np.argsort(results.numpy())
        print("Inference result of image/infer_3.png is: %d" % lab[0][0])

	
	
	
	Loss at epoch 3 , Test avg_loss is: 0.0721620170576, acc is: 0.97796474359
	Loss at epoch 4 step 0: [0.01078923]
	Loss at epoch 4 step 100: [0.10447877]
	Loss at epoch 4 step 200: [0.05149534]
	Loss at epoch 4 step 300: [0.0122997]
	Loss at epoch 4 step 400: [0.0281883]
	Loss at epoch 4 step 500: [0.10709661]
	Loss at epoch 4 step 600: [0.1306036]
	Loss at epoch 4 step 700: [0.01628026]
	Loss at epoch 4 step 800: [0.07947419]
	Loss at epoch 4 step 900: [0.02067161]
	Loss at epoch 4 , Test avg_loss is: 0.0802323290939, acc is: 0.976963141026
	checkpoint saved
	checkpoint loaded
	
	
	Ran 1 test in 208.017s
	
	Inference result of image/infer_3.png is: 3


## Build Compatible Model

Take the "Handwriting Digit Recognition" in the last step for example, the same modlel codes can execute in `Executor` of PaddlePaddle:

	exe = fluid.Executor(fluid.CPUPlace(
        ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

        mnist = MNIST("mnist")
        sgd = SGDOptimizer(learning_rate=1e-3)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size= BATCH_SIZE, drop_last=True)

        img = fluid.layers.data(
            name='pixel', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        cost = mnist(img)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        sgd.minimize(avg_loss)

        # initialize params and fetch them
        static_param_init_value = {}
        static_param_name_list = []
        for param in mnist.parameters():
            static_param_name_list.append(param.name)

        out = exe.run(fluid.default_startup_program(),
                      fetch_list=static_param_name_list)

        for i in range(len(static_param_name_list)):
            static_param_init_value[static_param_name_list[i]] = out[i]

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                static_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape([BATCH_SIZE, 1])

                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)
                out = exe.run(
                    fluid.default_main_program(),
                    feed={"pixel": static_x_data,
                          "label": y_data},
                    fetch_list=fetch_list)

                static_param_value = {}
                static_out = out[0]
                for i in range(1, len(out)):
                    static_param_value[static_param_name_list[i - 1]] = out[
                        i]

			
			
