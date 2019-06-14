# 动态图机制-DyGraph



PaddlePaddle的DyGraph模式是一种动态的图执行机制，可以立即执行结果，无需构建整个图。同时，和以往静态的执行计算图不同，DyGraph模式下您的所有操作可以立即获得执行结果，而不必等待所构建的计算图全部执行完成，这样可以让您更加直观地构建PaddlePaddle下的深度学习任务，以及进行模型的调试，同时还减少了大量用于构建静态计算图的代码，使得您编写、调试网络的过程变得更加便捷。



PaddlePaddle DyGraph是一个更加灵活易用的模式，可提供：      

*	更加灵活便捷的代码组织结构： 使用python的执行控制流程和面向对象的模型设计


* 	更加便捷的调试功能： 直接使用python的打印方法即时打印所需要的结果，从而检查正在运行的模型结果便于测试更改


*  和静态执行图通用的模型代码：同样的模型代码可以使用更加便捷的DyGraph调试，执行，同时也支持使用原有的静态图模式执行


## 设置和基本用法

1. 升级到最新的PaddlePaddle 1.5:		
		
		pip install -q --upgrade paddlepaddle==1.5

2. 使用`fluid.dygraph.guard(place=None)` 上下文：
		
		import paddle.fluid as fluid
		with fluid.dygraph.guard():
			# write your executable dygraph code here             
			 
	现在您就可以在`fluid.dygraph.guard()`上下文环境中使用DyGraph的模式运行网络了，DyGraph将改变以往PaddlePaddle的执行方式： 现在他们将会立即执行，并且将计算结果返回给Python。


	Dygraph将非常适合和Numpy一起使用，使用`fluid.dygraph.to_variable(x)`将会将ndarray转换为`fluid.Variable`，而使用`fluid.Variable.numpy()`将可以把任意时刻获取到的计算结果转换为Numpy`ndarray`：         
	
			x = np.ones([2, 2], np.float32)
			with fluid.dygraph.guard():
		        inputs = []
		        for _ in range(10):
		            inputs.append(fluid.dygraph.to_variable(x))
		        ret = fluid.layers.sums(inputs)
		        print(ret.numpy())
					
			                        
			[[10. 10.]
			[10. 10.]]

			Process finished with exit code 0


	>	这里创建了一系列`ndarray`的输入，执行了一个`sum`操作之后，我们可以直接将运行的结果打印出来
	
	然后通过调用`reduce_sum`后使用`Variable.backward()`方法执行反向，使用`Variable.gradient()`方法即可获得反向网络执行完成后的梯度值的`ndarray`形式：

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
	            var_inp = fluid.dygraph.to_variable(np_inp)
	            outs = my_py_layer(var_inp)
	            dy_out = np.sum(outs[0].numpy())
	            outs[0].backward()
	            dy_grad = var_inp.gradient()

	>  请注意，继承自`fluid.PyLayer`的网络和继承自`fluid.Layer`的网络暂时不可混用	
-->
## 基于DyGraph构建网络
		
1. 编写一段用于DyGraph执行的Object-Oriented-Designed, PaddlePaddle模型代码主要由以下**三个部分**组成： **请注意，如果您设计的这一层结构是包含参数的，则必须要使用继承自`fluid.dygraph.Layer`的Object-Oriented-Designed的类来描述该层的行为。**

	
	1. 建立一个可以在DyGraph模式中执行的，Object-Oriented的网络，需要继承自`fluid.dygraph.Layer`，其中需要调用基类的`__init__`方法，并且实现带有参数`name_scope`（用来标识本层的名字）的`__init__`构造函数，在构造函数中，我们通常会执行一些例如参数初始化，子网络初始化的操作，执行这些操作时不依赖于输入的动态信息:
		
			class MyLayer(fluid.Layer):
			    def __init__(self, name_scope):
			        super(MyLayer, self).__init__(name_scope)    
			        
	2. 实现一个`forward(self, *inputs)`的执行函数，该函数将负责执行实际运行时网络的执行逻辑， 该函数将会在每一轮训练/预测中被调用，这里我们将执行一个简单的`relu` -> `elementwise add` -> `reduce sum`：
	
			def forward(self, inputs):
		        x = fluid.layers.relu(inputs)
		        self._x_for_debug = x
		        x = fluid.layers.elementwise_mul(x, x)
		        x = fluid.layers.reduce_sum(x)
		        return [x]       
		        

2. 在`fluid.dygraph.guard()`中执行：

	1. 使用Numpy构建输入：
		
			np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)

	2. 转换输入的`ndarray`为`Variable`, 并执行前向网络获取返回值： 使用`fluid.dygraph.to_variable(np_inp)`转换Numpy输入为DyGraph接收的输入，然后使用`my_layer(var_inp)[0]`调用callable object并且获取了`x`作为返回值，利用`x.numpy()`方法直接获取了执行得到的`x`的`ndarray`返回值。

			with fluid.dygraph.guard():
			    var_inp = fluid.dygraph.to_variable(np_inp)
			    my_layer = MyLayer("my_layer")
			    x = my_layer(var_inp)[0]
			    dy_out = x.numpy()

	3. 计算梯度：自动微分对于实现机器学习算法（例如用于训练神经网络的反向传播）来说很有用， 使用`x.backward()`方法可以从某个`fluid.Varaible`开始执行反向网络，同时利用`my_layer._x_for_debug.gradient()`获取了网络中`x`梯度的`ndarray` 返回值：
		
			    x.backward()
			    dy_grad = my_layer._x_for_debug.gradient()


完整代码如下：

			
	import paddle.fluid as fluid
	import numpy as np
	
	
	class MyLayer(fluid.dygraph.Layer):
	    def __init__(self, name_scope):
	        super(MyLayer, self).__init__(name_scope)
	
	    def forward(self, inputs):
	        x = fluid.layers.relu(inputs)
	        self._x_for_debug = x
	        x = fluid.layers.elementwise_mul(x, x)
	        x = fluid.layers.reduce_sum(x)
	        return [x]
	
	
	if __name__ == '__main__':
	    np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
	    with fluid.dygraph.guard():
	        var_inp = fluid.dygraph.to_variable(np_inp)
	        my_layer = MyLayer("my_layer")
	        x = my_layer(var_inp)[0]
	        dy_out = x.numpy()
	        x.backward()
	        dy_grad = my_layer._x_for_debug.gradient()
	        my_layer.clear_gradients() # 将参数梯度清零以保证下一轮训练的正确性

## 使用DyGraph训练模型

接下来我们将以“手写数字识别”这个最基础的模型为例，展示如何利用DyGraph模式搭建并训练一个模型：

有关手写数字识别的相关理论知识请参考[PaddleBook](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits)中的内容，我们在这里默认您已经了解了该模型所需的深度学习理论知识。


1.	准备数据，我们使用`paddle.dataset.mnist`作为训练所需要的数据集：
		
		train_reader = paddle.batch(
		paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

2. 构建网络，虽然您可以根据之前的介绍自己定义所有的网络结构，但是您也可以直接使用`fluid.dygraph.Layer`当中我们为您定制好的一些基础网络结构，这里我们利用`fluid.dygraph.Conv2D`以及`fluid.dygraph.Pool2d`构建了基础的`SimpleImgConvPool`：

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
		
		        self._conv2d = fluid.dygraph.Conv2D(
		            self.full_name(),
		            num_filters=num_filters,
		            filter_size=filter_size,
		            stride=conv_stride,
		            padding=conv_padding,
		            dilation=conv_dilation,
		            groups=conv_groups,
		            param_attr=None,
		            bias_attr=None,
		            act=act,
		            use_cudnn=use_cudnn)
		
		        self._pool2d = fluid.dygraph.Pool2D(
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



	> 注意: 构建网络时子网络的定义和使用请在`__init__`中进行， 而子网络的执行部分则在`forward`函数中进行


		       

3. 利用已经构建好的`SimpleImgConvPool`组成最终的`MNIST`网络：

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
		        self._fc = fluid.dygraph.FC(self.full_name(),
		                      10,
		                      param_attr=fluid.param_attr.ParamAttr(
		                          initializer=fluid.initializer.NormalInitializer(
		                              loc=0.0, scale=scale)),
		                      act="softmax")
		
		    def forward(self, inputs, label=None):
		        x = self._simple_img_conv_pool_1(inputs)
		        x = self._simple_img_conv_pool_2(x)
		        x = self._fc(x)
		        if label is not None:
		            acc = fluid.layers.accuracy(input=x, label=label)
		            return x, acc
		        else:
		            return x
				  


			
4. 在`fluid.dygraph.guard()`中定义配置好的`MNIST`网络结构，此时即使没有训练也可以在`fluid.dygraph.guard()`中调用模型并且检查输出：
	
			with fluid.dygraph.guard():
				mnist = MNIST("mnist")
				id, data = list(enumerate(train_reader()))[0]
				dy_x_data = np.array(
				    [x[0].reshape(1, 28, 28)
				     for x in data]).astype('float32')
				img = fluid.dygraph.to_variable(dy_x_data)
				print("cost is: {}".format(mnist(img).numpy()))
				
				
				
				cost is: [[0.10135901 0.1051138  0.1027941  ... 0.0972859  0.10221873 0.10165327]
				[0.09735426 0.09970362 0.10198303 ... 0.10134517 0.10179105 0.10025002]
				[0.09539858 0.10213123 0.09543551 ... 0.10613529 0.10535969 0.097991  ]
				...
				[0.10120598 0.0996111  0.10512722 ... 0.10067689 0.10088114 0.10071224]
				[0.09889644 0.10033772 0.10151272 ... 0.10245881 0.09878646 0.101483  ]
				[0.09097178 0.10078511 0.10198414 ... 0.10317434 0.10087223 0.09816764]]
					
				Process finished with exit code 0

5. 构建训练循环，在每一轮参数更新完成后我们调用`mnist.clear_gradients()`来重置梯度：

		with fluid.dygraph.guard():
		    epoch_num = 5		
		    BATCH_SIZE = 64
		    train_reader = paddle.batch(
		        paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
		    mnist = MNIST("mnist")
		    id, data = list(enumerate(train_reader()))[0]
		    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
		    for epoch in range(epoch_num):
		        for batch_id, data in enumerate(train_reader()):
		            dy_x_data = np.array([x[0].reshape(1, 28, 28)
		                                  for x in data]).astype('float32')
		            y_data = np.array(
		                [x[1] for x in data]).astype('int64').reshape(-1, 1)
		
		            img = fluid.dygraph.to_variable(dy_x_data)
		            label = fluid.dygraph.to_variable(y_data)
		
		            cost = mnist(img)
		
		            loss = fluid.layers.cross_entropy(cost, label)
		            avg_loss = fluid.layers.mean(loss)
		
		            if batch_id % 100 == 0 and batch_id is not 0:
		                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
		            avg_loss.backward()
		            adam.minimize(avg_loss)
		            mnist.clear_gradients()




6. 变量及优化器

	模型的参数或者任何您希望检测的值可以作为变量封装在类中，并且通过对象获取并使用`numpy()`方法获取其`ndarray`的输出， 在训练过程中您可以使用`mnist.parameters()`来获取到网络中所有的参数，也可以指定某一个`Layer`的某个参数或者`parameters()`来获取该层的所有参数，使用`numpy()`方法随时查看参数的值

	反向运行后调用之前定义的`Adam`优化器对象的`minimize`方法进行参数更新:
		
		with fluid.dygraph.guard():
		    epoch_num = 5
		    BATCH_SIZE = 64
		
		    mnist = MNIST("mnist")
		    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
		    train_reader = paddle.batch(
		        paddle.dataset.mnist.train(), batch_size= BATCH_SIZE, drop_last=True)
		
		    np.set_printoptions(precision=3, suppress=True)
		    for epoch in range(epoch_num):
		        for batch_id, data in enumerate(train_reader()):
		            dy_x_data = np.array(
		                [x[0].reshape(1, 28, 28)
		                 for x in data]).astype('float32')
		            y_data = np.array(
		                [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
		
		            img = fluid.dygraph.to_variable(dy_x_data)
		            label = fluid.dygraph.to_variable(y_data)
		            label.stop_gradient = True
		
		            cost = mnist(img)
		            loss = fluid.layers.cross_entropy(cost, label)
		            avg_loss = fluid.layers.mean(loss)
		
		            dy_out = avg_loss.numpy()
		
		            avg_loss.backward()
		            adam.minimize(avg_loss)
		            mnist.clear_gradients()
		
		            dy_param_value = {}
		            for param in mnist.parameters():
		                dy_param_value[param.name] = param.numpy()
		
		            if batch_id % 20 == 0:
		                print("Loss at step {}: {}".format(batch_id, avg_loss.numpy()))
		    print("Final loss: {}".format(avg_loss.numpy()))
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

7.	性能

	在使用`fluid.dygraph.guard()`可以通过传入`fluid.CUDAPlace(0)`或者`fluid.CPUPlace()`来选择执行DyGraph的设备，通常如果不做任何处理将会自动适配您的设备。

## 模型参数的保存

 在模型训练中可以使用`                    fluid.dygraph.save_persistables(your_model_object.state_dict(), "save_dir", optimizers=None)`来保存`your_model_object`中所有的模型参数, 以及使用`learning rate decay`的优化器。也可以自定义需要保存的“参数名” - “参数对象”的Python Dictionary传入。

同样可以使用`models，optimizers =     fluid.dygraph.load_persistables("save_dir")`获取保存的模型参数和优化器。


再使用`your_modle_object.load_dict(models)`接口来恢复保存的模型参数从而达到继续训练的目的。

以及使用`your_optimizer_object.load(optimizers)`接口来恢复保存的优化器中的`learning rate decay`值




下面的代码展示了如何在“手写数字识别”任务中保存参数并且读取已经保存的参数来继续训练。


	with fluid.dygraph.guard():
	    epoch_num = 5
	    BATCH_SIZE = 64
	
	    mnist = MNIST("mnist")
	    adam = fluid.optimizer.Adam(learning_rate=0.001)
	    train_reader = paddle.batch(
	        paddle.dataset.mnist.train(), batch_size= BATCH_SIZE, drop_last=True)
	
	    np.set_printoptions(precision=3, suppress=True)
	    dy_param_init_value={}
	    for epoch in range(epoch_num):
	        for batch_id, data in enumerate(train_reader()):
	            dy_x_data = np.array(
	                [x[0].reshape(1, 28, 28)
	                 for x in data]).astype('float32')
	            y_data = np.array(
	                [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
	
	            img = fluid.dygraph.to_variable(dy_x_data)
	            label = fluid.dygraph.to_variable(y_data)
	            label.stop_gradient = True
	
	            cost = mnist(img)
	            loss = fluid.layers.cross_entropy(cost, label)
	            avg_loss = fluid.layers.mean(loss)
	
	            dy_out = avg_loss.numpy()
	
	            avg_loss.backward()
	            adam.minimize(avg_loss)
	            if batch_id == 20:
	                fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir", adam)
	            mnist.clear_gradients()
	
	            if batch_id == 20:
	                for param in mnist.parameters():
	                    dy_param_init_value[param.name] = param.numpy()
	                model, _ = fluid.dygraph.load_persistables("save_dir")
	                mnist.load_dict(model)
	                break
	        if epoch == 0:
	            break
	    restore = mnist.parameters()
	    # check save and load
	
	    success = True
	    for value in restore:
	        if (not np.array_equal(value.numpy(), dy_param_init_value[value.name])) or (not np.isfinite(value.numpy().all())) or (np.isnan(value.numpy().any())):
	            success = False
	    print("model save and load success? {}".format(success))


        

## 模型评估

当我们需要在DyGraph模式下利用搭建的模型进行预测任务，可以使用`YourModel.eval()`接口，在之前的手写数字识别模型中我们使用`mnist.eval()`来启动预测模式（我们默认在`fluid.dygraph.guard()`上下文中是训练模式），在预测的模式下，DyGraph将只会执行前向的预测网络，而不会进行自动求导并执行反向网络：

下面的代码展示了如何使用DyGraph模式训练一个用于执行“手写数字识别”任务的模型并保存，并且利用已经保存好的模型进行预测。

我们在`fluid.dygraph.guard()`上下文中进行了模型的保存和训练，值得注意的是，当我们需要在训练的过程中进行预测时需要使用`YourModel.eval()`切换到预测模式，并且在预测完成后使用`YourModel.train()`切换回训练模式继续训练。

我们在`inference_mnist `中启用另一个`fluid.dygraph.guard()`，并在其上下文中`load`之前保存的`checkpoint`进行预测，同样的在执行预测前需要使用`YourModel.eval()`来切换的预测模式。
			
	def test_mnist(reader, model, batch_size):
	    acc_set = []
	    avg_loss_set = []
	    for batch_id, data in enumerate(reader()):
	        dy_x_data = np.array([x[0].reshape(1, 28, 28)
	                              for x in data]).astype('float32')
	        y_data = np.array(
	            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
	
	        img = fluid.dygraph.to_variable(dy_x_data)
	        label = fluid.dygraph.to_variable(y_data)
	        label.stop_gradient = True
	        prediction, acc = model(img, label)
	        loss = fluid.layers.cross_entropy(input=prediction, label=label)
	        avg_loss = fluid.layers.mean(loss)
	        acc_set.append(float(acc.numpy()))
	        avg_loss_set.append(float(avg_loss.numpy()))
	
	        # get test acc and loss
	    acc_val_mean = np.array(acc_set).mean()
	    avg_loss_val_mean = np.array(avg_loss_set).mean()
	
	    return avg_loss_val_mean, acc_val_mean
	
	
	def inference_mnist():
	    with fluid.dygraph.guard():
	        mnist_infer = MNIST("mnist")
	        # load checkpoint
	        model_dict, _ = fluid.dygraph.load_persistables("save_dir")
	        mnist_infer.load_dict(model_dict)
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
	        tensor_img = load_image(cur_dir + '/image/2.png')
	
	        results = mnist_infer(fluid.dygraph.to_variable(tensor_img))
	        lab = np.argsort(results.numpy())
	        print("Inference result of image/infer_3.png is: %d" % lab[0][-1])

	with fluid.dygraph.guard():
	    epoch_num = 1
	    BATCH_SIZE = 64
	    mnist = MNIST("mnist")
	    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
	    test_reader = paddle.batch(
	        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
	
	    train_reader = paddle.batch(
	        paddle.dataset.mnist.train(),
	        batch_size=BATCH_SIZE,
	        drop_last=True)
	
	    for epoch in range(epoch_num):
	        for batch_id, data in enumerate(train_reader()):
	            dy_x_data = np.array([x[0].reshape(1, 28, 28)
	                                  for x in data]).astype('float32')
	            y_data = np.array(
	                [x[1] for x in data]).astype('int64').reshape(-1, 1)
	
	            img = fluid.dygraph.to_variable(dy_x_data)
	            label = fluid.dygraph.to_variable(y_data)
	            label.stop_gradient = True
	
	            cost, acc = mnist(img, label)
	
	            loss = fluid.layers.cross_entropy(cost, label)
	            avg_loss = fluid.layers.mean(loss)
	
	
	            avg_loss.backward()
	
	            adam.minimize(avg_loss)
	            # save checkpoint
	            mnist.clear_gradients()
	            if batch_id % 100 == 0:
	                print("Loss at epoch {} step {}: {:}".format(
	                    epoch, batch_id, avg_loss.numpy()))
	
	        mnist.eval()
	        test_cost, test_acc = test_mnist(test_reader, mnist, BATCH_SIZE)
	        mnist.train()
	        print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
	            epoch, test_cost, test_acc))
	
	    fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
	    print("checkpoint saved")
	
	    inference_mnist()
	
	
	
	Loss at epoch 0 step 0: [2.2991252]
	Loss at epoch 0 step 100: [0.15491392]
	Loss at epoch 0 step 200: [0.13315125]
	Loss at epoch 0 step 300: [0.10253005]
	Loss at epoch 0 step 400: [0.04266362]
	Loss at epoch 0 step 500: [0.08894891]
	Loss at epoch 0 step 600: [0.08999012]
	Loss at epoch 0 step 700: [0.12975612]
	Loss at epoch 0 step 800: [0.15257305]
	Loss at epoch 0 step 900: [0.07429226]
	Loss at epoch 0 , Test avg_loss is: 0.05995981965082674, acc is: 0.9794671474358975
	checkpoint saved
	No optimizer loaded. If you didn't save optimizer, please ignore this. The program can still work with new optimizer. 
	checkpoint loaded
	Inference result of image/infer_3.png is: 3


## 编写兼容的模型

以上一步中手写数字识别的例子为例，动态图的模型代码可以直接用于静态图中作为模型代码，执行时，直接使用PaddlePaddle静态图执行方式即可，这里以静态图中的`executor`为例, 模型代码可以直接使用之前的模型代码，执行时使用`Executor`执行即可
	
	epoch_num = 1
	BATCH_SIZE = 64
	exe = fluid.Executor(fluid.CPUPlace())
	
	mnist = MNIST("mnist")
	sgd = fluid.optimizer.SGDOptimizer(learning_rate=1e-3)
	train_reader = paddle.batch(
	    paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
	img = fluid.layers.data(
	    name='pixel', shape=[1, 28, 28], dtype='float32')
	label = fluid.layers.data(name='label', shape=[1], dtype='int64')
	cost = mnist(img)
	loss = fluid.layers.cross_entropy(cost, label)
	avg_loss = fluid.layers.mean(loss)
	sgd.minimize(avg_loss)
	
	out = exe.run(fluid.default_startup_program())
	
	for epoch in range(epoch_num):
	    for batch_id, data in enumerate(train_reader()):
	        static_x_data = np.array(
	            [x[0].reshape(1, 28, 28)
	             for x in data]).astype('float32')
	        y_data = np.array(
	            [x[1] for x in data]).astype('int64').reshape([BATCH_SIZE, 1])
	
	        fetch_list = [avg_loss.name]
	        out = exe.run(
	            fluid.default_main_program(),
	            feed={"pixel": static_x_data,
	                  "label": y_data},
	            fetch_list=fetch_list)
	
	        static_out = out[0]
	
	        if batch_id % 100 == 0 and batch_id is not 0:
	            print("epoch: {}, batch_id: {}, loss: {}".format(epoch, batch_id, static_out))
    

			
			
