# Activation Function

Activation Function 是激活函数，它将非线性的特性引入到神经网络当中。

PaddlePaddle Fluid 对大部分的激活函数进行了支持，其中有:   
 `relu`, `tanh`, `sigmoid`, `elu`, `relu6`, `pow`, `stanh`, `hard_sigmoid`, `swish`, `prelu`, `brelu`, `leaky_relu`, `soft_relu`, `thresholded_relu`, `maxout`, `logsigmoid`, `hard_shrink`,	`softsign`, `softplus`, `tanh_shrink`, `softshrink`, `exp` 等。
 

## Fluid提供了两种激活函数的使用方式：

1. 如果一个层的接口提供了`act`变量（默认值为`None`），我们可以通过该变量指定该层的激活函数类型。该方式支持常见的激活函数，如`relu`, `tanh`, `sigmoid`。 

	```
	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
	```


2. Fluid为每个Activation提供了接口，我们可以显式的对它们进行调用。

	```
	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3)
	relu1 = fluid.layer.relu(conv2d)
	```
