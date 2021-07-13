# Introduction to models and layers

Model is one of the key concepts in the field of deep learning. Roughly speaking, a model represents a specific deep learning algorithm, which maps a group of input variables to their corresponding outputs. In Paddle, a model holds the following two contents:

1. A series of layers to perform the variable mapping (forward pass).
2. A group of parameters which will update repeatedly along the training process.

In this guide, you will learn how to define and make use of models in Paddle. You will also learn the relationship between models and layers.

## Defining models and layers in Paddle

In Paddle, most models consist of a series of layers. Layer serves as the foundamental logical unit of a model, which holds two different contents: the variable that participates in the computation and the operator(s) that actually perform the execution.

Contructing a model from scratch could be a painful process with tons of nested codes to write and maintain. To make our life easier, Paddle provides a foundamental data structure ``paddle.nn.Layer`` to simplify the contruction of layer and model. Through inheriting from ``paddle.nn.Layer``, one may easily define a customized layer or model. Furthermore, since both model and layer are essentially inherited from ``paddle.nn.Layer``, model is nothing but a special layer in Paddle. 

Now let's construct a model making use of ``paddle.nn.Layer``:

```python
class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()
    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

Here we contructed a ``Model`` which inherites from ``paddle.nn.Layer``. This model only holds a single layer of ``paddle.nn.Flatten``, which flattens the input variables **inputs** upon execution.

## Sublayers

As discussed in the above section, a model is a special ``paddle.nn.Layer`` that holds a series of sublayers. Paddle provides a series of interfaces to visit and modify these sublayers in a model.

Taking the model we just constructed as an example, let's say we would like to print all its sublayers:

```python
model = Model()
print(model.sublayers())
for item in model.named_sublayers():
    print(item) 
```

```text
[Flatten()]
('flatten', Flatten())
```

As we can see, ``model.sublayers()`` allow us to access all the sublayers of a model (Remember there's only a single ``paddle.nn.Flatten`` layer in the model).

We can also iterate the sublayers through calling ``model.named_sublayers()``, which returns a tuple of (sublayer_name('flatten'), sublayer_object(paddle.nn.Flatten))

Now if we would like to further add a sublayer:

```python
fc = paddle.nn.Linear(10, 3)
model.add_sublayer("fc", fc)
print(model.sublayers()) 
```

```text
[Flatten(), Linear(in_features=10, out_features=3, dtype=float32)]
```

As we can see, ``model.add_sublayer()`` appends another ``paddle.nn.Linear`` sublayer to the model, resulting in two sublayers in the model: ``paddle.nn.Flatten`` and ``paddle.nn.Linear``

In the scenario where thousands of sublayers were added to a single model, one may request an interface to efficiently batch-modify all the sublayers. One may define a custom function and use ``apply()`` interface to apply that function to each of the sublayers of the model.

```python
def function(layer):
    print(layer)

model.apply(function)
```

```text
Flatten()
Linear(in_features=10, out_features=3, dtype=float32)
Model(
  (flatten): Flatten()
  (fc): Linear(in_features=10, out_features=3, dtype=float32)
)
```

Here we defined a function taking **layer** as argument, which further prints out the layer information. By calling ``model.apply()`` we applied the function to all the sublayers of the model and had each layer printed.

Another interface for batch-processing is ``children()`` or ``named_children()``. These two interfaces allow us to iterate through each sublayer:

```python
sublayer_iter = model.children()
for sublayer in sublayer_iter:
    print(sublayer)
```

```text
Flatten()
Linear(in_features=10, out_features=3, dtype=float32)
```

By calling ``model.children()``, we are able to get ``paddle.nn.Layer`` object through iterating the model.

## Variable of layer

### Add or modify a parameter to layer

In certain scenario, we may need to add a parameter to a model. One example could be image style transfer task, where we use a parameter as input and update that input parameter along with the training process to finally obtain the style-transferred image.

We can use the combination of ``create_parameter()`` and ``add_parameter()`` to create and add parameters to the model:

```python
class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        img = self.create_parameter([1,3,256,256])
        self.add_parameter("img", img)
        self.flatten = paddle.nn.Flatten()
    def forward(self):
        y = self.flatten(self.img)
        return y
```

Here we created a parameter named "img" and added it to the model, which can be accessed through **model.img**.

We can use  ``parameters()`` or ``named_parameters()`` to visit all the parameters in a model.

```python
model = Model()
model.parameters() 
for item in model.named_parameters():
    print(item)
```

```text
[Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
('img', Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
```

As we can see, ``model.parameters()`` returns all the parameters in a list.

In a real-time training process, after calling the backward method, Paddle will compute the gradient value for each parameters in a model and further saves them in the corresponding parameter object. If the parameter already got updated or for some reason we would like to discard the gradient, then we can call ``clear_gradients()`` to clear all the gradients in a parameter.

```python
model = Model()
out = model()
out.backward()
model.clear_gradients()
```

### Add or modify non-parameter variable to layer

Parameters actively participate in the training process and gets updated regularly. However, many times we only need a temporary variable or even a constant variable. For instance, if we want to temporarily save an intermediate variable during execution process for debug purpose, then we can make use of ``create_tensor()``:

```python
class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.saved_tensor = self.create_tensor(name="saved_tensor0")
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)
    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y
```

Making use of ``create_tensor()``, we created a temporary variable and assigned it to ``self.saved_tensor`` in the model. During execution, we called ``paddle.assign`` to snapshot the value of variable **y**.

### Add or modify **Buffer** variables to layer

The concept of **Buffer** only influence the transformation from dynamic graph to static graph. The key difference between **Buffer** variable and a temporary variable introduced in the previous section, lies in the fact that the temporary variable will not show up in the transformed static graph. Therefore, to create a variable that could be captured by the static graph, we need to use ``register_buffers()`` to create a **Buffer** variable instead.

```python
class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        saved_tensor = self.create_tensor(name="saved_tensor0")
        self.register_buffer("saved_tensor", saved_tensor, persistable=True)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(10, 100)
    def forward(self, input):
        y = self.flatten(input)
        # Save intermediate tensor
        paddle.assign(y, self.saved_tensor)
        y = self.fc(y)
        return y
```

Now **saved_tensor** will show up in the static graph.

To visit or modify the buffers in a model, one may call ``buffers()`` or ``named_buffers()``

```python
model = Model()
print(model.buffers())
for item in model.named_buffers():
    print(item)
```

```text
[Tensor(Not initialized)]
('saved_tensor', Tensor(Not initialized))
```

As we can see, ``model.buffers()`` returns all the buffers registered in a list.

## Execute a layer

After all the configurations, let's say we finally settled down a model as follow:

```python
class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()
    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

Before going through the execution process, we have to configure the execution mode first.

### Configure execution mode

There are two different execution modes in total, ``train()`` for training and ``eval()`` for inference.

```python
x = paddle.randn([10, 1], 'float32')
model = Model()
model.eval()  # set model to eval mode
out = model(x)
model.train()  # set model to train mode
out = model(x)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       ...
```

Here we first set the execution mode to **eval**, and soon after to **train**. The two execution modes are exlusive so the latter mode will override the former.

### Perform an execution

After setting the execution model, the model is ready to go. One may either call ``forward()`` or use ``__call__()`` to perform the forward logic defined in the model.

```python
model = Model()
x = paddle.randn([10, 1], 'float32')
out = model(x)
print(out)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       ...
```

Here we performed the forward execution through calling ``__call__()`` method.

### Add extra execution logic

There are times we would like to preprocess the inputs or postprocess the outputs before and after the layer execution, which can be realized making use of **hooks**. **hook** is a user-defined function taking one or more variables as argument, and get called during execution time. One may register **pre_hook** or **post_hook** or both to a layer. **pre_hook** preprocess the inputs of a layer, returning a set of new inputs to participate in the layer execution. **post_hook**, on the other hand, postprocess the outputs of a layer and returns the postprocessed outputs.

Let's register a **post_hook** through ``forward_post_hook()``

```python
def forward_post_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_post_hook_handle = model.flatten.register_forward_post_hook(forward_post_hook)
out = model(x)
print(out)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[2.],
        [2.],
        ...
```

Similarly, let's register a **pre_hook** through ``forward_pre_hook()``

```python
def forward_pre_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
out = model(x)
```

```text
Tensor(shape=[10, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[2.],
        [2.],
        ...
```

## Save a model's data

``state_dict()`` allows us to only save the data of the model without having to save the entire model itself. ``state_dict()`` fetches the parameters and persistent variables of a model and saves them into a **Python** dictionary. We can then save this dictionary to file.

```python
model = Model()
state_dict = model.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")
```

And recover the model anytime:

```text
model = Model()
state_dict = paddle.load("paddle_dy.pdparams")
model.set_state_dict(state_dict)
```

To entirely save a model, please refer to [paddle.jit.save()](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/jit/save_en.html)
