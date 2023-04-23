# Introduction to models and layers

Model is one of the key concepts in the field of deep learning. Roughly speaking, a model represents a specific deep learning algorithm, which maps a group of input variables to another group of outputs. In Paddle, a model holds the following two contents:

1. A series of layers to perform the variable mapping (forward pass).
2. A group of parameters updateid repeatedly along with the training process.

In this guide, you will learn how to define and make use of models in Paddle, and further understand the relationship between models and layers.

## Defining models and layers in Paddle

In Paddle, most models consist of a series of layers. Layer serves as the foundamental logical unit of a model, composed of two parts: the variable that participates in the computation and the operator(s) that actually perform the execution.

Contructing a model from scratch could be painful, with tons of nested codes to write and maintain. To make life easier, Paddle provides foundamental data structure ``paddle.nn.Layer`` to simplify the contruction of layer or model. One may easily inherit from ``paddle.nn.Layer`` to define thier custom layers or models. In addition, since both model and layer are essentially inherited from ``paddle.nn.Layer``, model is nothing but a special layer in Paddle.

Now let us construct a model using ``paddle.nn.Layer``:

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

Here we contructed a ``Model`` which inherited from ``paddle.nn.Layer``. This model only holds a single layer of ``paddle.nn.Flatten``, which flattens the input variables **inputs** upon execution.

## Sublayers

As discussed in the above section, a model is a special ``paddle.nn.Layer`` that holds a series of sublayers. Paddle provides a series of interfaces to visit and modify these sublayers in a model.

Taking the model we just constructed as an example, let's say we would like to print all its sublayers:

```python
model = Model()
print(model.sublayers())

print("--------------------------")

for item in model.named_sublayers():
    print(item)
```

```text
[Flatten()]
--------------------------
('flatten', Flatten())
```

As we can see, ``model.sublayers()`` allows us to access all the sublayers of a model (Remember there's only a single ``paddle.nn.Flatten`` layer in the model).

We can also iterate through the sublayers via ``model.named_sublayers()``, which returns a tuple of (sublayer_name('flatten'), sublayer_object(paddle.nn.Flatten))

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

In certain scenario, we may need to add a parameter to a model. One example is image style transfer, where we use a parameter as input and update that input parameter along with the training process to finally obtain the style-transferred image.

We can use the combination of ``create_parameter()`` and ``add_parameter()`` to create and add parameters to the model:

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
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

print('-----------------------------------------------------------------------------------')

for item in model.named_parameters():
    print(item)
```

```text
[Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
-----------------------------------------------------------------------------------
('img', Parameter containing:
Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CPUPlace, stop_gradient=False,
       ...
```

As we can see, ``model.parameters()`` returns all the parameters in a list.

In a real-time training process, after calling the backward method, Paddle will compute the gradient value for each parameters in a model and further saves them in the corresponding parameter object. If the parameter already got updated, or for some reason we would like to discard the gradient, we can then seek ``clear_gradients()`` to clear all the gradients in a parameter.

```python
model = Model()
out = model()
out.backward()
model.clear_gradients()
```

### Add or modify non-parameter variable to layer

Parameters actively participate in the training process and gets updated regularly. However, many times we only need a temporary variable or even a constant variable. For instance, if we want to temporarily save an intermediate variable during execution simply for debugging purpose, we can then make use of ``create_tensor()``:

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
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

By calling ``create_tensor()``, we created a temporary variable and assigned it to ``self.saved_tensor`` in the model. During execution, ``paddle.assign`` was called to snapshot the value of variable **y**.

### Add or modify **Buffer** variables to layer

The concept of **Buffer** only influence the transformation from dynamic graph to static graph. The key difference between **Buffer** variable and temporary variable (introduced in the previous section), lies in the fact that temporary variables will not show up in the transformed static graph. To create a variable that captureable by the static graph, we have to use ``register_buffers()`` to create a **Buffer** variable instead.

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
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

print('-------------------------------------------')

for item in model.named_buffers():
    print(item)
```

```text
[Tensor(Not initialized)]
-------------------------------------------
('saved_tensor', Tensor(Not initialized))
```

As we can see, ``model.buffers()`` returns all the buffers registered in a list.

## Execute a layer

After all the configurations, let's say we finally settled down a model as follow:

```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```

Before going through the execution process, we have to first configure the execution mode first.

### Configure execution mode

In Paddle, there are two different execution modes for a model, ``train()`` for training and ``eval()`` for inference.

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

Here we first set the execution mode to **eval**, and soon after to **train**. The two execution modes are exlusive therefore the latter mode will override the former.

### Perform an execution

After setting the execution model, the model is ready to go. One may either call ``forward()`` or use ``__call__()`` to perform the forward function defined in the model.

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

There are times where we want to preprocess the inputs or postprocess the outputs before and after the layer execution, which can be realized using **hooks**. **hook** is a user-defined function taking one or more variables as argument, and get called during execution. One may register **pre_hook** or **post_hook** or both to a specific layer. **pre_hook** preprocess the inputs of a layer, returning a set of new inputs to participate in the layer execution. **post_hook**, on the other hand, postprocess the outputs of a layer.

Let's register a **post_hook** through ``register_forward_post_hook()``

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

Similarly, we can also register a **pre_hook** through ``register_forward_pre_hook()``

```python
def forward_pre_hook(layer, input):
    print(input)
    return input

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
out = model(x)
```

```text
(Tensor(shape=[10, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]]),)
```

## Save a model's data

``state_dict()`` allows us to only save the data in a model without having to save the model itself. ``state_dict()`` fetches the parameters and persistent variables of a model and put them into a **Python** dictionary. We can then save this dictionary to file.

```python
model = Model()
state_dict = model.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")
```

And recover the model anytime:

```python
model = Model()
state_dict = paddle.load("paddle_dy.pdparams")
model.set_state_dict(state_dict)
```

To save a model, please refer to [paddle.jit.save()](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/jit/save_en.html)
