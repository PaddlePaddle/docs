```python
import paddle
from paddle.vision.models import vgg11
import paddle.nn.functional as F
import numpy as np

print(paddle.__version__)
```

    2.2.0



```python
model = vgg11()

x = paddle.rand([1,3,224,224])
label = paddle.randint(0,1000)
```

    W1111 00:45:40.487871   104 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1111 00:45:40.493650   104 device_context.cc:465] device: 0, cuDNN Version: 7.6.



```python
predicts = model(x)
```


```python
loss = F.cross_entropy(predicts, label)
```


```python
loss.backward()
```


```python
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
```


```python
optim.step()
```


```python
import paddle

a = paddle.to_tensor([1.0, 2.0, 3.0])
b = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False) # 将b设置为需要计算梯度的属性
print(a.stop_gradient)
print(b.stop_gradient)
```

    True
    False



```python
a.stop_gradient = False
print(a.stop_gradient)
```

    False



```python
import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
z = x ** 2 + 4 * y
```


```python
z.backward()
print("Tensor x's grad is: {}".format(x.grad))
print("Tensor y's grad is: {}".format(y.grad))
```

    Tensor x's grad is: Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [2., 4., 6.])
    Tensor y's grad is: Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [4., 4., 4.])



```python
import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
y = x + 3
y.backward(retain_graph=True) # 设置retain_graph为True，保留反向计算图
print("Tensor x's grad is: {}".format(x.grad))
```

    Tensor x's grad is: Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [1., 1., 1.])



```python
import paddle
import numpy as np

x = np.ones([2, 2], np.float32)
inputs2 = []

for _ in range(10):
    tmp = paddle.to_tensor(x)
    tmp.stop_gradient = False
    inputs2.append(tmp)

ret2 = paddle.add_n(inputs2)
loss2 = paddle.sum(ret2)

loss2.backward()
print("Before clear {}".format(loss2.gradient()))

loss2.clear_grad()
print("After clear {}".format(loss2.gradient()))
```

    Before clear [1.]
    After clear [0.]



```python
import paddle

a = paddle.to_tensor(2.0, stop_gradient=False)
b = paddle.to_tensor(5.0, stop_gradient=True)
c = a * b
c.backward()
print("Tensor a's grad is: {}".format(a.grad))
print("Tensor b's grad is: {}".format(b.grad))
```

    Tensor a's grad is: Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [5.])
    Tensor b's grad is: None



```python

import paddle

a = paddle.to_tensor(2.0, stop_gradient=False)
b = paddle.to_tensor(5.0, stop_gradient=False)
c = a * b
d = paddle.to_tensor(4.0, stop_gradient=False)
e = c * d
e.backward()
print("Tensor a's grad is: {}".format(a.grad))
print("Tensor b's grad is: {}".format(b.grad))
print("Tensor c's grad is: {}".format(c.grad))
print("Tensor d's grad is: {}".format(d.grad))
```

    Tensor a's grad is: Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [20.])
    Tensor b's grad is: Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [8.])
    Tensor c's grad is: Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [4.])
    Tensor d's grad is: Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [10.])



```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```


```python
model = Model()
print(model.sublayers())

print("----------------------")

for item in model.named_sublayers():
    print(item)
```

    [Flatten()]
    ----------------------
    ('flatten', Flatten())



```python
fc = paddle.nn.Linear(10, 3)
model.add_sublayer("fc", fc)
print(model.sublayers())
```

    [Flatten(), Linear(in_features=10, out_features=3, dtype=float32)]



```python
def function(layer):
    print(layer)

model.apply(function)
```

    Flatten()
    Linear(in_features=10, out_features=3, dtype=float32)
    Model(
      (flatten): Flatten()
      (fc): Linear(in_features=10, out_features=3, dtype=float32)
    )





    Model(
      (flatten): Flatten()
      (fc): Linear(in_features=10, out_features=3, dtype=float32)
    )




```python
sublayer_iter = model.children()
for sublayer in sublayer_iter:
    print(sublayer)
```

    Flatten()
    Linear(in_features=10, out_features=3, dtype=float32)



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


```python
model = Model()
model.parameters()
print("----------------------------------------------------------------------------------")
for item in model.named_parameters():
    print(item)
```

    ----------------------------------------------------------------------------------
    ('img', Parameter containing:
    Tensor(shape=[1, 3, 256, 256], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [[[[ 0.00330893,  0.00146855, -0.00315564, ..., -0.00037254,
               -0.00398024,  0.00175103],
              [-0.00269739,  0.00015307,  0.00215079, ...,  0.00083044,
                0.00433949,  0.00183416],
              [ 0.00124980,  0.00066814, -0.00296695, ..., -0.00166787,
               -0.00208646,  0.00066172],
              ...,
              [ 0.00118238, -0.00020917, -0.00211811, ...,  0.00341913,
                0.00110805, -0.00007380],
              [-0.00283090,  0.00450932, -0.00027968, ...,  0.00141592,
                0.00147790, -0.00163899],
              [ 0.00473807,  0.00005514,  0.00163972, ..., -0.00105391,
                0.00130420, -0.00455226]],
    
             [[ 0.00370526, -0.00421996, -0.00161209, ...,  0.00098369,
               -0.00364983,  0.00031144],
              [ 0.00173886,  0.00339773,  0.00141036, ...,  0.00346697,
                0.00417612,  0.00012173],
              [ 0.00120599,  0.00061922, -0.00084213, ..., -0.00172405,
                0.00378877, -0.00097374],
              ...,
              [-0.00322239,  0.00413360,  0.00473170, ...,  0.00415691,
                0.00108459, -0.00351989],
              [-0.00416756,  0.00164984,  0.00244981, ...,  0.00053153,
               -0.00464938,  0.00450330],
              [-0.00406198, -0.00193215, -0.00431253, ..., -0.00257889,
               -0.00165101, -0.00138488]],
    
             [[ 0.00441089,  0.00360072,  0.00199083, ..., -0.00120336,
                0.00208172,  0.00016561],
              [ 0.00456772, -0.00385161,  0.00081078, ..., -0.00298249,
               -0.00269728, -0.00413104],
              [ 0.00370318,  0.00103516,  0.00258130, ..., -0.00003251,
               -0.00032389, -0.00006440],
              ...,
              [-0.00348314, -0.00025856, -0.00374935, ..., -0.00344840,
                0.00243370, -0.00292505],
              [ 0.00477740,  0.00388781, -0.00466578, ...,  0.00121291,
                0.00004315, -0.00295597],
              [ 0.00455716,  0.00302863,  0.00055869, ...,  0.00052850,
                0.00218663,  0.00267356]]]]))



```python
model = Model()
out = model()
out.backward()
model.clear_gradients()
```


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


```python
model = Model()
print(model.buffers())
for item in model.named_buffers():class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
    print(item)
```


      File "/tmp/ipykernel_104/851298997.py", line 3
        for item in model.named_buffers():class Model(paddle.nn.Layer):
                                              ^
    SyntaxError: invalid syntax




```python
class Model(paddle.nn.Layer):

    def __init__(self):
        super(Model, self).__init__()
        self.flatten = paddle.nn.Flatten()

    def forward(self, inputs):
        y = self.flatten(inputs)
        return y
```


```python
x = paddle.randn([10, 1], 'float32')
model = Model()
model.eval()  # set model to eval mode
out = model(x)
model.train()  # set model to train mode
out = model(x)
```


```python
model = Model()
x = paddle.randn([10, 1], 'float32')
out = model(x)
print(out)
```

    Tensor(shape=[10, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [[-0.12737122],
            [-0.57012707],
            [-0.70294005],
            [ 0.14529558],
            [ 0.50616348],
            [-0.96126020],
            [ 0.51200545],
            [ 2.64334464],
            [ 1.11839330],
            [ 0.61924362]])



```python
def forward_post_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_post_hook_handle = model.flatten.register_forward_post_hook(forward_post_hook)
out = model(x)
print(out)
```

    Tensor(shape=[10, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [[2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.]])



```python
def forward_pre_hook(layer, input, output):
    return 2*output

x = paddle.ones([10, 1], 'float32')
model = Model()
forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
out = model(x)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipykernel_104/1696797913.py in <module>
          5 model = Model()
          6 forward_pre_hook_handle = model.flatten.register_forward_pre_hook(forward_pre_hook)
    ----> 7 out = model(x)
    

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        912                 self._built = True
        913 
    --> 914             outputs = self.forward(*inputs, **kwargs)
        915 
        916             for forward_post_hook in self._forward_post_hooks.values():


    /tmp/ipykernel_104/2161125479.py in forward(self, inputs)
          6 
          7     def forward(self, inputs):
    ----> 8         y = self.flatten(inputs)
          9         return y


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        892         with param_guard(self._parameters), param_guard(self._buffers):
        893             for forward_pre_hook in self._forward_pre_hooks.values():
    --> 894                 hook_result = forward_pre_hook(self, inputs)
        895                 if hook_result is not None:
        896                     if not isinstance(hook_result, tuple):


    TypeError: forward_pre_hook() missing 1 required positional argument: 'output'



```python

```
