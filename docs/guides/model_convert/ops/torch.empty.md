## torch.empty
### [torch.empty](https://pytorch.org/docs/stable/generated/torch.empty.html?highlight=empty#torch.empty)

```python
torch.empty(*size,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False,
            pin_memory=False)
```

### [paddle.empty](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/empty_cn.html#empty)

```python
paddle.empty(shape,
             dtype=None,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle 无此参数。           |


### 功能差异

#### 使用方式
***PyTorch***：生成 Tensor 的形状大小以可变参数的方式传入。
***PaddlePaddle***：生成 Tensor 的形状大小以 list 的方式传入。


### 转写示例
#### out：指定输出
```python
torch.full([3, 5], 1., out=y)
```

```python
y = paddle.full([3, 5], 1.)
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
x = torch.full([3, 5], 1., requires_grad=True)
```

```python
x = paddle.full([3, 5], 1.)
x.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上
```python
x = torch.eye(3, pin_memory=True)
```

```python
x = paddle.eye(3).pin_memory()
```


### 代码示例
``` python
# PyTorch 示例：
torch.empty(2, 3)
# 输出
# tensor([[9.1835e-41, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

``` python
# PaddlePaddle 示例：
paddle.empty([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0., 0., 0.],
#         [0., 0., 0.]])
```
