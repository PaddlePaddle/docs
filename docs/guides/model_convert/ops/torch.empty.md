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

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle 无此参数。           |


### 转写示例
#### *size：输出形状大小
```python
# Pytorch 写法
torch.empty(3, 5)

# Paddle 写法
paddle.empty([3, 5])
```

#### out：指定输出
```python
# Pytorch 写法
torch.empty((2,3), out=y)

# Paddle 写法
y = paddle.empty([2, 3])
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.empty((2,3), requires_grad=True)

# Paddle 写法
x = paddle.empty([2, 3])
x.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.empty((2,3), pin_memory=True)

# Paddle 写法
x = paddle.empty([2, 3]).pin_memory()
```
