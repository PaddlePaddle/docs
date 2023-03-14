## [ 参数用法不一致 ]torch.rand

### [torch.rand](https://pytorch.org/docs/stable/generated/torch.rand.html?highlight=rand#torch.rand)

```python
torch.rand(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.rand](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rand_cn.html#rand)

```python
paddle.rand(shape,
            dtype=None,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小，Pytorch 以可变参数方式传入，Paddle 以 list 或 tuple 的方式传入。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |
| dtype           | dtype            | 表示数据类型。               |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，需要进行转写。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数，需要进行转写。 |


### 转写示例
#### *size：输出形状大小
```python
# Pytorch 写法
torch.rand(3, 5)

# Paddle 写法
paddle.rand([3, 5])
```

#### out：指定输出
```python
# Pytorch 写法
torch.rand([3, 5], out=y)

# Paddle 写法
y = paddle.rand([3, 5])
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.rand(3, 5, device=torch.device('cpu'))

# Paddle 写法
y = paddle.rand([3, 5])
y.cpu()
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.rand([3, 5], requires_grad=True)

# Paddle 写法
x = paddle.rand([3, 5])
x.stop_gradient = False
```
