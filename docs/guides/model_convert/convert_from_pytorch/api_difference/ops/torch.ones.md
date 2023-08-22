## [ 参数不一致 ]torch.ones
### [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html?highlight=ones#torch.ones)

```python
torch.ones(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.ones](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/ones_cn.html#ones)

```python
paddle.ones(shape,
            dtype=None,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小，Pytorch 以可变参数方式传入，Paddle 以 list 或 tuple 的方式传入。                                     |
| out           | -            | 表示输出的 Tensor， Paddle 无此参数，需要转写。               |
| dtype         | dtype            | 表示数据类型。                                     |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。                  |
| requires_grad | -            | 表示是否不阻断梯度传导，Paddle 无此参数，需要转写。 |


### 转写示例
#### *size：输出形状大小
```python
# Pytorch 写法
torch.ones(3, 5)

# Paddle 写法
paddle.ones([3, 5])
```

#### out：指定输出
```python
# Pytorch 写法
torch.ones((3, 2), out=y)

# Paddle 写法
paddle.assign(paddle.ones([3, 2]), y)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.ones(3, 2, requires_grad=True)

# Paddle 写法
x = paddle.ones([3, 2])
x.stop_gradient = False
```


#### device: Tensor 的设备
```python
# Pytorch 写法
torch.ones(3, 2, device=torch.device('cpu'))

# Paddle 写法
y = paddle.ones([3, 2])
y.cpu()
```
