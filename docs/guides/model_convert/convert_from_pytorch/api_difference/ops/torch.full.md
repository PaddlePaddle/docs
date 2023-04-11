## [torch 参数更多 ]torch.full

### [torch.full](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=full#torch.full)

```python
torch.full(size,
           fill_value,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.full](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_cn.html#full)

```python
paddle.full(shape,
            fill_value,
            dtype=None,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> size </font>  | <font color='red'> shape </font>       | 表示输出形状大小，仅参数名不一致。 |
| fill_value  |  fill_value  |  表示填充值。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| dtype | dtype  | 表示数据类型。|
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.full([3, 5], 1., out=y)

# Paddle 写法
paddle.assign(paddle.full([3, 5], 1.), y)
```
#### device: Tensor 的设备
```python
# Pytorch 写法
torch.full([3, 5], 1., device=torch.device('cpu'))

# Paddle 写法
y = paddle.full([3, 5], 1.)
y.cpu()
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.full([3, 5], 1., requires_grad=True)

# Paddle 写法
x = paddle.full([3, 5], 1.)
x.stop_gradient = False
```
