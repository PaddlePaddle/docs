## [torch 参数更多 ]torch.randint
### [torch.randint](https://pytorch.org/docs/stable/generated/torch.randint.html?highlight=randint#torch.randint)
```python
torch.randint(low=0,
              high,
              size,
              *,
              generator=None,
              out=None,
              dtype=None,
              layout=torch.strided,
              device=None,
              requires_grad=False)
```

### [paddle.randint](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randint_cn.html#randint)
```python
paddle.randint(low=0,
               high=None,
               shape=[1],
               dtype=None,
               name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| low          | low        | 表示生成的随机值范围的下限。                                     |
| high          | high        | 表示生成的随机值范围的上限。                                     |
| size          | shape        | 表示输出形状大小。                                     |
| <font color='red'>generator</font>  | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| dtype           | dtype            | 表示输出 Tensor 的数据类型。               |
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.randint(10, (2, 2), out=y)

# Paddle 写法
y = paddle.randint(10, [2, 2])
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.randint(10, (2, 2), requires_grad=True)

# Paddle 写法
x = paddle.randint(10, [2, 2])
x.stop_gradient = False
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.randint(10, (2, 2), device=torch.device('cpu'))

# Paddle 写法
y = paddle.randint(10, [2, 2])
y.cpu()
```
