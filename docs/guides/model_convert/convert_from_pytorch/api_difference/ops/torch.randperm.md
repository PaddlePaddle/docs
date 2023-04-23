## [torch 参数更多 ]torch.randperm
### [torch.randperm](https://pytorch.org/docs/stable/generated/torch.randperm.html?highlight=randperm#torch.randperm)
```python
torch.randperm(n,
               *,
               generator=None,
               out=None,
               dtype=torch.int64,
               layout=torch.strided,
               device=None,
               requires_grad=False,
               pin_memory=False)
```
### [paddle.randperm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randperm_cn.html#randperm)
```python
paddle.randperm(n,
                dtype='int64',
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| n  |  n |  表示随机序列的上限  |
| <font color='red'>generator</font>  | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| dtype           | dtype            | 表示输出 Tensor 的数据类型。               |
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |
| <font color='red'> pin_memeory </font>   | - | 表示是否使用锁页内存， Paddle 无此参数，需要进行转写。   |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.randperm(10, (2, 2), out=y)

# Paddle 写法
paddle.assign(paddle.randperm(10, [2, 2], 1.), y)
```


#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.randperm(10, (2, 2), requires_grad=True)

# Paddle  写法
x = paddle.randperm(10, [2, 2])
x.stop_gradient = False
```

#### pin_memory：是否分配到固定内存上
```python
# Pytorch 写法
x = torch.randperm(10, (2, 2), pin_memory=True)

# Paddle 写法
x = paddle.randperm(10, [2, 2]).pin_memory()
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.randperm(10, (2, 2), device=torch.device('cpu'))

# Paddle  写法
y = paddle.randperm(10, [2, 2])
y.cpu()
```
