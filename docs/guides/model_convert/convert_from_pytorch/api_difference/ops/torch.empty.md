## [torch 参数更多 ]torch.empty
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
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> *size </font>         | <font color='red'> shape  </font>       | 表示输出形状大小， PyTorch 是多个元素， Paddle 是列表或元组，需要进行转写。 |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| dtype | dtype  | 表示数据类型。|
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |
| <font color='red'> pin_memeory </font>   | - | 表示是否使用锁页内存， Paddle 无此参数，需要进行转写。   |

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
paddle.assign(paddle.empty([2, 3]), y)
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.zeros_like(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.zeros_like(x)
y.cpu()
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
