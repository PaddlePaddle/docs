## [ 参数不一致 ]torch.randn

### [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html?highlight=randn#torch.randn)

```python
torch.randn(*size,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```

### [paddle.randn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/randn_cn.html#randn)

```python
paddle.randn(shape,
             dtype=None,
             name=None)
```

其中 torch 的 `size` 和 paddle 的 `shape` 用法不一致，torch 还支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| *size         | shape        | 表示输出形状大小，Pytorch 以可变参数方式传入，Paddle 以 list 或 tuple 的方式传入。                                     |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |
| dtype           | dtype            | 表示输出 Tensor 的数据类型。               |
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要转写。 |



### 转写示例
#### *size：输出形状大小
```python
# Pytorch 写法
torch.randn(3, 5)

# Paddle 写法
paddle.randn([3, 5])
```

#### out：指定输出
```python
# Pytorch 写法
torch.randn([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.randn([3, 5]), y)
```

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.randn((2, 2), device=torch.device('cpu'))

# Paddle 写法
y = paddle.randn([2, 2])
y.cpu()
```

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.randn([3, 5], requires_grad=True)

# Paddle 写法
x = paddle.randn([3, 5])
x.stop_gradient = False
```
