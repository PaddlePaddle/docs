## [ 组合替代实现 ]torch.randn_like

### [torch.randn_like](https://pytorch.org/docs/stable/generated/torch.randn_like.html#torch.randn_like)
```python
torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。

### 参数介绍
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | -            | 表示输入的 Tensor                                   |
| dtype         | -            | 表示数据类型。               |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，Paddle 无此参数，需要转写。                   |
| requires_grad | stop_gradient            | 表示是否不阻断梯度传导，Paddle 无此参数，需要转写。 |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。               |

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 转写示例
#### input：表示输入的 Tensor
```python
# Pytorch 写法
torch.randn_like(x)

# Paddle 写法
paddle.randn(shape=x.shape, dtype=x.dtype)
```

#### dtype：表示数据类型
```python
# Pytorch 写法
torch.randn_like(x，dtype=torch.float64)

# Paddle 写法
paddle.randn(shape=x.shape, dtype=paddle.float64)
```

#### requires_grad：表示是否不阻断梯度传导
```python
# Pytorch 写法
y = torch.randn_like(x，requires_grad=True)

# Paddle 写法
y = paddle.randn(shape=x.shape, dtype=x.dtype)
y.stop_gradient = False
```
