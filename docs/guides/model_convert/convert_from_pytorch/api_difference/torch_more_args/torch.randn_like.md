## [ torch 参数更多 ] torch.randn_like

### [torch.randn_like](https://pytorch.org/docs/stable/generated/torch.randn_like.html#torch.randn_like)

```python
torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

### [paddle.randn_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/randn_like_cn.html#randn_like)

```python
paddle.randn_like(x, dtype=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor，仅参数名不一致。                                   |
| dtype         | dtype        | 表示数据类型。               |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，Paddle 无此参数，需要转写。 |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。               |

### 转写示例

#### requires_grad：表示是否不阻断梯度传导
```python
# PyTorch 写法
y = torch.randn_like(x，requires_grad=True)

# Paddle 写法
y = paddle.randn_like(x)
y.stop_gradient = False
```
