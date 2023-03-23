## [ 一致的参数 ] torch.Tensor.mul_
### [torch.Tensor.mul_](https://pytorch.org/docs/1.13/generated/torch.Tensor.mul_.html?highlight=mul_)

```python
torch.Tensor.mul_(value)

#示例代码
import torch

x = torch.Tensor([[1, 2], [3, 4]])
res = x.mul_(2)
print(res) # [[2, 4], [6, 8]]
```

### [paddle.Tensor.scale_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#id16)

```python
paddle.Tensor.scale_(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

#示例代码
import paddle

x = paddle.to_tensor([[1, 2], [3, 4]])
res = x.scale_(2)
print(res) # [[2, 4], [6, 8]]
```
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| value          | scale         | 放缩的大小                                     |
| -          | bias         | 偏置                                     |

两者功能一致，输入一个常数value(scale)，将矩阵x放大value(scale)倍，需要将bias设置为0.0