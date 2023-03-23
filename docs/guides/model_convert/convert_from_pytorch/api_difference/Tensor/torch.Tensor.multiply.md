## [ 一致的参数 ] torch.Tensor.multiply

### [torch.Tensor.multiply](https://pytorch.org/docs/1.13/generated/torch.Tensor.multiply.html)

```python
torch.Tensor.multiply(value) 

#示例代码
import torch

x = torch.Tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
res = x.multiply(y)
print(res) # [[5, 12], [21, 32]]
```

### [paddle.Tensor.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#multiply-y-axis-1-name-none)

```python
paddle.Tensor.multiply(y, axis=-1, name=None)

#示例代码
import paddle

x = paddle.to_tensor([[1, 2], [3, 4]])
y = paddle.to_tensor([[5, 6], [7, 8]])
res = x.multiply(y)
print(res) # [[5, 12], [21, 32]]
```

两者功能一致，输入两个矩阵x,y,将x与y的对应元素相乘

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| value          | y         | 相乘的矩阵                                     |
| -          | axis         | 维度                                     |
| -          | name         | 表示填充的模式。                                     |

name (str，可选) - 具体用法请参见 [name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。