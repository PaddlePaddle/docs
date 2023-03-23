## [ 无参数 ] torch.Tensor.ndimension

### [torch.Tensor.ndimension](https://pytorch.org/docs/1.13/generated/torch.Tensor.ndimension.html?highlight=ndimension#torch.Tensor.ndimension)

```python
torch.Tensor.ndimension

# 代码示例
import torch

a = torch.ones(2, 3, 4)
print(a.ndimension())  # 3
```

### [paddle.Tensor.ndimension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)未找到文档

```python
paddle.Tensor.place

# 代码示例
import paddle

a = paddle.ones([2, 3, 4])
print(a.ndimension())  # 3
```

两者功能一致，返回张量的维度