## [ 一致的参数 ] torch.Tensor.mv
### [torch.Tensor.mv](https://pytorch.org/docs/1.13/generated/torch.Tensor.mv.html)

```python
torch.Tensor.mv(vec)

import torch

x = torch.Tensor([[2, 1, 3], [3, 0, 1]])
vec = torch.Tensor([3, 5, 1])
out = x.mv(vec)
print(out)    #[14., 10.])
```

### [paddle.Tensor.mv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mv_cn.html)

```python
paddle.Tensor.mv(vec)

# x: [M, N], vec: [N]
# paddle.mv(x, vec)  # out: [M]

import paddle

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1]]).astype("float64")
vec = paddle.to_tensor([3, 5, 1]).astype("float64")
out = x.mv(vec)
print(out)    #[14., 10.])
```
两者功能一致，计算张量 x 和向量 vec 的乘积。

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| vec          | vec         | 相乘的向量                                     |
