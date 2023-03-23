## [ 一致的参数 ] torch.Tensor.ne
### [torch.Tensor.ne](https://pytorch.org/docs/1.13/generated/torch.Tensor.ne.html?highlight=ne)

```python
torch.Tensor.ne(other)

# 代码示例
import torch

a = torch.Tensor([[1, 1],[1, 1]])
b = torch.Tensor([[1, 0],[0, 1]])
print(a.ne(b))  # tensor([[False,  True],[ True, False]])
```

### [paddle.Tensor.not_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/not_equal_cn.html)

```python
paddle.Tensor.not_equal(y)

# 代码示例
import paddle

x = paddle.to_tensor([[1, 1],[1, 1]])
y = paddle.to_tensor([[1, 0],[0, 1]])
print(x.not_equal(y))   # [[False, True ],[True , False]]
```

两者功能一致，返回 x!=y逐元素比较 x 和 y 是否相等，相同位置的元素不相同则返回 True，否则返回 False。
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| other          | y         | 比较的矩阵                                     |
