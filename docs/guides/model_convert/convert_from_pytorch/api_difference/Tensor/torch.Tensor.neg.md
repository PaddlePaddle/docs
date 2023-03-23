## [ 无参数 ] torch.Tensor.neg

### [torch.Tensor.neg](https://pytorch.org/docs/1.13/generated/torch.Tensor.neg.html?highlight=neg#torch.Tensor.neg)

```python
torch.Tensor.neg()

# 示例代码
import torch

x = torch.Tensor(([2, float('-nan'), 3]))
x.neg()
print(x)    # [-2., nan, -3.]
```

### [paddle.Tensor.neg](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/neg_cn.html)

```python
paddle.Tensor.neg()

# 示例代码
import paddle

x = paddle.to_tensor([2, float('-nan'), 3])
x.neg()
print(x)    # [-2., nan, -3.]
```

两者功能一致，将张量x上的各个值取相反数。