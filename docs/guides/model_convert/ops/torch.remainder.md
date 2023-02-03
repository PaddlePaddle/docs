## torch.remainder
### [torch.remainder](https://pytorch.org/docs/stable/generated/torch.remainder.html?highlight=remainder#torch.remainder)

```python
torch.remainder(input,
                other,
                *,
                out=None)
```

### [paddle.remainder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/remainder_cn.html#remainder)

```python
paddle.remainder(x,
                y,
                name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 被除数。                                               |
| other         | y            | 除数。                                                |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 代码示例
``` python
# PyTorch 示例：
torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
# 输出
# tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
# 输出
# tensor([ -0.5000, -1.0000,  0.0000, -0.5000, -1.0000 ])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([2, 3, 8, 7])
y = paddle.to_tensor([1, 5, 3, 3])
z = paddle.remainder(x, y)
print(z)
# 输出
# [0, 3, 2, 1]
```
