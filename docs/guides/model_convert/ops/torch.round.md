## torch.round
### [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round)

```python
torch.round(input, 
            *, 
            decimals=0, 
            out=None)
```

### [paddle.round](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/round_cn.html#round)

```python
paddle.round(x, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| decimals      | -            | 要舍入到的小数位数，PaddlePaddle无此参数。               |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
torch.round(torch.tensor((4.7, -2.3, 9.1, -7.7)))
# 输出
# tensor([ 5.,  -2.,  9., -8.])

# Values equidistant from two integers are rounded towards the
#   the nearest even value (zero is treated as even)
torch.round(torch.tensor([-0.5, 0.5, 1.5, 2.5]))
# 输出
# tensor([-0., 0., 2., 2.])

# A positive decimals argument rounds to the to that decimal place
torch.round(torch.tensor([0.1234567]), decimals=3)
# 输出
# tensor([0.1230])

# A negative decimals argument rounds to the left of the decimal
torch.round(torch.tensor([1200.1234567]), decimals=-3)
# 输出
# tensor([1000.])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
out = paddle.round(x)
print(out)
# 输出
# [-1. -0.  1.  2.]
```
