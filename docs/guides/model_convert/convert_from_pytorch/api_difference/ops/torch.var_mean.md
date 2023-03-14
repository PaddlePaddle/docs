## [ 组合替代实现 ]torch.var_mean

### [torch.var_mean](https://pytorch.org/docs/stable/generated/torch.var_mean.html?highlight=var_mean#torch.var_mean)
```python
# 用法一：
torch.var_mean(input,
               unbiased=True)
# 用法二：
torch.var_mean(input,
               dim,
               keepdim=False,
               unbiased=True)
```

### 功能介绍
用于实现返回 Tensor 的方差和均值，PaddlePaddle 目前暂无对应 API，可使用如下代码组合实现该 API。

```python
# PyTorch 写法
var, mean = torch.var_mean(x, dim=1)

# Paddle 写法
var = paddle.var(x, axis=1)
mean = paddle.mean(x, axis=1)
```
