## [ 组合替代实现 ]torch.std_mean

### [torch.std_mean](https://pytorch.org/docs/stable/generated/torch.std_mean.html?highlight=std_mean#torch.std_mean)
```python
# 用法一：
torch.std_mean(input, unbiased=True)
# 用法二：
torch.std_mean(input, dim, unbiased=True, keepdim=False)
```

### 功能介绍
用于实现返回 Tensor 的标准差和均值，PaddlePaddle 目前暂无对应 API，可使用如下代码组合实现该 API。

```python
# PyTorch 写法
std, mean = torch.std_mean(x, dim=1)

# Paddle 写法
std = paddle.std(x, axis=1)
mean = paddle.mean(x, axis=1)
```
