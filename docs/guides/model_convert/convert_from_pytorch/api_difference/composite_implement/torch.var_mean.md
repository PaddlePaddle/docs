## [ 组合替代实现 ]torch.var_mean
### [torch.var\_mean](https://docs.pytorch.org/docs/stable/generated/torch.var_mean.html#torch.var_mean)
```python
torch.var_mean(input, dim=None, unbiased=True, keepdim=False, *, correction=None)
```
用于实现返回 Tensor 的方差和均值，PaddlePaddle 目前暂无对应 API，可使用如下代码组合实现该 API。

> 注：torch 旧版本额外重载了 torch.var_mean(input, unbiased=True) 的签名，该用法未提供转写示例

### 转写示例
```python
# PyTorch 写法
var, mean = torch.var_mean(x, dim=1)

# Paddle 写法
var = paddle.var(x, axis=1)
mean = paddle.mean(x, axis=1)
```
