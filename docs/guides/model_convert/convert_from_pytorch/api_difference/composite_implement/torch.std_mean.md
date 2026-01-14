## [ 组合替代实现 ]torch.std_mean
### [torch.std_mean](https://pytorch.org/docs/stable/generated/torch.std_mean.html?highlight=std_mean#torch.std_mean)
```python
torch.std_mean(input, dim=None, unbiased=True, keepdim=False, *, correction=None)
```
用于实现返回 Tensor 的标准差和均值，PaddlePaddle 目前暂无对应 API，可使用如下代码组合实现该 API。

### 转写示例
```python
# PyTorch 写法
std, mean = torch.std_mean(x, dim=1)
std, mean = torch.std_mean(x, True) # torch 支持 unbiased 以第二个位置参数的形式传入

# Paddle 写法
std = paddle.std(x, axis=1)
mean = paddle.mean(x, axis=1)
```
