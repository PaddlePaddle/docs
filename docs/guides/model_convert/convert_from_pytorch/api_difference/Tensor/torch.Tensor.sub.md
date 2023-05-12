## [ torch 参数更多 ]torch.Tensor.sub
### [torch.Tensor.sub](https://pytorch.org/docs/stable/generated/torch.Tensor.sub.html#torch.Tensor.sub)

```python
torch.Tensor.sub(other, *, alpha=1)
```

### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)

```python
paddle.subtract(x,
                y,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -         | x            | 表示被减数的 Tensor，Pytorch 无此参数。  |
| other         | y            | 表示减数的 Tensor，仅参数名不一致。  |
| alpha         | -            | 表示`other`的乘数，PaddlePaddle 无此参数，需要进行转写。Paddle 应设置 y = alpha * other。  |


### 转写示例
#### alpha：表示`other`的乘数
```python
# Pytorch 写法
x.sub(y, alpha=2)

# Paddle 写法
x - 2 * y

# 注：Paddle 直接将 alpha 与 y 相乘实现
```
