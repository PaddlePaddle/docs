## [ torch 参数更多 ]torch.Tensor.subtract
### [torch.Tensor.subtract](https://pytorch.org/docs/stable/generated/torch.Tensor.subtract.html#torch.Tensor.subtract)

```python
torch.Tensor.subtract(other, *, alpha=1)
```

### [paddle.Tensor.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)

```python
paddle.Tensor.subtract(y,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| other         | y            | 表示减数的 Tensor，仅参数名不一致。  |
| alpha         | -            | 表示`other`的乘数，PaddlePaddle 无此参数，需要进行转写。Paddle 应设置 y = alpha * other。  |


### 转写示例
#### alpha：表示`other`的乘数
```python
# Pytorch 写法
x.subtract(y, alpha=2)

# Paddle 写法
x.subtract(2 * y)

# 注：Paddle 直接将 alpha 与 y 相乘实现
```
