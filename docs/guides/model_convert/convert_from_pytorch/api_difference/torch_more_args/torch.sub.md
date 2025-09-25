## [ torch 参数更多 ]torch.sub
### [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html?highlight=torch%20sub#torch.sub)

```python
torch.sub(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/subtract_cn.html#subtract)

```python
paddle.subtract(x,
                y,
                name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示被减数的 Tensor，仅参数名不一致。  |
| other         | y            | 表示减数的 Tensor，仅参数名不一致。  |
| alpha         | -            | 表示`other`的乘数，Paddle 无此参数，需要转写。 Paddle 应设置 y = alpha * other。   |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。  |


### 转写示例
#### alpha：表示`other`的乘数
```python
# PyTorch 写法
torch.sub(x, y, alpha=2)

# Paddle 写法
paddle.subtract(x, 2*y)

# 注：Paddle 直接将 alpha 与 y 相乘实现
```
#### out：指定输出
```python
# PyTorch 写法
torch.sub(x, y, out=z)

# Paddle 写法
paddle.assign(paddle.subtract(x, y), z)
```
