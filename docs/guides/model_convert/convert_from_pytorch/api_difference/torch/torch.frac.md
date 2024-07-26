## [torch 参数更多 ]torch.frac
### [torch.frac](https://pytorch.org/docs/stable/generated/torch.frac.html?highlight=frac#torch.frac)

```python
torch.frac(input,
            *,
            out=None)
```

### [paddle.frac](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/frac_cn.html#frac)

```python
paddle.frac(x,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |



### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.frac(input, out=y)

# Paddle 写法
paddle.assign(paddle.frac(input), y)
```
