## [ torch 参数更多 ]torch.ldexp

### [torch.ldexp](https://pytorch.org/docs/stable/generated/torch.ldexp.html#torch.ldexp)

```python
torch.ldexp(input, other, *, out=None)
```

### [paddle.ldexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/ldexp_cn.html#ldexp)

```python
paddle.ldexp(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  other  | y  | 表示输入的 Tensor ，仅参数名不一致。    |
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |

### 转写示例

```python
# PyTorch 写法
torch.ldexp(a, b, out=output)

# Paddle 写法
paddle.assign(paddle.ldexp(a,b), output=output)
