## [ torch 参数更多 ]torch.logaddexp

### [torch.logaddexp](https://pytorch.org/docs/stable/generated/torch.logaddexp.html#torch.logaddexp)

```python
torch.logaddexp(input, other, *, out=None)
```

### [paddle.logaddexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logaddexp_cn.html#logaddexp)

```python
paddle.logaddexp(x, y, name=None)
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
torch.logaddexp(a, b, out=output)

# Paddle 写法
paddle.assign(paddle.logaddexp(a,b), output=output)
```
