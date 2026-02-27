## [ torch 参数更多 ]torch.signbit
### [torch.signbit](https://docs.pytorch.org/docs/stable/generated/torch.signbit.html#torch.signbit)
```python
torch.signbit(input, *, out=None)
```

### [paddle.signbit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/signbit_cn.html#paddle.signbit)
```python
paddle.signbit(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                           |
| ------- | ------------ | ---------------------------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。                  |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.signbit([1., 2., 3., -1.], out=y)

# Paddle 写法
paddle.assign(paddle.signbit([1., 2., 3., -1.]), y)
```
