## [ torch 参数更多 ] torch.special.sinc

### [torch.special.sinc](https://pytorch.org/docs/stable/special.html#torch.special.sinc)

```python
torch.special.sinc(input, *, out=None)
```

### [paddle.sinc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sinc_cn.html#sinc)

```python
paddle.sinc(x, name=None)
```

### 参数映射

| PyTorch       | PaddlePaddle | 备注                    |
| ------------- | ------------ | ----------------------------------------------------------------------------- |
| input      | x  | 表示输出的 Tensor， 仅参数名不一致  |
| out         | -  | 表示输出的 Tensor，Paddle 无此参数。 |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.special.sinc(input, out)

# Paddle 写法
paddle.assign(paddle.sinc(input), output=out)
```
