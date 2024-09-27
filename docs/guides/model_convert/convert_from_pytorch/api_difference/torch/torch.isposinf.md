## [torch 参数更多]torch.isposinf

### [torch.isposinf](https://pytorch.org/docs/stable/generated/torch.isposinf.html#torch-isposinf)

```python
torch.isposinf(input, *, out=None)
```

### [paddle.isposinf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/isposinf_cn.html)

```python
paddle.isposinf(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                             |
| ------ | ----------- | ------------------------------------------------ |
| input   | x            | 输入的 Tensor，仅参数名不一致。                   |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写实例

#### out 参数：指定输出

```python
# Pytorch 写法
torch.isposinf(x, out=y)

# Paddle 写法
paddle.assign(paddle.isposinf(x), y)
```
