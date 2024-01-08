## [ torch 参数更多 ]torch.mul

### [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html?highlight=torch+mul#torch.mul)

```python
torch.mul(input, other, *, out=None)
```

### [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multiply_cn.html)

```python
paddle.multiply(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| other     | y           | 表示输入的 Tensor ，仅参数名不一致。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要转写。         |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.mul([3, 5], [1, 2], out=y)

# Paddle 写法
paddle.assign(paddle.multiply([3, 5], [1, 2]),y)
```
