## [ torch 参数更多]torch.lcm

### [torch.lcm](https://pytorch.org/docs/stable/generated/torch.lcm.html#torch-lcm)

```python
torch.lcm(input, other, *, out=None)
```

### [paddle.lcm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/lcm_cn.html#lcm)

```python
paddle.lcm(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| input   | x            | 表示输入的第一个 Tensor ，仅参数名不一致。          |
| other   | y            | 表示输入的第二个 Tensor ，仅参数名不一致。           |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.lcm(x,y, out=output)

# Paddle 写法
paddle.assign(paddle.lcm(x,y), output)
```
