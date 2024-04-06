## [ torch 参数更多]torch.bmm

### [torch.bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=bmm#torch.bmm)

```python
torch.bmm(input,mat2,*,out=None)
```

### [paddle.bmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bmm_cn.html)

```python
paddle.bmm(x,y,name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| input | x | 表示输入的第一个 Tensor ，仅参数名不一致。 |
| mat2 | y | 表示输入的第二个 Tensor ，仅参数名不一致。 |
| out | - | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out: 输出的 Tensor

```python
# PyTorch 写法
torch.bmm(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.bmm(x, y), output)
```
