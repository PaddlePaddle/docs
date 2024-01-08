## [ torch 参数更多]torch.tensordot

### [torch.tensordot](https://pytorch.org/docs/stable/generated/torch.tensordot.html?highlight=tensordot#torch.tensordot)

```python
torch.tensordot(a,b,dims=2,out=None)
```

### [paddle.tensordot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensordot_cn.html)

```python
paddle.tensordot(x,y,axes=2,name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| a | x | 表示缩并运算的左张量,仅参数名不一致。 |
| b | y | 表示缩并运算的右张量，仅参数名不一致。 |
| dims | axes | 表示对张量做缩并运算的轴，默认值为 2 ，仅参数名不一致。 |
| out | - | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out: 输出的 Tensor

```python
# PyTorch 写法
torch.tensordot(x,y,axes,out=output)

# Paddle 写法
paddle.assign(paddle.tensordot(x,y,axes),output)
```
