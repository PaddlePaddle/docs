## [ torch 参数更多]torch.conj_physical
### [torch.conj_physical](https://pytorch.org/docs/stable/generated/torch.conj_physical.html#torch.conj_physical)

```python
torch.conj_physical(input, *, out=None)
```

### [paddle.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/conj_cn.html#conj)

```python
paddle.conj(x,
            name=None)
```

PyTorch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | - | 表示输出的 Tensor ，paddle 无此参数， 需要转写。  |

#### out：指定输出
```python
# PyTorch 写法
torch.conj_physical(input, out=out)

# Paddle 写法
paddle.assign(paddle.conj(input), output=out)
```
