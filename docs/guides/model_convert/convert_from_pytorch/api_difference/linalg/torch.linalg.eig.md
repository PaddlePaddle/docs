## [ torch 参数更多 ]torch.linalg.eig

### [torch.linalg.eig](https://pytorch.org/docs/stable/generated/torch.linalg.eig.html?highlight=torch+linalg+eig#torch.linalg.eig)

```python
torch.linalg.eig(input, *, out=None)
```

### [paddle.linalg.eig](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/eig_cn.html)

```python
paddle.linalg.eig(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                                            |
| ------- | ------------ | ----------------------------------------------- |
| input   | x            | 表示输入的 Tensor，仅参数名不一致。            |
| out     | -            | 表示输出的 tuple，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.eig(t, out=(L,V))

# Paddle 写法
L,V=paddle.linalg.eig(t)
```
