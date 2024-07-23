## [ torch 参数更多 ]torch.linalg.eigvals

### [torch.linalg.eigvals](https://pytorch.org/docs/stable/generated/torch.linalg.eigvals.html?highlight=torch+linalg+eigvals#torch.linalg.eigvals)

```python
torch.linalg.eigvals(input,
                     out=None)
```

### [paddle.linalg.eigvals](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/eigvals_cn.html)

```python
paddle.linalg.eigvals(x,
                      name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
| ------- | ------------ | ---------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                 |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.linalg.eigvals(t, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.eigvals(t), y)
```
