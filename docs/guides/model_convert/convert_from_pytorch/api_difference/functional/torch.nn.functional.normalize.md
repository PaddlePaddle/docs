## [torch 参数更多 ]torch.nn.functional.normalize

### [torch.nn.functional.normalize](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html?highlight=normalize#torch.nn.functional.normalize)

```python
torch.nn.functional.normalize(input,
                             p=2.0,
                             dim=1,
                             eps=1e-12,
                             out=None)
```

### [paddle.nn.functional.normalize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/normalize_cn.html)

```python
paddle.nn.functional.normalize(x,
                               p=2,
                               axis=1,
                               epsilon=1e-12,
                               name=None)
```

其中 PyTorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输出 Tensor 的 size 。                                     |
| p          | p         | 表示范数公式中的指数值 。                                     |
| dim          | axis         | 表示要进行归一化的轴 。                                     |
| eps          | epsilon         | 表示添加到分母上的值 。                                     |
| out           | -            | 表示输出 Tensor 。               |

### 转写示例
####  out: 指定输出
```python
# PyTorch 写法
torch.nn.functional.normalize(x, out=y)

# Paddle 写法
y = paddle.nn.functional.normalize(x)
```
