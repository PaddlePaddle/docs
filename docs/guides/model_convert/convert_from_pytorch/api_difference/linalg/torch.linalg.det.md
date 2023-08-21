## [ torch 参数更多 ]torch.linalg.det
### [torch.linalg.det](https://pytorch.org/docs/stable/generated/torch.linalg.det.html#torch.linalg.det)

```python
torch.linalg.det(A, *, out=None)
```

### [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/det_cn.html#det)

```python
paddle.linalg.det(x)
```

torch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> A </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | <font color='red'> - </font> | 表示输出 Tensor， Paddle 无此参数，需要转写。  |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.det(x, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.det(x), y)
```
