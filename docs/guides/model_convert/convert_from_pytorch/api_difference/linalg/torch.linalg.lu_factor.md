## [ torch 参数更多 ]torch.linalg.lu_factor

### [torch.linalg.lu_factor](https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor)

```python
torch.linalg.lu_factor(A, *, pivot=True, out=None)
```

### [paddle.linalg.lu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_cn.html)

```python
paddle.linalg.lu(x, pivot=True, get_infos=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                  |
| ------- | ------------ | ----------------------------------------------------- |
| A       | x            | 表示需要进行 LU 分解的输入 Tensor ，仅参数名不一致。  |
| pivot   | pivot        | 表示 LU 分解时是否进行旋转。                          |
| -       | get_infos    | 表示是否返回分解状态信息 ， Paddle 保持默认即可。     |
| out     | -            | 表示输出的 Tensor 元组 ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.linalg.lu_factor(A, out=(LU, pivots))

# Paddle 写法
y = paddle.linalg.lu(A)
paddle.assign(y[0], out[0]), paddle.assign(y[1], out[1])
```
