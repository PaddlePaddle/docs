## [ 参数不一致 ]torch.Tensor.lstsq

### [torch.Tensor.lstsq](https://pytorch.org/docs/1.9.0/generated/torch.Tensor.lstsq.html?highlight=torch%20tensor%20lstsq#torch.Tensor.lstsq)

```python
torch.Tensor.lstsq(A)
```

### [paddle.Tensor.lstsq]()

```python
paddle.Tensor.lstsq(y, rcond=None, driver=None, name=None)
```

两者功能一致，参数不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                        |
| ------- | ------------ | --------------------------------------------------------------------------- |
| A       | y            | 表示输入的 Tensor ，PyTorch 和 Paddle 参数相反，需要转写。                  |
| -       | rcond        | 用来决定 x 有效秩的 float 型浮点数。PyTorch 无此参数，Paddle 保持默认即可。 |
| -       | driver       | 用来指定计算使用的 LAPACK 库方法。PyTorch 无此参数，Paddle 保持默认即可。   |

### 转写示例

#### A 参数转写

```python
# PyTorch 写法:
y = x.lstsq(A)

# Paddle 写法:
y = A.lstsq(x)
```
