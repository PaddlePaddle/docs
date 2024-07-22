## [ torch 参数更多 ]torch.complex


### [torch.complex](https://pytorch.org/docs/stable/generated/torch.complex.html)

```python
torch.complex(real, imag, *, out=None)
```

### [paddle.complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/complex_cn.html#complex)

```python
paddle.complex(real, imag, name=None)
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| real    | real         | 实部，数据类型为：float32 或 float64。 |
| imag    | imag         | 虚部，数据类型和 real 相同。 |
| out     | -          | 输出 Tensor。 |

### 转写示例

```python
# PyTorch 写法
torch.complex(a, b, out=out)

# Paddle 写法
paddle.assign(paddle.complex(a, b), out)
```
