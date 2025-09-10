## [torch 参数更多]torch.special.i0e

### [torch.special.i0e](https://pytorch.org/docs/stable/special.html#torch.special.i0e)

```python
torch.special.i0e(input, *, out=None)
```

### [paddle.i0e](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/i0e_cn.html)

```python
paddle.i0e(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| input   | x            | 表示输入的 Tensor，仅参数名不一致。                |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
torch.special.i0e(x, out=y)

# Paddle 写法
x = paddle.to_tensor([1, 2, 3, 4, 5], dtype="float32")
paddle.assign(paddle.i0e(x), y)
```
