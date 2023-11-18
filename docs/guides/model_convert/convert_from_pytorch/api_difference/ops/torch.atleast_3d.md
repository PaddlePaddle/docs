## [ 参数不一致 ]torch.atleast_3d

### [torch.atleast_3d](https://pytorch.org/docs/stable/generated/torch.atleast_3d.html#torch-atleast-3d)

```python
torch.atleast_3d(*tensors)
```

### [paddle.atleast_3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/atleast_3d_cn.html#atleast_3d)

```python
paddle.atleast_3d(*inputs, name=None)
```

PyTorch 与 Paddle 参数形式上一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> tensors </font> | <font color='red'> inputs </font> | 输入的 Tensor ，仅参数名不一致。 |

PyTorch 与 Paddle 功能一致，但对于多个 Tensor 输入的处理方式略有不同，具体请看转写示例。

### 转写示例

#### 多个 Tensor
```python
# Pytorch 写法
x = torch.tensor(0.3)
y = torch.tensor(0.4)
# 可以写为多个输入的方式
torch.atleast_3d(x, y)
# 或者组合为一个 tuple，这两种方式是等价的
torch.atleast_3d((x, y))

# Paddle 写法
x = paddle.to_tensor(0.3)
y = paddle.to_tensor(0.4)
# 这里只能分别传入
paddle.atleast_3d(x, y)
# 下面这种方式会将 (x, y) 经过 broadcast 后作为单个输入
paddle.atleast_3d((x, y))

```
