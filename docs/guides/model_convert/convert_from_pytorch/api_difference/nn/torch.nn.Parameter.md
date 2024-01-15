## [ 组合替代实现 ]torch.nn.Parameter

### [torch.nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch+nn+parameter#torch.nn.parameter.Parameter)

```python
torch.nn.Parameter(data=None, requires_grad=True)
```
Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
y = torch.nn.Parameter(data=x, requires_grad=requires_grad)

# Paddle 写法
y = paddle.create_parameter(shape=x.shape,
                        dtype=x.dtype,
                        default_initializer=paddle.nn.initializer.Assign(x))
y.stop_gradient = not requires_grad
```
