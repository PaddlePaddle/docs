## [ 组合替代实现 ]torch.Tensor.resize_as_

### [torch.Tensor.resize_as_](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_as_.html?highlight=resize_as#torch.Tensor.resize_as_)

```python
# PyTorch 文档有误，测试第一个参数为 the_template
torch.Tensor.resize_as_(the_template, memory_format=torch.contiguous_format)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.resize_as_(b)

# Paddle 写法
y = a.reshape_(b.shape)
```
