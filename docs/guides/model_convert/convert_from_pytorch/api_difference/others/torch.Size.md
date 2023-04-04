## [ 组合替代实现 ] torch.Generator

### [torch.Size]

```python
shape = torch.Size((1,2,3,4))
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API 转写。

### 转写示例
```python
# torch 写法
shape = torch.Size((1,2,3,4))

# paddle 写法
shape = paddle.shape(paddle.empty((1, 2, 3, 4)))
```
