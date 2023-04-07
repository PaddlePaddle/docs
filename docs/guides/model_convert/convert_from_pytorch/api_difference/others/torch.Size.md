## [ 组合替代实现 ] torch.Size

### [torch.Size]

```python
torch.Size((1,2,3,4))
```

Pytorch 中 torch.Size 返回 tensor 形状, PaddlePaddle 目前无对应 API，可通过 list 实现，可使用如下代码组合实现该 API 转写。

### 转写示例
```python
# torch 写法
torch.Size((1,2,3,4))

# paddle 写法
list((1, 2, 3, 4))
```
