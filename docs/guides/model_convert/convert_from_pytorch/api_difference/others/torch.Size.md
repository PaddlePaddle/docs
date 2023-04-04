## [ 仅参数名不一致 ] torch.Size

### [torch.Size]

```python
shape = torch.Size((1,2,3,4))
```

两者功能一致，仅参数名不一致，具体如下：

### 转写示例
```python
# torch 写法
shape = torch.Size((1,2,3,4))

# paddle 写法
shape = paddle.empty((1, 2, 3, 4)).shape
```
