## [ 无参数 ] torch.Tensor.is_sparse

### [torch.Tensor.is_sparse](https://pytorch.org/docs/stable/generated/torch.Tensor.is_sparse.html)

```python
torch.Tensor.is_sparse
```

### [paddle.Tensor.is_sparse](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html)

```python
paddle.Tensor.is_sparse()
```

两者功能一致，但使用方式不一致，前者可以直接访问属性，后者需要调用方法，具体如下：

### 转写示例
```
# torch 版本可以直接访问属性
# x = torch.rand(3)
# print(x.is_sparse)

# Paddle 版本需要调用
x = paddle.rand([3])
print(x.is_sparse())
```
