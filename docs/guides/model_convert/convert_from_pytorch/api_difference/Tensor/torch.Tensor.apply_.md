## [ 参数完全一致 ]torch.Tensor.apply_
### [torch.Tensor.apply_](https://pytorch.org/docs/stable/generated/torch.Tensor.apply_.html)

```python
torch.Tensor.apply_(callable)
```

### [paddle.Tensor.apply_]()

```python
paddle.Tensor.apply_(callable)
```


两者功能不同，pytorch 只支持 Inplace 操作，且只对 CPU tensor 才能使用。Paddle 版本可以在 GPU 下运行同时也有 outplace 版本。两者参数一致，具体如下：

### 参数映射
| PyTorch  | PaddlePaddle | 备注        |
|----------|--------------|-----------|
| callable | callable     | 一个被调用的函数。 |
