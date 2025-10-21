## [ 仅 API 调用方式不一致 ]torch.Tensor.lu_solve
### [torch.Tensor.lu_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.lu_solve.html#torch-tensor-lu-solve)
```python
torch.Tensor.lu_solve(LU_data, LU_pivots)
```

### [paddle.linalg.lu_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_solve_cn.html)
```python
paddle.linalg.lu_solve(b, lu, pivots, trans="N", name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，另外 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
