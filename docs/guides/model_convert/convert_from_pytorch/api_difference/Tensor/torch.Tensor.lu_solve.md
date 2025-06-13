## [ paddle 参数更多 ]torch.Tensor.lu_solve

### [torch.Tensor.lu_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.lu_solve.html#torch-tensor-lu-solve)

```python
torch.Tensor.lu_solve(LU_data, LU_pivots)
```

### [paddle.linalg.lu_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_solve_cn.html)

```python
paddle.linalg.lu_solve(b, lu, pivots, trans="N", name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，另外 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| self  |   b   | 表示欲进行线性方程组求解的右值 Tensor ，调用 torch.Tensor 类方法的 self Tensor 传入。 |
| LU_data  |   lu   | 表示 LU 分解结果矩阵，由 L、U 拼接组成，仅参数名不一致。 |
| LU_pivots  | pivots       | 表示 LU 分解结果的主元信息 Tensor ，仅参数名不一致。         |
| -  | trans       | 是否对 A 进行转置 ，PyTorch 无此参数，Paddle 保持默认即可。           |
