## [ 仅参数名不一致 ] torch.sparse.addmm

### [torch.sparse.addmm](https://pytorch.org/docs/1.13/generated/torch.sparse.addmm.html?highlight=addmm#torch.sparse.addmm)

```python
torch.sparse.addmm(mat, mat1, mat2, beta=1.0, alpha=1.0)
```

### [paddle.sparse.admm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/addmm_cn.html)

```python
paddle.sparse.addmm(input, x, y, beta=1.0, alpha=1.0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

|PyTorch |  PaddlePaddle |  备注   |
|--------|  ------------- | --------------------------------------------------------------------------------------|
|mat | input|         输入 Tensor，仅参数名不一致。|
|mat1 |      x   |输入 Tensor，仅参数名不一致。|
|mat2|y| 输入 Tensor，仅参数名不一致。|
|beta|beta| input 的系数，默认 1.0。两者完全一致|
|alpha|alpha|  x * y 的系数，默认 1.0。两者完全一致|
