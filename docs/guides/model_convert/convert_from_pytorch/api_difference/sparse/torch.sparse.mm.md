## [ 仅参数名不一致 ] torch.sparse.mm

### [torch.sparse.mm](https://pytorch.org/docs/1.13/generated/torch.sparse.mm.html?highlight=torch+sparse+mm#torch.sparse.mm)

```python
torch.sparse.mm(input, mat2)
```

### [paddle.sparse.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/matmul_cn.html)

```python
paddle.sparse.matmul(x, y, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

 |PyTorch |  PaddlePaddle |  备注|
 |--------|  ------------- | ------|
 |input | x|         输入的 Tensor，仅参数名不一致。|
 |mat2   |      y   |输入的第二个 Tensor，仅参数名不一致。|
