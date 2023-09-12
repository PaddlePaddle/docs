## [ 仅参数名不一致 ] torch.sparse.mm

### [torch.sparse.mm](https://pytorch.org/docs/stable/generated/torch.sparse.mm.html?highlight=torch+sparse+mm#torch.sparse.mm)

```python
# PyTorch 文档有误，测试 PyTorch 参数名为 sparse, dense
torch.sparse.mm(sparse, dense)
```

### [paddle.sparse.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/matmul_cn.html)

```python
paddle.sparse.matmul(x, y, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

 |PyTorch |  PaddlePaddle |  备注|
 |--------|  ------------- | ------|
 |sparse | x|         输入的 Tensor，仅参数名不一致。|
 |dense   |      y   |输入的第二个 Tensor，仅参数名不一致。|
