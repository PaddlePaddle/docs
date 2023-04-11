## [ 仅参数名不一致 ]torch.linalg.lstsq
### [torch.linalg.lstsq](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html?highlight=lstsq#torch.linalg.lstsq)

```python
torch.linalg.lstsq(A, B, rcond=None, *, driver=None)
```

### [paddle.linalg.lstsq](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/lstsq_cn.html)

```python
paddle.linalg.lstsq(x, y, rcond=None, driver=None, name=None)
```

两者功能完全一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A         | x         | 表示输入的 Tensor 。                                     |
| B           | y           | 表示输入的 Tensor 。     |
| rcond           | rcond           | 用来决定 x 有效秩的 float 型浮点数。               |
| driver           | driver           | 用来指定计算使用的 LAPACK 库方法。               |
