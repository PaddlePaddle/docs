## [ 仅参数名不一致 ]torch.linalg.lstsq
### [torch.linalg.lstsq](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html?highlight=lstsq#torch.linalg.lstsq)

```python
torch.linalg.lstsq(input, b, rcond=None, *, driver=None)
```

### [paddle.linalg.lstsq](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lstsq_cn.html)

```python
paddle.linalg.lstsq(x, y, rcond=None, driver=None, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x         | 表示输入的 Tensor，仅参数名不一致。                            |
| b             | y           | 表示输入的 Tensor，仅参数名不一致。                          |
| rcond         | rcond           | 用来决定 x 有效秩的 float 型浮点数。                     |
| driver        | driver           | 用来指定计算使用的 LAPACK 库方法。                      |
