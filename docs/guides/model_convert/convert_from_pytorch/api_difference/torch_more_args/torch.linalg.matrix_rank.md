## [ torch 参数更多 ]torch.linalg.matrix_rank
### [torch.linalg.matrix\_rank](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html#torch.linalg.matrix_rank)
```python
torch.linalg.matrix_rank(input, tol=None, hermitian = False, *, atol=None, rtol=None, out=None)
```

### [paddle.linalg.matrix\_rank](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_rank_cn.html#paddle.linalg.matrix_rank)
```python
paddle.linalg.matrix_rank(x, tol=None, hermitian=False, atol=None, rtol=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input             | x            | 输入的 Tensor ，仅参数名不一致。                         |
| tol           | tol          | 阈值，类型为 Tensor 或者 float，参数完全一致。                         |
| hermitian     | hermitian    | 表示输入矩阵是否是 Hermitian 矩阵，类型为 bool ，参数完全一致。                 |
| atol          | atol         | 绝对阈值，类型为 Tensor 或者 float，参数完全一致。                         |
| rtol          | rtol         | 相对阈值，类型为 Tensor 或者 float，参数完全一致。                         |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。  |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.linalg.matrix_rank(torch.ones(3, 4, 5, 5), tol=0.01, hermitian=True, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.matrix_rank(paddle.ones(shape=[3, 4, 5, 5]), tol=0.01, hermitian=True), y)
```
