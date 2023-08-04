## [torch 参数更多 ]torch.linalg.matrix_rank
### [torch.linalg.matrix_rank](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html?highlight=matrix_rank#torch.linalg.matrix_rank)
```python
torch.linalg.matrix_rank(A, tol=None, hermitian=False, *, out=None)
```

### [paddle.linalg.matrix_rank](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/matrix_rank_cn.html)
```python
paddle.linalg.matrix_rank(x, tol=None, hermitian=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A             | x            | 输入的 Tensor ，仅参数名不一致。                         |
| tol           | tol          | 输入的 Tensor 或者 float，参数完全一致。                         |
| hermitian     | hermitian    | 输入的 bool ，参数完全一致。                            |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.linalg.matrix_rank(torch.ones(3, 4, 5, 5), tol=0.01, hermitian=True, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.matrix_rank(paddle.ones(shape=[3, 4, 5, 5]), tol=0.01, hermitian=True), y)
```
