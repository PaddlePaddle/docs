## [ 仅 API 调用方式不一致 ]torch.cuda.nvtx.range_pop

### [torch.cuda.nvtx.range_pop](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_pop.html#torch.cuda.nvtx.range_pop)

```python
torch.cuda.nvtx.range_pop()
```

### [paddle.core.nvprof_nvtx_pop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/framework/core/nvprof_nvtx_pop_cn.html#paddle/core/nvprof_nvtx_pop_cn#cn-api-paddle-core-nvprof_nvtx_pop)

```python
paddle.core.nvprof_nvtx_pop()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.nvtx.range_pop()

# Paddle 写法
result = paddle.core.nvprof_nvtx_pop()
```
