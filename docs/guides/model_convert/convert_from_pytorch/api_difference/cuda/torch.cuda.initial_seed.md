## [组合替代实现]torch.cuda.initial_seed

### [torch.cuda.initial_seed](https://pytorch.org/docs/stable/generated/torch.cuda.initial_seed.html?highlight=torch+cuda+initial_seed#torch.cuda.initial_seed)

```python
torch.cuda.initial_seed()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.cuda.initial_seed()

# Paddle 写法
paddle.get_cuda_rng_state()[0].current_seed()
```
