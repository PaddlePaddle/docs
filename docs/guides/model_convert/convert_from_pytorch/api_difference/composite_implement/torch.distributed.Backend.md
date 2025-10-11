## [组合替代实现]torch.distributed.Backend

### [torch.distributed.Backend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend)

```python
torch.distributed.Backend(name)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.distributed.Backend("GLOO")

# Paddle 写法
"gloo"
```

```python
# PyTorch 写法
torch.distributed.Backend("NCCL")

# Paddle 写法
"nccl"
```


```python
# PyTorch 写法
torch.distributed.Backend("UCC")

# Paddle 写法
"ucc"
```


```python
# PyTorch 写法
torch.distributed.Backend("MPI")

# Paddle 写法
"mpi"
```


```python
# PyTorch 写法
torch.distributed.Backend("XCCL")

# Paddle 写法
"xccl"
```
