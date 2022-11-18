# Introduction of IR Schedule and Schedule Primitives

## IR Schedule

In the CINN framework, we defined the behavior of the operator through the computation module and optimize the process of computation of the operator (i.e., the generated source code) through the schedule module. The IR is the abstraction of the operator's code representation (essentially an abstract syntax tree), and the IR Schedule serves to modify the optimized IR to generate higher-performance code. The schedule primitives are the base API module to achieve this modification, and developers can use different schedule primitives to accomplish the optimization of the computation.

## Schedule Primitives

Current CINN schedule primitives fall into three broad categories:

- Loop Transformation：Fuse, Split, Unroll ...

- Storage Hierarchy：CacheRead, CacheWrite ...

- Parallel Optimization：Vectorize, Bind, Parallel ...


source code reference: `cinn/ir/ir_schedule.h`

### Loop Transformation

#### Fuse

Fuse multiple for loops and return the fused loop. Examples are as follows：

```c
// Original IR：
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 64)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }

// IR After Fuse：
  ScheduleBlock(root)
  {
    serial for (i_j_fused, 0, 2048)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind((i_j_fused / 64), (i_j_fused % 64))
        B[i0, i1] = A[i0, i1]
      }
    }
  }
```

#### Split

Split a for loop into multiple loops, based on the factors. Examples are as follows：

```c
// Original IR：
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 64)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
         B[i0, i1] = A[i0, i1]
        }
      }
    }
  }

// IR After Split loop `j` with factors {4,16}：
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j_0, 0, 4)
      {
        serial for (j_1, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0, i1 = axis.bind(i, ((16 * j_0) + j_1))
            B[i0, i1] = A[i0, i1]
          }
        }
      }
    }
  }
```

#### Reorder

Reorder the loops in specific order. Examples are as follows：

```c
// Original IR：
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 64)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
         B[i0, i1] = A[i0, i1]
        }
      }
    }
  }

// IR After Reorder：
  ScheduleBlock(root)
  {
    serial for (j, 0, 64)
    {
      serial for (i, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
```

#### Unroll

Unroll a certain loop. Examples are as follows：

```c
// Original Generated Code：
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; i < 2; j += 1) {
      B[(2 * i) + j] = A[(2 * i) + j];
    }
  };


// Generated Code After Unroll：
  for (int32_t i = 0; i < 32; i += 1) {
    B[(2 * i)] = A[(2 * i)];
    B[(1 + (2 * i))] = A[(1 + (2 * i))];
  };
```

#### ComputeInline

Mark an schedule block as inlined. Examples are as follows：

```c
// Original IR：
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * B[i1, i0, i2])
            }
          }
        }
      }
    }
  }
// IR After ComputeInline：
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * (1 + A[i1, i0, i2]))
            }
          }
        }
      }
    }
  }
```

#### ComputeAt

Move a block's location under a loop. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * B[i1, i0, i2])
            }
          }
        }
      }
    }
  }
// IR After doing a Fuse for C's first two loops:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i_j_fused, 0, 1024)
      {
        serial for (k, 0, 32)
        {
          ScheduleBlock(C)
          {
            i0, i1, i2 = axis.bind((i_j_fused / 32), (i_j_fused % 32), k)
            C[i0, i1, i2] = (2 * B[i1, i0, i2])
          }
        }
      }
    }
  }

// IR After ComputeAt. Since B will inherit the changes of the relevant loop in C after ComputeAt, B will also do a Fuse automatically, generating the loop `i_j_fused` layer, with the following result:
  ScheduleBlock(root)
  {
    {
      serial for (i_j_fused, 0, 1024)
      {
        serial for (k, 0, 32)
        {
          ScheduleBlock(B)
          {
            i0, i1, i2 = axis.bind((i_j_fused % 32), (i_j_fused / 32), k)
            B[i0, i1, i2] = (1 + A[i0, i1, i2])
          }
          ScheduleBlock(C)
          {
            i0, i1, i2 = axis.bind((i_j_fused / 32), (i_j_fused % 32), k)
            C[i0, i1, i2] = (2 * B[i1, i0, i2])
          }
        }
      }
    }
  }
```

#### SimpleComputeAt

Move the position of a Tensor calculation.
The main difference between SimpleComputeAt and ComputeAt is that:
- ComputeAt requires that the Tensor being moved is subsequently used, e.g. `B = A + 1`, `C = B + 1`, then moving B to a certain loop of C satisfies the requirement (B is used by C), while SimpleComputeAt does not have this requirement.
- When SimpleComputeAt moves B to the n-th loop of C, it needs the range of first n loops of B and C to be indentical. For example, if B moves to the 3rd loop of C, then the range of the first 3 loops of B and C must be the same, while ComputeAt does not have this requirement (because it automatically inherits the transformations of C's loops).
Examples are as follows.

```c
// Original IR:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * A[i1, i0, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
          }
        }
      }
    }
  }
// IR After SimpleComputeAt:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
            {
              ScheduleBlock(C)
              {
                i0, i1, i2 = axis.bind(i, j, k)
                C[i0, i1, i2] = (2 * A[i1, i0, i2])
              }
            }
          }
        }
      }
    }
  }
```
In this example, we cannot use ComputeAt because `C = 2 * A` does not depend on (use) `B = 1 + A`.
Similarly, in the ComputeAt example, we can't use SimpleComputeAt because `C = 2 * A` does not have the same range as `B = 1 + A` for the first Loop after doing a Fuse. You need to do the same Fuse manually for `B = 1 + A` to use SimpleComputeAt.

### Storage Hierarchy

#### CacheRead

Find a buffer that is being read, and create its cache. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        serial for (k, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0, i1, i2 = axis.bind(i, j, k)
            B[i0, i1, i2] = A[i0, i1]
          }
        }
      }
    }
  }

// IR After CacheRead with "shared" memory_type:
  ScheduleBlock(root)
  {
    {
      serial for (cache_ax0, 0, 32)
      {
        serial for (cache_ax1, 0, 32)
        {
          ScheduleBlock(A_shared_temp_buffer)
          {
            v0, v1 = axis.bind(cache_ax0, cache_ax1)
            {
              A_shared_temp_buffer[v0, v1] = A[v0, v1]
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 16)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              {
                B[i0, i1, i2] = A_shared_temp_buffer[i0, i1]
              }
            }
          }
        }
      }
    }
  }
```

You can see that after doing a CacheRead on A and caching A's data to shared memory, the 32 * 32 * 16 read operations on global memory become 32 * 32 read operations on global memory plus 32 * 32 * 16 read operations on shared memory. Since the read speed of shared memory is much faster than that of global memory, the IR will run faster after doing CacheRead.

#### CacheWrite

Find a buffer that is being written, and create its cache. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 64)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0, i1 = axis.bind(i, j)
          B__reduce_init[i0, i1] = 0
        }
        serial for (k0, 0, 32)
        {
          ScheduleBlock(B)
          {
            i0, i1, i2 = axis.bind(i, j, k0)
            B[i0, i1] = (B[i0, i1] + A[i0, i1, i2])
          }
        }
      }
    }
  }
// IR After CacheWrite with "shared" memory_type:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 64)
      {
        serial for (j, 0, 32)
        {
          ScheduleBlock(B__reduce_init)
          {
            i0, i1 = axis.bind(i, j)
            {
              B_shared_temp_buffer__reduce_init[i0, i1] = 0
            }
          }
          serial for (k0, 0, 32)
          {
            ScheduleBlock(B_shared_temp_buffer)
            {
              i0, i1, i2 = axis.bind(i, j, k0)
              {
                B_shared_temp_buffer[i0, i1] = (B_shared_temp_buffer[i0, i1] + A[i0, i1, i2])
              }
            }
          }
        }
      }
      serial for (cache_ax0, 0, 64)
      {
        serial for (cache_ax1, 0, 32)
        {
          ScheduleBlock(B)
          {
            v0, v1 = axis.bind(cache_ax0, cache_ax1)
            {
              B[v0, v1] = B_shared_temp_buffer[v0, v1]
            }
          }
        }
      }
    }
  }
```

You can see that after doing CacheWrite on B and caching the result of B to shared memory first, the 64 * 32 * 32 write operations on global memory become 64 * 32 write operations on global memory plus 64 * 32 * 32 write operations on shared memory. Since the write speed of shared memory is much faster than that of global memory, the IR will run faster after CacheWrite.

#### SetBuffer

Set a tensor's buffer type(memory_type). Currently the available memory_type are `local`, `shared` and `global`. Examples are as follows：

```c
// Set Tensor a's memory type as local memory in cuda backends.
ir_sch.SetBuffer(a, "local");
```

#### Rfactor

Factorize the reduction block by the given loop. The block will be split into two blocks: the rfactor block and the final write-back block. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      ScheduleBlock(B__reduce_init)
      {
        i0 = axis.bind(i)
        B__reduce_init[i0] = 0
      }
      serial for (j0, 0, 2)
      {
        serial for (k0, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0, i1, i2 = axis.bind(i, j0, k0)
            B[i0] = (B[i0] + A[i0, i1, i2])
          }
        }
      }
    }
  }
// IR After Rfactor:
  ScheduleBlock(root)
  {
    {
      serial for (rf_k0, 0, 16)
      {
        serial for (i, 0, 32)
        {
          ScheduleBlock(rf_B__reduce_init)
          {
            i0, i1 = axis.bind(i, rf_k0)
            rf_B__reduce_init[i1, i0] = 0
          }
          serial for (j0, 0, 2)
          {
            ScheduleBlock(rf_B)
            {
              i0, i1, i2 = axis.bind(i, j0, rf_k0)
              rf_B[i2, i0] = (rf_B[i2, i0] + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          B__reduce_init[i0] = 0
        }
        serial for (k0, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0, i2 = axis.bind(i, k0)
            B[i0] = (B[i0] + rf_B[i2, i0])
          }
        }
      }
    }
  }
```

### Parallel Optimization

#### SyncThreads

Add SyncThreads statements in AST. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = (2 * A[i0, i1])
        }
      }
    }
  }
// IR After SyncThreads:
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        __syncthreads()
        serial for (j, 0, 32)
        {
          ScheduleBlock(B)
          {
            i0, i1 = axis.bind(i, j)
            {
              B[i0, i1] = (2 * A[i0, i1])
            }
          }
        }
      }
    }
  }
// Generated code in cuda backends:
  for (int32_t i = 0; i < 32; i += 1) {
    __syncthreads();
    for (int32_t j = 0; j < 32; j += 1) {
      B[((32 * i) + j)] = (2 * A[((64 * i) + j)]);
    };
  };
```

#### Parallel

Parallelize the given loop in X86 backends. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }

// IR After Parallel:
  ScheduleBlock(root)
  {
    parallel for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
// Generated Code After Parallel:
void test_parallel(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  int num_task = max_concurrency();
  omp_set_num_threads(num_task);
  auto flambda = [=](int task_id, int num_task) -> int {
    int n_per_task = (((32 + num_task) - 1) / num_task);
    for (int32_t i = (task_id * n_per_task); i < 32 && i < ((task_id + 1) * n_per_task); i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B[((32 * i) + j)] = A[((32 * i) + j)];
      };
    }
    return 0;
  };
#pragma omp parallel num_threads(num_task)
  {
    int task_id = omp_get_thread_num();
    flambda(task_id, num_task);
  };
  cinn_buffer_free((void*)(0), _B);
}
```

#### Vectorize

Vectorize the given loop. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }

// IR After Vectorize:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      vectorize[16] for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
// Generated Code After Vectorize:
void test_vectorize(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 2; j += 1) {
      B[StackVec<16,int32_t>::Ramp(((32 * i) + (16 * j)), 1, 16)] = StackedVec<float,16>::Load(A,((32 * i) + (16 * j)));
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
```

#### Bind

Bind a loop to the given thread axis. Examples are as follows：

```c
// Original IR:
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
// IR After Bind:
  ScheduleBlock(root)
  {
    thread_bind[threadIdx.x] for (i, 0, 32)
    {
      serial for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
// Generated Code After Bind：
if (((int)threadIdx.x < 32)) {
  for (int32_t j = 0; j < 2; j += 1) {
    B[(int)threadIdx.x * 2 + j] = A[(int)threadIdx.x * 2 + j]
  }
}
```
