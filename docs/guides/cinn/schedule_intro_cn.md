# IR Schedule 与调度原语介绍

## IR Schedule

在 CINN 的框架中，我们通过计算模块来定义算子的行为，通过调度模块来优化算子计算的过程（即生成的代码）。而 IR 就是对算子生成的代码表示的抽象（本质为一个抽象语法树），IR Schedule 的作用是修改优化 IR 来生成更高性能的代码。调度原语就是实现这个修改功能的基础 api 模块，开发者能够通过使用不同的调度原语来完成对计算的优化。

## 调度原语

目前 CINN 的调度原语大致分为三类：

- 循环变换类：Fuse, Split, Unroll ...

- 存储层次类：CacheRead, CacheWrite ...

- 并行优化类：Vectorize, Bind, Parallel ...


源代码位置在 `cinn/ir/ir_schedule.h`

### 循环变换类

#### Fuse

将多个循环融合成一个，举例如下：

```c
// 原 ir：
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

// Fuse 后：
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

将一个循环拆分成多个，举例如下：

```c
// 原 ir：
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

// 将`j`这层 loop 以{4, 16}作为 factor 进行 Split 后：
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

调整循环的顺序，举例如下：

```c
// 原 ir：
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

// Reorder 后：
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

将一个循环展开，将其每个语句都生成，举例如下：

```c
// 原生成代码：
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; i < 2; j += 1) {
      B[(2 * i) + j] = A[(2 * i) + j];
    }
  };


// Unroll 后：
  for (int32_t i = 0; i < 32; i += 1) {
    B[(2 * i)] = A[(2 * i)];
    B[(1 + (2 * i))] = A[(1 + (2 * i))];
  };
```

#### ComputeInline

将一个 Tensor 的计算内联化，举例如下：

```c
// 原 ir：
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
// ComputeInline 之后：
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

移动一个 Tensor 计算的位置（移动的 Tensor 会被后续 Tensor 使用）；ComputeAt 后，被移动的 tensor 会继承相关 loop 的变化，举例如下：

```c
// 原 ir：
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
// Fuse C 的前两层 loop 后的 IR:
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

// 将 B ComputeAt 到 C 的最后一层 loop。由于 ComputeAt 后，B 会继承 C 中相关 loop 的变化，因此 B 也会自动做一次 Fuse，生成`i_j_fused`这层 loop，结果如下：
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

移动一个 Tensor 计算的位置。
SimpleComputeAt 与 ComputeAt 的主要区别在于：
- ComputeAt 需要被移动的 Tensor 后续被用到（被依赖），比如`B = A + 1`, `C = B + 1`，那么 B 移动到 C 的某层 loop 下就是满足要求的（B 被 C 用到），而 SimpleComputeAt 没有这个要求。
- SimpleComputeAt 移动到第 n 层 loop 时，需要前 n 层 loop 的 range 相同。比如 B 移动到 C 的第 3 层 loop 内，那么 B 和 C 的前 3 层 loop range 必须相同，而 ComputeAt 没有这个要求（因为他会自动继承 loop 做的变换）
举例如下：

```c
// 原 ir:
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
// SimpleComputeAt 之后：
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
在这个例子中，我们不能使用 ComputeAt，因为`C = 2 * A` 并没有依赖（用到）`B = 1 + A`；
同样，在 ComputeAt 的例子中，我们不能使用 SimpleComputeAt，因为`C = 2 * A` 做了 Fuse 后第一层 Loop 的 range 与`B = 1 + A`不相同。需要手动对`B = 1 + A`也做同样的 Fuse 后才能使用 SimpleComputeAt。

### 存储层次类

#### CacheRead

对于一个输入 Tensor，创建一个临时 cache tensor 将其被读的部分数据存入，代替原 tensor 进行读操作，举例如下：

```c
// 原 ir:
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

// 通过 CacheRead 建立 shared memory 的 cache 之后：
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
可以看到在对 A 做 CacheRead，将 A 的结果缓存到 shared memory 后，原本对 global memory 的 32 * 32 * 16 次读操作变成了对 global memory 的 32 * 32 次读操作加上对 shared memory 的 32 * 32 * 16 次读操作。由于 shared memory 的读速度远快于 global memory，做 CacheRead 后的 IR 运行速度会更快。

#### CacheWrite

对于一个输出 Tensor，创建一个临时 cache tensor 将其被写的部分数据存入，代替原 tensor 进行写操作，举例如下：

```c
// 原 ir:
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
// 通过 CacheWrite 建立 shared memory 的 cache 之后：
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
可以看到在对 B 做 CacheWrite，将 B 的结果先缓存到 shared memory 后，原本对 global memory 的 64 * 32 * 32 次写操作变成了对 global memory 的 64 * 32 次写操作加上对 shared memory 的 64 * 32 * 32 次写操作。由于 shared memory 的写速度远快于 global memory，做 CacheWrite 后的 IR 运行速度会更快。

#### SetBuffer

设置一个 Tensor 的内存属性，目前可选的属性有`local`, `shared`, `global`。举例如下：

```c
// 在 cuda 后端中，将 tensor a 的内存变为 local memory
ir_sch.SetBuffer(a, "local");
```

#### Rfactor

将一个 Reduce 类的计算拆成两次 Reduce 计算，举例如下：

```c
// 原 ir:
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
// Rfactor 之后：
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

### 并行优化类

#### SyncThreads

在代码中添加 syncthreads()指令，只在 cuda 后端有效，举例如下：

```c
// 原 ir:
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
// SyncThreads 之后的 ir：
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
// 生成的 cuda 端代码：
  for (int32_t i = 0; i < 32; i += 1) {
    __syncthreads();
    for (int32_t j = 0; j < 32; j += 1) {
      B[((32 * i) + j)] = (2 * A[((64 * i) + j)]);
    };
  };
```

#### Parallel

在 X86 端，对一个 loop 进行并行化标记，让其多线程执行。举例如下：

```c
// 原 ir：
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
// Parallel 后的 ir：
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
// 最终生成的代码：
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

对一个 loop 进行向量化计算标记，让其通过向量化指令执行计算。举例如下：

```c
// 原 ir：
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
// Vectorize 后生成的 ir：
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
// 最终生成的代码：
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

对一个 loop 进行绑定标记，让其分配到某个线程中，举例如下：

```c
// 原 ir：
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
// Bind 到 threadIdx.x 之后，在 cuda 后端生成的 ir：
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
// 最终生成代码如下：
if (((int)threadIdx.x < 32)) {
  for (int32_t j = 0; j < 2; j += 1) {
    B[(int)threadIdx.x * 2 + j] = A[(int)threadIdx.x * 2 + j]
  }
}
```
