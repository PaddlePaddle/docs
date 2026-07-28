random
------------------

  py:class::paddle.distributed.fleet.meta_parallel.parallel_layers.random

 random主要是根据模型并行方式产生对应的随机数，从而在dropout策略中针对不同的并行方式有不同的随机数进行概率对，以此来决定某个梯度值的去留。


RNGStatesTracker类：
-----------------

方法
.........
.........
add（name, seed）
'''''''''

将模型并行的方式与随机数状态种子相匹配(映射)

**参数**
     - **name** (srting) 模型并行的方式
     - **seed** (int) 随机数
**返回**
dict

**代码示例**
..code-block::python
             def add(self, name, seed):
                 if seed in self.seeds_:
                     raise ValueError('seed {} already exists'.format(seed))
                 self.seeds_.add(seed)
                 if name in self.states_:
                     raise ValueError('state {} already exists'.format(name))
                 orig_rng_state = paddle.get_cuda_rng_state()
                 paddle.seed(seed)
                 self.states_[name] = paddle.get_cuda_rng_state()
                 paddle.set_cuda_rng_state(orig_rng_state)


rng_state(name)
'''''''''''

设置随机数的状态信息，并将其加以保存

**参数**
   - **name** 模型状态信息
**返回**
None

**代码示例**
..code-block::python
     def rng_state(self, name=MODEL_PARALLEL_RNG):
         if name not in self.states_:
            raise ValueError('state {} does not exist'.format(name))
         orig_cuda_rng_state = paddle.get_cuda_rng_state()
         paddle.set_cuda_rng_state(self.states_[name])
         try:
             yield
         finally:
             self.states_[name] = paddle.get_cuda_rng_state()
             paddle.set_cuda_rng_state(orig_cuda_rng_state)



model_parallel_random_seed(seed)
'''''''''''

设置模型并行的随机种子，并且将模型并行的方式与随机数种子相匹配(映射)

**参数**
   - **seed** (int) 随机数种子，默认为None
**返回**
None


**代码示例**
..code-block::python
          def model_parallel_random_seed(seed=None):
              import paddle.distributed.fleet as fleet
              hcg = fleet.get_hybrid_communicate_group()
              rank = hcg.get_model_parallel_rank()
              if seed:
                  global_seed = seed
                  local_seed = seed * 1024 + rank * 100
              else:
                  global_seed = np.random.randint(0, 655350)
                  local_seed = np.random.randint(rank * 10000, (rank + 1) * 10000 - 1)

              RNG_STATE_TRACKER.reset()
              RNG_STATE_TRACKER.add(MODEL_PARALLEL_RNG, local_seed)
              paddle.seed(global_seed)



determinate_seed(rng_name)
''''''''''''

根据模型并行的方式获取最终的用于dropout的最终随机数种子

**参数**
    - **rng_name** 模型并行的名称
**返回**
Int

**代码示例**
..code-block::python
     def determinate_seed(rng_name):
         assert rng_name is not None and rng_name != ""
         helper = LayerHelper('seed', ** locals())
         out = helper.create_variable_for_type_inference(dtype=paddle.int32)
         helper.append_op(type='seed',
                     outputs={'Out': out},
                     attrs={
                         'deterministic': True,
                         'rng_name': rng_name,
                         'force_cpu': True
                     })
         return out


dropout(x, p, axis, rng_name, training, mode, name)
''''''''''''

根据模型并行方式产生不同的随机数种子，并且将这些随机数种子适用于dropout方法时，与设定的概率比较大小，决定提督元素的保留和舍入。

**参数**
     - **x** (Tensor) 输入的矩阵张量
     - **p** (float) 张量元素置零的概率，即丢弃的概率
     - **axis** (int｜list| tuple) dropout操作沿着某一坐标轴运行
     - **rng_name** （string) 随机数种子生成器的名称，被用来获取随机数种子
     - **training** (bool) 标识当前流程是否在训练中
     - **mode** (string) 针对保留下来的梯度元素进行数值上的放缩
     - **name** (string) 操作的名称
**返回**
处理后的张量矩阵

**代码示例**
..code-block::python
def dropout(x, p=0.5, axis=None，rng_name=None, training=True, mode="upscale_in_train", name=None):
         if rng_name is None:
             return paddle.nn.functional.dropout(x, p, axis, training, mode, name)

         if not isinstance(p, (float, int, Variable)):
             raise TypeError("p argument should be a number(int|float) or Variable")

         if isinstance(p, (int, float)) and p == 0: return x

         assert 0 <= p <= 1, ValueError("p argument should between 0 and 1")
         assert mode in ('downscale_in_infer', 'upscale_in_train'), \
             ValueError("mode argument should be 'downscale_in_infer' or 'upscale_in_train'")

         assert axis is None, \
             TypeError("unsupport axis when using random seed generator")

         mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

         if _non_static_mode():
             out, mask = _legacy_C_ops.dropout(x, 'dropout_prob', p, 'is_test',
                                          not training, 'fix_seed', False,
                                          'seed', 0, 'dropout_implementation',
                                          mode)
             return out

         seed = determinate_seed(rng_name)

         if isinstance(p, Variable) and not p.shape != [1]:
             raise TypeError( "Required p.shape == [1] if type(p) is Variable, but received p.shape = {}"
            .format(p.shape))

         helper = LayerHelper('dropout', ** locals())
         check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'dropout')

         out = helper.create_variable_for_type_inference(dtype=x.dtype)
         mask = helper.create_variable_for_type_inference(dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

         helper.append_op(type='dropout',
                     inputs={
                         'X': [x],
                         'Seed': seed
                     },
                     outputs={
                         'Out': [out],
                         'Mask': [mask]
                     },
                     attrs={
                         'dropout_prob': p,
                         'is_test': not training,
                         'dropout_implementation': mode,
                     })
         return out

