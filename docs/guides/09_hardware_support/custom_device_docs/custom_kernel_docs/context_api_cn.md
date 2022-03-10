# Context API

## CustomContext
`CustomContext`为自定义Kernel函数模板参数Context的实参，具体参照[custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h)

```c++
  // 构造函数
  // 参数：place - CustomPlace对象
  // 返回：None
  explicit CustomContext(const CustomPlace&);

  // 析构函数
  virtual ~CustomContext();

  // 获取设备上下文Place信息
  // 参数：None
  // 返回：place - Place对象
  const Place& GetPlace() const override;

  // 获取设备上下文stream信息
  // 参数：None
  // 返回：stream - void*类型指针
  void* stream() const;

  // 等待stream上的操作完成
  // 参数：None
  // 返回：None
  void Wait() const override;
```

## DeviceContext
`CustomContext`继承自`DeviceContext`，具体参照[device_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/device_context.h)

```c++
  // 无参构造函数
  DeviceContext();

  // 拷贝构造函数
  DeviceContext(const DeviceContext&);

  // 移动构造函数
  DeviceContext(DeviceContext&&);

  // 移动赋值操作
  DeviceContext& operator=(DeviceContext&&);

  // 析构函数
  virtual ~DeviceContext();

  // 设置Device Allocator
  // 参数：Allocator指针
  // 返回：None
  void SetAllocator(const Allocator*);

  // 设置Host Allocator
  // 参数：Allocator指针
  // 返回：None
  void SetHostAllocator(const Allocator*);

  // 设置zero-size Allocator
  // 参数：Allocator指针
  // 返回：None
  void SetZeroAllocator(const Allocator*);

  // 获取 Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetAllocator() const;

  // 获取 Host Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetHostAllocator() const;

  // 获取 zero-size Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetZeroAllocator() const;

  // 为Tensor分配Device内存
  // 参数：TensorBase类型指针
  //      dtype - DataType类型变量
  //      requested_size - size_t类型变量，默认值为0
  // 返回：数据指针 - void*类型指针
  void* Alloc(TensorBase*, DataType dtype, size_t requested_size = 0) const;

  // 为Tensor分配Device内存
  // 模板参数：T - 数据类型
  // 参数：TensorBase类型指针
  //      requested_size - size_t类型变量，默认值为0
  // 返回：数据指针 - T*类型指针
  template <typename T>
  T* Alloc(TensorBase* tensor, size_t requested_size = 0) const;

  // 为Tensor分配Host内存
  // 参数：TensorBase指针
  //      dtype - DataType类型变量
  //      requested_size - size_t类型变量，默认值为0
  // 返回：数据指针 - void*类型指针
  void* HostAlloc(TensorBase* tensor,
                  DataType dtype,
                  size_t requested_size = 0) const;

  // 为Tensor分配Host内存
  // 模板参数：T - 数据类型
  // 参数：TensorBase指针
  //      requested_size - size_t类型变量，默认值为0
  // 返回：数据指针 - T*类型数据指针
  template <typename T>
  T* HostAlloc(TensorBase* tensor, size_t requested_size = 0) const;

  // 获取设备上下文Place信息，子类实现
  // 参数：None
  // 返回：place - Place对象
  virtual const Place& GetPlace() const = 0;

  // 等待stream上的操作完成，子类实现
  // 参数：None
  // 返回：None
  virtual void Wait() const {}

  // 设置随机数发生器
  // 参数：Generator指针
  // 返回：None
  void SetGenerator(Generator*);

  // 获取随机数发生器
  // 参数：None
  // 返回：Generator指针
  Generator* GetGenerator() const;

  // 设置Host随机数发生器
  // 参数：Generator指针
  // 返回：None
  void SetHostGenerator(Generator*);

  // 获取Host随机数发生器
  // 参数：None
  // 返回：Generator指针
  Generator* GetHostGenerator() const;

```

## 相关内容

- `Place`与`CustomPlace`：请参照[place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
- `Allocation`与`Allocator`：请参照[allocator.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/allocator.h)
- `TensorBase`：请参照[tensor_base.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/tensor_base.h)
- `DataType`：请参照[data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
- `Generator`：请参照[generator.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/generator.h)
