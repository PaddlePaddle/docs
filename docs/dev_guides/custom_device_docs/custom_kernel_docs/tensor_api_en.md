# Tensor APIs

There are many kinds of tensors released by PaddlePaddle, and their base class is `TensorBase`, and here will take the commonly-used API `DenseTensor` as an example. For the `TensorBase` and other tensors, please refer to the link at the end of this text.

## DenseTensor

All element data of `DenseTensor` are stored in contiguous memory, and you can refer to [dense_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/dense_tensor.h).

```c++
  // Construct the DenseTensor and allocate memory
  // Parameter：a - pointer type of the Allocator
  //      meta - DenseTensorMeta object
  // Return：None
  DenseTensor(Allocator* a, const DenseTensorMeta& meta);

  // Construct the DenseTensor and allocate memory
  // Parameter：a - pointer type of the Allocator
  //      meta - DenseTensorMeta moving object
  // Return：None
  DenseTensor(Allocator* a, DenseTensorMeta&& meta);

  // Construct the DenseTensor and allocate memory
  // Parameter：holder - shared pointer of Allocation
  //      meta - DenseTensorMeta moving object
  // Return：None
  DenseTensor(const std::shared_ptr<phi::Allocation>& holder,
              const DenseTensorMeta& meta);

  // Move Constructor
  // Parameter：other - DenseTensor moving object
  // Return：None
  DenseTensor(DenseTensor&& other) = default;

  // Copy Constructor
  // Parameter：other - DenseTensor object
  // Return：None
  DenseTensor(const DenseTensor& other);

  // Assignment
  // Parameter：other - DenseTensor object
  // Return：DenseTensor object
  DenseTensor& operator=(const DenseTensor& other);

  // Move Assignment
  // Parameter：other - DenseTensor object
  // Return：DenseTensor object
  DenseTensor& operator=(DenseTensor&& other);

  // No-Parameter Constructor
  DenseTensor();

  // Destructor
  virtual ~DenseTensor() = default;

  // Get the type name，static function
  // Parameter：None
  // Return：string pointer
  static const char* name();

  // Acquire the number of elements of the tensor
  // Parameter：None
  // Return：int64_t categorical variable
  int64_t numel() const override;

  // Acquire the dims of tbe tensor
  // Parameter：None
  // Return：DDim object
  const DDim& dims() const noexcept override;

  // Acquire the lod of the tensor
  // Parameter：None
  // Return：LoD object
  const LoD& lod() const noexcept;

  // Acquire the data type of the Tensor
  // Parameter：None
  // Return: DataType categorical variable
  DataType dtype() const noexcept override;

  // Acquire the memory layout of the tensor
  // Parameter：None
  // Return：DataLayout categorical variable
  DataLayout layout() const noexcept override;

  // Acquire the place of the tensor
  // Parameter：None
  // Return：Place categorical variable
  const Place& place() const override;

  // Acquire the meta of the tensor
  // Parameter：None
  // Return：DenseTensorMeta object
  const DenseTensorMeta& meta() const noexcept;

  // Set the meta of the tensor
  // Parameter：meta - DenseTensorMeta move object
  // Return：None
  void set_meta(DenseTensorMeta&& meta);

  // Set the meta of the tensor
  // Parameter：meta - DenseTensorMeta object
  // Return：None
  void set_meta(const DenseTensorMeta& meta);

  // Check whether the meta of the tensor is valid
  // Parameter：None
  // Return：bool categorical variable
  bool valid() const noexcept override;

  // Check wether the tensor is initialized
  // Parameter：None
  // Return：bool categorical variable
  bool initialized() const override;

  // Allocate memory for the tensor
  // Parameter：allocator - Allocator pointer type
  //      dtype - DataType variable
  //      requested_size - size_t categorical variable
  // Return：void* pointer
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0) override;

  // Check whether memory is shared with other tensors
  // Parameter：b - DenseTensor object
  // Return：bool categorical variable
  bool IsSharedWith(const DenseTensor& b) const;

  // Modify the dims of the tensor and allocate memory
  // Parameter：dims - DDim object
  // Return：None
  void ResizeAndAllocate(const DDim& dims);

  // Modify the dims of the tensor
  // Parameter：dims - DDim object
  // Return：DenseTensor object
  DenseTensor& Resize(const DDim& dims);

  // Reset the LoD of the tensor
  // Parameter：lod - LoD object
  // Return：None
  void ResetLoD(const LoD& lod);

  // Acquire the memory size of the tensor
  // Parameter：None
  // Return：size_t categorical variable
  size_t capacity() const;

  // Acquire the unchangeable data pointer of the tensor
  // Template parameter：T - data type
  // Parameter：None
  // Return：the unchangeable T data pointer
  template <typename T>
  const T* data() const;

  // Acquire the unchangeable data pointer of the tensor
  // Parameter：None
  // Return：the unchangeable void data pointer
  const void* data() const;

  // Acquire the revisable data pointer of the tensor
  // Template parameter：T - data type
  // Parameter：None
  // Return：the revisable T data pointer
  template <typename T>
  T* data();

  // Acquire the revisable data pointer of the tensor
  // Parameter：None
  // Return：the revisable void data pointer
  void* data();
```

## Other Tensors

- `TensorBase`：Please refer to [tensor_base.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/tensor_base.h)
- `SelectedRows`：Please refer to [selected_rows.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/selected_rows.h)
- `SparseCooTensor`：Please refer to [sparse_coo_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_coo_tensor.h)
- `SparseCsrTensor`：Please refer to [sparse_csr_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_csr_tensor.h)


## Relevant Information

- `Allocation` and `Allocator`：Please refer to [allocator.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/allocator.h)
- `DenseTensorMeta`：Please refer to [tensor_meta.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/tensor_meta.h)
- `DDim`：Please refer to [ddim.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/ddim.h)
- `LoD`：Please refer to [lod_utils.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/lod_utils.h)
- `DataType`：Please refer to [data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
- `DataLayout`：Please refer to [layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
- `Place`：Please refer to [place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
