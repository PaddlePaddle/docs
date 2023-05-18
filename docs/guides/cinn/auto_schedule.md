# CINN 自动调优框架
  CINN 在算子(融合算子)实现层采用的是 compute 与 schedule 分离的思想，compute 表示算子的朴素实现，schedule 表示具体的计算方式。auto-schedule 的作用是自动生成算子的 schedule 配置，降低新硬件接入编译器的人力成本和技术门槛，并且满足极致追求性能场景的优化需求。
