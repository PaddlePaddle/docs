.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None)

均方根平均传播（RMSProp）法是一种未发表的,自适应学习率的方法。原始slides提出了RMSProp：[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]中的第29张slide。等式如下所示：

.. math::
    \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\w &amp; = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)\end{aligned}\end{align}
    
  第一个等式计算每个权重平方梯度的移动平均值，然后将梯度除以 :math:`sqrtv（w，t）` 。
  
.. math::
   \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\v(w, t) &amp; = \beta v(w, t-1) +\frac{\eta} {\sqrt{r(w,t) +\epsilon}} \nabla Q_{i}(w)\\w &amp; = w - v(w, t)\end{aligned}\end{align}

  如果居中为真：
  
.. math::
      \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\g(w, t) &amp; = \rho g(w, t-1) + (1 -\rho)\nabla Q_{i}(w)\\v(w, t) &amp; = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +\epsilon}} \nabla Q_{i}(w)\\w &amp; = w - v(w, t)\end{aligned}\end{align}
      
      其中， :math:`ρ` 是超参数，典型值为0.9,0.95等。 :math:`beta` 是动量术语。  :math:`epsilon` 是一个平滑项，用于避免除零，通常设置在1e-4到1e-8的范围内。
      
参数：
    - **learning_rate** （float）：全球学习率。
    - **rho** （float）：rho是：math：rho in equation，默认设置为0.95。
    - **epsilon** （float）：等式中的epsilon是平滑项，避免被零除，默认设置为1e-6。
    - **momentum** （float）：方程中的β是动量项，默认设置为0.0。
    - **centered** （bool）： 如果为True，则通过梯度估计方差对梯度进行归一化；如果false，则由未centered的第二个moment归一化。将此设置为True有助于培训，但在计算和内存方面稍微昂贵一些。默认为False。
    - **regularization** ：正规化项，如fluid.regularizer.L2DecayRegularizer。
    - **name** ：可选的名称前缀。
    
抛出：   ValueError：如果 ``learning_rate`` ， ``rho`` ， ``epsilon`` ， ``momentum`` 为None。

**示例代码**

..  code-block:: python

        optimizer = fluid.optimizer.RMSProp(0.0001)
        _, params_grads = optimizer.minimize(cost)
        
    
    
.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None)

均方根平均传播（RMSProp）法是一种未发表的,自适应学习率的方法。原始slides提出了RMSProp：[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]中的第29张slide。等式如下所示：

.. math::
    \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\w &amp; = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)\end{aligned}\end{align}
    
  第一个等式计算每个权重平方梯度的移动平均值，然后将梯度除以 :math:`sqrtv（w，t）` 。
  
.. math::
   \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\v(w, t) &amp; = \beta v(w, t-1) +\frac{\eta} {\sqrt{r(w,t) +\epsilon}} \nabla Q_{i}(w)\\w &amp; = w - v(w, t)\end{aligned}\end{align}

  如果居中为真：
  
.. math::
      \begin{align}\begin{aligned}r(w, t) &amp; = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\g(w, t) &amp; = \rho g(w, t-1) + (1 -\rho)\nabla Q_{i}(w)\\v(w, t) &amp; = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +\epsilon}} \nabla Q_{i}(w)\\w &amp; = w - v(w, t)\end{aligned}\end{align}
      
      其中， :math:`ρ` 是超参数，典型值为0.9,0.95等。 :math:`beta` 是动量术语。  :math:`epsilon` 是一个平滑项，用于避免除零，通常设置在1e-4到1e-8的范围内。
      
参数：
    - **learning_rate** （float）：全球学习率。
    - **rho** （float）：rho是：math：rho in equation，默认设置为0.95。
    - **epsilon** （float）：等式中的epsilon是平滑项，避免被零除，默认设置为1e-6。
    - **momentum** （float）：方程中的β是动量项，默认设置为0.0。
    - **centered** （bool）： 如果为True，则通过梯度估计方差对梯度进行归一化；如果false，则由未centered的第二个moment归一化。将此设置为True有助于培训，但在计算和内存方面稍微昂贵一些。默认为False。
    - **regularization** ：正规化项，如fluid.regularizer.L2DecayRegularizer。
    - **name** ：可选的名称前缀。
    
抛出：   ValueError：如果 ``learning_rate`` ， ``rho`` ， ``epsilon`` ， ``momentum`` 为None。

**示例代码**

..  code-block:: python

        optimizer = fluid.optimizer.RMSProp(0.0001)
        _, params_grads = optimizer.minimize(cost)
        
        
        
        

