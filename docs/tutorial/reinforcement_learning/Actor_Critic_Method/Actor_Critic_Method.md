# **强化学习——Actor Critic Method**
**作者：** [EastSmith](https://github.com/EastSmith)<br>
**日期：** 2021.05 <br>
**摘要：** 展示 `CartPole-V0` 环境中 `Actor-Critic` 方法的一个实现。

## **一、介绍**
本案例展示了CartPole-V0环境中Actor-Critic方法的一个实现。

### Actor Critic Method（演员--评论家算法）
当代理在环境中执行操作和移动时，它将观察到的环境状态映射到两个可能的输出：
* 推荐动作：动作空间中每个动作的概率值。代理中负责此输出的部分称为actor（演员）。
* 未来预期回报：它预期在未来获得的所有回报的总和。负责此输出的代理部分是critic（评论家）。

演员和评论家学习执行他们的任务，这样演员推荐的动作就能获得最大的回报。

### CartPole-V0
在无摩擦的轨道上，一根杆子系在一辆手推车上。agent（代理）必须施加力才能移动手推车。每走一步，杆子就保持直立，这是奖励。因此，agent（代理）必须学会防止杆子掉下来。

## **二、环境配置**
本教程基于Paddle 2.1 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.1 。


```python
import gym, os
from itertools import count
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.distribution import Categorical

print(paddle.__version__)
```

    2.1.0


## **三、实施演员-评论家网络**
### 这个网络学习两个功能：
* 演员Actor：它将环境的状态作为输入，并为其动作空间中的每个动作返回一个概率值。
* 评论家Critic：它将的环境状态作为输入，并返回对未来总回报的估计。


```python
device = paddle.get_device()
env = gym.make("CartPole-v0")  ### 或者 env = gym.make("CartPole-v0").unwrapped 开启无锁定环境训练

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.001

class Actor(nn.Layer):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, axis=-1))
        return distribution


class Critic(nn.Layer):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

```

## **四、训练模型**


```python
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(lr, parameters=actor.parameters())
    optimizerC = optim.Adam(lr, parameters=critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            # env.render()
            state = paddle.to_tensor(state,dtype="float32",place=device)
            dist, value = actor(state), critic(state)

            action = dist.sample([1])
            next_state, reward, done, _ = env.step(action.cpu().squeeze(0).numpy()) 

            log_prob = dist.log_prob(action);
            # entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(paddle.to_tensor([reward], dtype="float32", place=device))
            masks.append(paddle.to_tensor([1-done], dtype="float32", place=device))

            state = next_state

            if done:
                if iter % 10 == 0:
                    print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = paddle.to_tensor(next_state, dtype="float32", place=device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = paddle.concat(log_probs)
        returns = paddle.concat(returns).detach()
        values = paddle.concat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.clear_grad()
        optimizerC.clear_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    paddle.save(actor.state_dict(), 'model/actor.pdparams')
    paddle.save(critic.state_dict(), 'model/critic.pdparams')
    env.close()



if __name__ == '__main__':
    if os.path.exists('model/actor.pdparams'):
        actor = Actor(state_size, action_size)
        model_state_dict  = paddle.load('model/actor.pdparams')
        actor.set_state_dict(model_state_dict )
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size)
    if os.path.exists('model/critic.pdparams'):
        critic = Critic(state_size, action_size)
        model_state_dict  = paddle.load('model/critic.pdparams')
        critic.set_state_dict(model_state_dict )
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size)
    trainIters(actor, critic, n_iters=201)
```

    Iteration: 0, Score: 32
    Iteration: 10, Score: 43
    Iteration: 20, Score: 11
    Iteration: 30, Score: 18
    Iteration: 40, Score: 39
    Iteration: 50, Score: 18
    Iteration: 60, Score: 104
    Iteration: 70, Score: 82
    Iteration: 80, Score: 199
    Iteration: 90, Score: 199
    Iteration: 100, Score: 199



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-3-c23b84fbafb3> in <module>
         84     else:
         85         critic = Critic(state_size, action_size)
    ---> 86     trainIters(actor, critic, n_iters=201)
    

    <ipython-input-3-c23b84fbafb3> in trainIters(actor, critic, n_iters)
         46         next_state = paddle.to_tensor(next_state, dtype="float32", place=device)
         47         next_value = critic(next_state)
    ---> 48         returns = compute_returns(next_value, rewards, masks)
         49 
         50         log_probs = paddle.concat(log_probs)


    <ipython-input-3-c23b84fbafb3> in compute_returns(next_value, rewards, masks, gamma)
          3     returns = []
          4     for step in reversed(range(len(rewards))):
    ----> 5         R = rewards[step] + gamma * R * masks[step]
          6         returns.insert(0, R)
          7     return returns


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py in __impl__(self, other_var)
        248             axis = -1
        249             math_op = getattr(core.ops, op_type)
    --> 250             return math_op(self, other_var, 'axis', axis)
        251 
        252         comment = OpProtoHolder.instance().get_op_proto(op_type).comment


    KeyboardInterrupt: 


## **五、效果展示**
在训练的早期：

![](https://ai-studio-static-online.cdn.bcebos.com/d8826cc5bb8a4106bdd871a7f35c449d90029a3ae3f6465aa373c614baa78a9f)

在训练的后期
![](https://ai-studio-static-online.cdn.bcebos.com/88b967da1ba74e049b3ff28dd9083d1e527ba734dc064a798374f99199f84086)


## **六、总结**

* Actor-Critic，其实是用了两个网络： 一个输出策略，负责选择动作，这个网络称为Actor；一个负责计算每个动作的分数，这个网络称为Critic。
* 可以形象地想象为，Actor是舞台上的舞者，Critic是台下的评委，Actor在台上跳舞，一开始舞姿并不好看，Critic根据Actor的舞姿打分。Actor通过Critic给出的分数，去学习：如果Critic给的分数高，那么Actor会调整这个动作的输出概率；相反，如果Critic给的分数低，那么就减少这个动作输出的概率。
* Actor-Critic方法结合了值函数逼近（Critic）和策略函数逼近（Actor），它从与环境的交互中学习到越来越精确的Critic（评估），能够实现单步更新，相对单纯的策略梯度，Actor-Critic能够更充分的利用数据。
