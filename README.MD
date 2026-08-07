# Swing-UP : V-JEPA2 + AC 101

> Experimental personal repository to implement the latest advancements in physical action planning from video perception using world-models ( paper : [V-JEPA2](https://arxiv.org/abs/2506.09985) )

### The Vision :

- Physical simulation model of a **Pendulum** swinging in a 2D space, where all physical constraints are included : drag, acceleration, mass, momentum...
- The pendulum (_the agent_) can be given a torque (_the action_) ranging in [$-T_{max}, +T_{max}$] ( _1D continous action space_ )
- The objective : learn a the rigth sequence of torques to communicate to the Pendulum (_the optimal policy_) so it moves from being Vertical ( $ \theta = \pi $) To be Horizental ( $ \theta = 0 $).

### The Roadmap :

- [ ] Code _jointly_ the Agent and the Video generator
- [ ] Pretrain V-Jepa on generated videos, no action conditioning in this step.
- [ ] Train the Action-Conditioned Predictor following the Jepa-EMbeddings.
- [ ] CEM planing

### Details :

- Although the problem looks simple (1D acton space, 2D pendulum), the best policy to learn isn't obvious, the world-models predict actions that will make the embeddings of frame:t looks closer to frame:t+1, learning that "we must get as far from target so we build energy to get closer in next period" seems complex for the model to catch it, this is what this repository for. (I have no prior idea if this will work !!)

- To learn the actual swinging back and forth (neregy pupmping) to move the ball to reach highest points every period, there are must be a constraint on the maximum posssible delivered torque. $T_{max} << T_h$ where $T_h$ is the torque required to keep the ball still ( _in equilibrium_ ) on $\theta = 0$. which equals by solving the PDE equation of equation gives : $ T_h = m*g*L $

- Equation of movement : $ I.\ddot{\theta} = m.g.l.sin(\theta) - b.\dot{\theta} + T $
