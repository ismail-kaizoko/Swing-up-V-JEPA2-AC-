# Swing-UP : V-JEPA2 + AC 101

> Experimental personal repository to implement the latest advancements in physical action planning from video perception using world-models ( paper : [V-JEPA2](https://arxiv.org/abs/2506.09985) )

### The Vision :

- Physical simulation model of a **Pendulum** swinging in a 2D space, where all physical constraints are included : drag, acceleration, mass, momentum...
- The pendulum (_the agent_) can be given a torque (_the action_) ranging in [$-T_{max}, +T_{max}$] ( _1D continous action space_ )
- The objective : learn a the rigth sequence of torques to communicate to the Pendulum (_the optimal policy_) so it moves from being Vertical ($\theta = 0 $) To be Horizental ($ +-\theta = \pi / 2$).

### The Roadmap :

- [ ] Code _jointly_ the Agent and the Video generator
- [ ] Pretrain V-Jepa on generated videos following the paper
- [ ] Train the Action-Conditioned Predictor following the Jepa-EMbeddings.
- [ ] Test

### Details :

- To learn the actual swinging back and forth (neregy pupmping) to move the ball to reach highest points every period, there are must be a constraint on the maximum posssible delivered torque. $T_{max} << T_h$ where $T_h$ is the torque required to keep the ball still ( _in equilibrium_ ) on $\theta = 0$. which equals by solving the PDE equation of equation gives : $ T_h = m*g*L $

- Equation of movement : $ I * theta_ddot = m*g*l*sin(theta) - b\*theta_dot + T $
