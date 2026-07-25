# Swing-UP : V-JEPA2 + AC 101

> Experimental personal repository to implement the latest advancements in physical action planning from video perception using world-models ( paper : [V-JEPA2](https://arxiv.org/abs/2506.09985)  )


### The Vision : 
- Physical simulation model of a **Pendulum** swinging in a 2D space, where all physical constraints are included : drag, acceleration, mass, momentum...
- The pendulum (*the agent*) can be given a torque (*the action*) ranging in [-F.d, +F.d] ( *continous action space* )
- The objective : learn a the rigth sequence of torques to communicate to the Pendulum (*the optimal policy*) so it moves from being **still** (Vertical) To be Horizental.


### The Roadmap : 

[ ] Code *jointly* the Agent and the Video generator
[ ] Pretrain V-Jepa on generated videos following the paper
[ ] Train the Action-Conditioned Predictor following the Jepa-EMbeddings.
[ ] Test