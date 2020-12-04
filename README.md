# actor_critic_agents
my re-implementation of actor critic agents. Part of another "learning progress repo" I had but it started to become too messy... For now the focus is on Continuous action spaces, and actor_critic agents.


##Requires 
* PyTorch (>1.7.0 for additional distributions)
* openAI gym (including mujoco)

### Mujoco-py needs some env variable fixing
On Ubuntu 18.04 (my machine) the following additional envirnoment variable have to be set.
(in.bashrc or manually in terminal)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco200/bin


export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.0
