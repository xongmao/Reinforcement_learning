import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=400., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 50.]) 
    
    def get_reward(self, i_episode=0, voc=0., c=0., old_goal=np.array([]), old_vec=np.array([])):
        """Uses current pose of sim to return reward."""
        reward = -abs(self.target_pos[2]-self.sim.pose[2])
        if self.sim.pose[2]>c:
            reward += 100.0
        
        reward -= 0.01*(self.sim.pose[0]**2 + self.sim.pose[1]**2)
        
        if self.sim.v[2]>voc:
            reward += 50.0
                
        reward -= 0.01*(self.sim.v[0]**2 + self.sim.v[1]**2)
        
        reward -= 0.2*(abs(self.sim.angular_v[:3])).sum()
        reward -= 0.2*(abs(self.sim.pose[3:6])).sum()
        
        return reward

    def step(self, rotor_speeds, i_episode=0, voc=0., c=0., old_goal=np.array([]), old_vec=np.array([])):
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(i_episode, voc, c, old_goal, old_vec) 
            a = abs(self.sim.pose[0]-self.target_pos[0])
            b = abs(self.sim.pose[1]-self.target_pos[1])
            c = abs(self.sim.pose[2]-self.target_pos[2])
            if done == True:
                print("lets go")
                reward -= 100.0
            else:
                if c<5. and b<5. and a<5.:
                    print("goal")
                    reward += 100.0
                    done = True
            
            pose_all1 = []
            pose_all1.append(self.sim.pose[0])
            pose_all1.append(self.sim.pose[1])
            pose_all1.append(self.sim.pose[2])
            pose_all1.append(self.sim.pose[3])
            pose_all1.append(self.sim.pose[4])
            pose_all1.append(self.sim.pose[5])
            pose_all1.append(self.sim.v[0])
            pose_all1.append(self.sim.v[1])
            pose_all1.append(self.sim.v[2])
            pose_all1.append(self.sim.angular_v[0])
            pose_all1.append(self.sim.angular_v[1])
            pose_all1.append(self.sim.angular_v[2])
            pose_all.append(pose_all1)
        next_state = np.concatenate(pose_all)
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        v_l = []
        v_l.append(self.sim.pose[0])
        v_l.append(self.sim.pose[1])
        v_l.append(self.sim.pose[2])
        v_l.append(self.sim.pose[3])
        v_l.append(self.sim.pose[4])
        v_l.append(self.sim.pose[5])
        v_l.append(self.sim.v[0])
        v_l.append(self.sim.v[1])
        v_l.append(self.sim.v[2])
        v_l.append(self.sim.angular_v[0])
        v_l.append(self.sim.angular_v[1])
        v_l.append(self.sim.angular_v[2])
        state = np.concatenate([v_l] * self.action_repeat) 
        return state