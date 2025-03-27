import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from ddpg_fol.model import (Actor, Critic)
from ddpg_fol.memory import SequentialMemory
from ddpg_fol.util import *
import torch.nn.functional as F

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        
        self.epsilon = 0.6
        self.depsilon = self.epsilon / args.epsilon

        self.is_training = True
        self.flag = 0
        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_state_tensor = to_tensor(next_state_batch)
            next_action_tensor = self.actor_target(next_state_tensor)
            next_q_values = self.critic_target([next_state_tensor, next_action_tensor])

            target_q_batch = to_tensor(reward_batch) + \
                self.discount * to_tensor(terminal_batch) * next_q_values

        # Critic update
        self.critic_optim.zero_grad()

        state_tensor = to_tensor(state_batch)
        action_tensor = to_tensor(action_batch)
        q_batch = self.critic([state_tensor, action_tensor])
    
        value_loss = F.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()

        policy_loss = -self.critic([
            state_tensor,
            self.actor(state_tensor)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self,s_t,a_t, r_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)
            #self.memory.append(s_t1, self.a_t, r_t, done)
            #self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        #self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        with torch.no_grad():
            action = to_numpy(
                self.actor(to_tensor(np.array([s_t])))
            ).squeeze(0)
     
        if self.is_training: 
            action = np.random.normal(action,max(0,self.epsilon)/max(1,2*np.abs(action)))
       
        action = np.clip(action, -1., 1.)

        self.flag+=1
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        return action


    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
