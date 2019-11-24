# -*- coding: utf-8 -*-
import numpy as np
import random
from actor_net import ActorNet
from critic_net import CriticNet
from collections import deque
from tensorflow_grad_inverter import grad_inverter

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 10
GAMMA = 0.99
is_grad_inverter = True


class DDPG:
    """
        Deep Deterministic Policy Gradient Algorithm.
        Sourced By: https://github.com/stevenpjg/ddpg-aigym/blob/master/ddpg.py
    """

    def __init__(self, num_states, num_actions, action_space_high, action_space_low, is_batch_norm):

        self.num_states = num_states
        self.num_actions = num_actions
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low

        # Batch normalisation disabled.
        self.critic_net = CriticNet(self.num_states, self.num_actions)
        self.actor_net = ActorNet(self.num_states, self.num_actions)

        # Replay Memory 초기화
        self.replay_memory = deque()

        # time 초기화
        self.time_step = 0
        self.counter = 0

        action_max = np.array(action_space_high).tolist()
        action_min = np.array(action_space_low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)

    # observation_1 = state at time t
    # observation 2 = state at time (t + 1)
    def add_experience(selfk, observation_1, observation_2, action, reward, done):
        self.observation_1 = observation_1
        self.observation_2 = observation_2
        self.action = action
        self.reward = reward
        self.done = done

        self.replay_memory.append(
            (self.observation_1, self.observation_2, self.action, self.reward, self.done))
        self.time_step = self.time_step + 1

        # Replay memory 가 가득차면 맨 첫 번째 memory 를 삭제한다
        if (len(self.replay_memory) > REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()

    def minibatches(self):
        # BATCH_SIZE 만큼 replay memory에서 가져온다.
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        # S(t) 와 S(T + 1), action, reward, done 에 대한 batch를
        # 각각 따로 저장한다
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array(self.state_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(
            self.action_batch, [len(self.action_batch), self.num_actions])
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)

    def train(self):
        print "######## Starting to train..."
        # batch 뽑기
        self.minibatches()
        # S(t + 1) 정보를 가지고 time (t + 1)에서의 action batch 생성
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(
            self.state_t_1_batch)
        # Q`(S(t + 1), a(t + 1))
        q_t_1 = self.critic_net.evaluate_target_critic(
            self.state_t_1_batch, self.action_t_1_batch)
        print "#### Evaluated ciritic value(Q value)"
        print q_t_1
        self.y_i_batch = []     # reward batch 의 item 을 가공하여 저장하는 곳

        for i in range(0, BATCH_SIZE):

            # done == True 이면 terminal state로 간 것이므로
            # 이 때의 reward 를 정답상태로 갔을 때의 reward 라고 할 수 있다
            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            # False 이면 terminal state 는 아니므로 reward에 (감마 * Q value) 값을 더한다
            else:
                self.y_i_batch.append(
                    self.reward_batch[i] + GAMMA * q_t_1[i][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])

        # loss 를 최소화하여 critic network 를 업데이트 한다
        # weight 을 업데이트 하는데 (y_i_batch - (state_t_batch, action_batch) 에서 예측한 y value) 가 최소가 되도록 한다
        self.critic_net.train_critic(
            self.state_t_batch, self.action_batch, self.y_i_batch)

        # gradient 에 따라 actor 를 업데이트 한다
        action_for_delQ = self.evaluate_actor(self.state_t_batch)

        if is_grad_inverter:
            self.del_Q_a = self.critic_net.compute_delQ_a(
                self.state_t_batch, action_for_delQ)
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a, action_for_delQ)
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(
                self.state_t_batch, action_for_delQ)[0]

        # actor network 학습
        self.actor_net.train_actor(self.state_t_batch, self.del_Q_a)

        # target critic, target actor network 업데이트
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        self.critic_net.save_critic("model/critic_model.ckpt")
        self.actor_net.save_actor("model/actor_model.ckpt")
        print "######## Finish to train ..."
