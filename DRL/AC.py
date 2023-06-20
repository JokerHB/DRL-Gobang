# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, input_data):
        output = F.relu(self.linear1(input_data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output


class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, input_data):
        output = F.relu(self.linear1(input_data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output.mean()


class Player(nn.Module):

    def __init__(self, input_size, hidden_size, ouput_size):
        super(Player, self).__init__()

        self.actor = Actor(input_size=input_size, hidden_size=hidden_size, output_size=ouput_size)
        self.critic = Critic(input_size=input_size, hidden_size=hidden_size, output_size=1)

    def forward(self, input_data):
        probs = self.actor(input_data)
        score = self.critic(input_data)

        return probs, score


class AC(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, gamma=0.9, lr=1e-4, episodes_num=100000, target_update=9, model_path='model/'):
        super(AC, self).__init__()

        self.gamma = gamma
        self.lr = lr
        self.model_path = model_path
        self.episodes_num = episodes_num
        self.target_update = target_update

        self.player1 = Player(input_size=input_size, hidden_size=hidden_size, ouput_size=output_size).to(device)
        self.player2 = Player(input_size=input_size, hidden_size=hidden_size, ouput_size=output_size).to(device)

        self.reload_model(model_path=self.model_path)
        self.player1.train()
        self.player2.train()

        self.player1_opt = optim.Adam(self.player1.parameters(), lr)
        self.player2_opt = optim.Adam(self.player2.parameters(), lr)

    def save_model(self, model_path):
        torch.save(self.player1.state_dict(), model_path + 'player1.pkl')
        torch.save(self.player2.state_dict(), model_path + 'player2.pkl')

    def reload_model(self, model_path):
        if os.path.exists(model_path + 'player1.pkl'):
            self.player1.load_state_dict(torch.load(model_path + 'player1.pkl', map_location=device))
        if os.path.exists(model_path + 'player2.pkl'):
            self.player2.load_state_dict(torch.load(model_path + 'player2.pkl', map_location=device))
            print('load previous model...')

    def compute_returns(self, next_value, rewards):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                R = rewards[step]
            else:
                R = rewards[step] + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_mask(self, state, mask):
        state = state.view(1, -1)
        index = torch.ne(state, 0.0)
        mask = mask.view(1, -1)
        mask[index] = 0.0
        return mask

    def train_model(self, env, run_log='run_log.log', is_first=None):
        for i in range(self.episodes_num):
            p1_log_p, p1_reward_list, p1_est_list = [], [], []
            p2_log_p, p2_reward_list, p2_est_list = [], [], []
            mask = torch.ones((env.board_size, env.board_size)).to(device)
            cnt = 0

            state, done = env.reset()

            while not done:
                state = torch.from_numpy(np.array(state)).float().to(device)
                state = state.view(1, -1)
                mask = self.update_mask(state=state, mask=mask)

                if torch.eq(mask, 0).all():
                    break

                if is_first is None:
                    if cnt % 2 == 0:
                        probs, score = self.player1(state)
                        p1_est_list.append(score)
                    else:
                        probs, score = self.player2(state)
                        p2_est_list.append(score)
                elif is_first is True:
                    probs, score = self.player1(state)
                    p1_est_list.append(score)
                else:
                    probs, score = self.player2(state)
                    p2_est_list.append(score)

                probs = probs.squeeze(0).squeeze(-1)
                probs = (probs + mask.log()).softmax(dim=-1)
                dist = torch.distributions.Categorical(probs)
                ptr = dist.sample()
                logp = dist.log_prob(ptr)
                logp = logp if not done else logp * 0.0

                state, reward, done = env.step(int(ptr.cpu().item()))

                if is_first is None:
                    if cnt % 2 == 0:
                        p1_log_p.append(logp)
                        if done:
                            p1_reward_list.append(reward * cnt * cnt)
                        else:
                            p1_reward_list.append(reward)
                    else:
                        p2_log_p.append(logp)
                        if done:
                            p2_reward_list.append(reward * cnt * cnt)
                        else:
                            p2_reward_list.append(reward)
                cnt += 1

            if len(p1_log_p) != 0:
                self.player1_opt = optim.Adam(self.player1.parameters(), self.lr)

                reward_list = self.compute_returns(float(p1_est_list[-1].cpu().item()), p1_reward_list)
                log_p = torch.stack(p1_log_p, dim=0).to(device)
                tensor_reward = torch.stack(p1_est_list, dim=0).to(device)
                reward_list = torch.from_numpy(np.array(reward_list)).to(device)

                advantage = reward_list - tensor_reward
                actor_loss = -torch.mean(log_p * advantage.detach())
                critic_loss = torch.mean(advantage.pow(2))

                self.player1_opt.zero_grad()
                actor_loss.backward()
                self.player1_opt.step()

                self.player1_opt.zero_grad()
                critic_loss.backward()
                self.player1_opt.step()

            if len(p2_log_p) != 0:
                self.player2_opt = optim.Adam(self.player2.parameters(), self.lr)

                reward_list = self.compute_returns(float(p2_est_list[-1].cpu().item()), p2_reward_list)
                log_p = torch.stack(p2_log_p, dim=0).to(device)
                tensor_reward = torch.stack(p2_est_list, dim=0).to(device)
                reward_list = torch.from_numpy(np.array(reward_list)).to(device)

                advantage = reward_list - tensor_reward
                actor_loss = -torch.mean(log_p * advantage.detach())
                critic_loss = torch.mean(advantage.pow(2))

                self.player2_opt.zero_grad()
                actor_loss.backward()
                self.player2_opt.step()

                self.player2_opt.zero_grad()
                critic_loss.backward()
                self.player2_opt.step()

            with open(run_log, 'a') as f:
                f.write('%d,%f,%f\n' % (i, sum(p1_reward_list), sum(p2_reward_list)))

            if i % (self.target_update - 1) == 0:
                self.save_model(model_path=self.model_path)

    @torch.no_grad()
    def eval_model(self, env, is_first=None):
        self.player1.eval()
        self.player2.eval()

        mask = torch.ones((env.board_size, env.board_size)).to(device)
        cnt = 0

        state, done = env.reset()

        while not done:
            state = torch.from_numpy(np.array(state)).float().to(device)
            state = state.view(1, -1)
            mask = self.update_mask(state=state, mask=mask)

            if torch.eq(mask, 0).all():
                break

            if is_first is None:
                if cnt % 2 == 0:
                    probs, score = self.player1(state)
                else:
                    probs, score = self.player2(state)
            elif is_first is True:
                probs, score = self.player1(state)
            else:
                probs, score = self.player2(state)

            probs = probs.squeeze(0).squeeze(-1)
            probs = (probs + mask.log()).softmax(dim=-1)
            dist = torch.distributions.Categorical(probs)
            ptr = dist.sample()
            logp = dist.log_prob(ptr)
            logp = logp if not done else logp * 0.0

            state, reward, done = env.step(int(ptr.cpu().item()))
            cnt += 1
