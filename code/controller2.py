import math
import time
import numpy as np
import torch as to
import torch.nn as nn
import torchvision
from util import *
import matplotlib.pyplot as plt
import gymnasium as gym
from gym import error, spaces, utils
from replay_b import ReplayBuffer
from value_n import ValueNetwork
from policy_n import PolicyNetwork
from softq_n import SoftQNetwork
from util import histogram_costs, histogram_energy, histogram_charging_evs
from scipy import stats


class PPCController(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, alpha, min_timestamp, n_EVs=54, n_levels=10, max_capacity=20):
        super(PPCController, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc2 = nn.Linear(dim_hid, dim_hid)
        self.fc3 = nn.Linear(dim_hid, dim_out)
        self.relu = nn.ReLU()
        self.optimizer = to.optim.Adam(self.parameters(),lr=alpha)

        self.o1 = 0.1
        self.o2 = 0.2
        self.o3 = 2
        self.u = 50

        self.init_alpha = alpha
        self.init_beta = 5
        self.init_gamma = 1
        self.init_signal = None
        self.init_state = None
        self.init_n_EVs = n_EVs
        self.init_n_levels = n_levels
        self.init_max_episode_steps = 100000
        self.init_flexibility = 0
        self.init_penalty = 0
        self.init_tracking_error = 0
        self.init_max_capacity = max_capacity
        self.init_max_rate = 6

        self.min_timestamp = min_timestamp
        self.time_unit = 1


        # Specify the observation space
        lower_bound = np.array([0])
        upper_bound = np.array([24, 70])
        low = np.append(np.tile(lower_bound, self.init_n_EVs * 2), lower_bound)
        high = np.append(np.tile(upper_bound, self.init_n_EVs), np.array([self.init_max_capacity]))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Specify the action space
        upper_bound = self.init_max_rate
        low = np.append(np.tile(lower_bound, self.init_n_EVs), np.tile(lower_bound, self.init_n_levels))
        high = np.append(np.tile(upper_bound, self.init_n_EVs), np.tile(upper_bound, self.init_n_levels))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset_environment()




    # normalization of time by earliest time and hours
    # for example first session is 1.1.2016 13:00 - value = 0
    # second session 1.1.2016 14:00 value = 1
    def process_inputs(self, inputs,r_t_j):
        (_, count) = inputs.shape
        new_arr = []
        for j in range(0, count, 1):
            x = inputs[:, j]
            a_t_j = x[0]
            d_t_j = x[1]
            e_t_j = float(x[2])
            a_t_j_timestamp = (time.mktime(time.strptime(a_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(
                time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600
            d_t_j_timestamp = (time.mktime(time.strptime(d_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(
                time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600
            new_arr.append([a_t_j_timestamp, d_t_j_timestamp, e_t_j, r_t_j])
        return new_arr

    def min_function(self, t):
        t = t % 24
        return 1 - (t / 24)
    def reset_environment(self):

        self.alpha = self.init_alpha
        self.beta = self.init_beta
        self.gamma = self.init_gamma
        self.signal = self.init_signal
        self.state = self.init_state
        self.n_EVs = self.init_n_EVs
        self.n_levels = self.init_n_levels
        self._max_episode_steps = self.init_max_episode_steps
        self.flexibility = self.init_flexibility
        self.penalty = self.init_penalty
        self.tracking_error = self.init_tracking_error
        self.max_capacity = self.init_max_capacity
        self.max_rate = self.init_max_rate

        return self.state

    def calculate_mef(self, action):
        if not np.any(action[-self.n_levels:]):
            self.flexibility = 0
        else:
            self.flexibility = self.alpha * (stats.entropy(action[-self.n_levels:])) ** 2


    def calculate_reward(self):
        ...
    def sample_action(self):
        return self.action_space.sample()


    def update(self, batch_size, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Calculate Q-values using value network
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = to.nn.functional.mse_loss(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = to.nn.functional.mse_loss(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()

        self.soft_q_optimizer2.step()
        # Training V function
        predicted_new_q_value = to.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
        print("V Loss")
        print(value_loss)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Update policy network using the policy optimization objective

        policy_loss = (log_prob - predicted_new_q_value).mean()  # Policy objective
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def training_experimental(self):
        self.action_dim = 10
        self.state_dim = 108
        self.hidden_dim = 256
        self.batch_size = 256
        polyak_parameter = 0.99
        target_smoothing_coefficient = 0.005
        self.done = False
        self.eps = 1000
        self.temperature_parameter = 0.5
        self.lr = 3 * 10**-4


        self.value_net = ValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.target_value_net = ValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)

        # create Q networks
        self.soft_q_net1 = SoftQNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.policy_net = PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)


        # Create optimizers
        self.value_optimizer = to.optim.Adam(self.value_net.parameters(),lr=self.lr)
        self.soft_q_optimizer1 = to.optim.Adam(self.soft_q_net1.parameters(), lr=self.lr)
        self.soft_q_optimizer2 = to.optim.Adam(self.soft_q_net2.parameters(), lr=self.lr)
        self.policy_optimizer = to.optim.Adam(self.policy_net.parameters(),lr=self.lr)

        self.replay_buffer = ReplayBuffer(10**6)
        self.done = False
        iterations = 1000
        environment_steps = 1000
        gradient_steps = 1000
        while not self.done:

            for env_step in range(environment_steps):
                self.state = self.reset_environment()
                self.action = self.sample_action()
                reward, next_state, done = self.step(self.action)
                self.replay_buffer.push(state=self.state,action=self.action,reward=reward,next_state=next_state,done=done)

            for grad_step in range(gradient_steps):
                self.update(batch_size=self.batch_size,)


    # MPE is testing the ratio of s(j) and e(j), - violation of condition 4d)
    # rate of undelivered energy
    def mpe(self, decisions, energy_requests, L, T, N):
        res = 0
        for k in range(L):
            for t in range(T):
                for j in range(N):
                    res += decisions[k][t][j]
        res /= (np.sum(energy_requests))
        res = 1 - res
        return res

    def mpe_updated(self, decisions, energy_requests, L, T, N):
        return 1 - (np.sum(decisions) / np.sum(energy_requests))
    # MSE measures - violation of condition 4c)
    def mse(self):
        ...
    def get_other(self,inputs):

        a_t_js = []
        d_t_js = []
        e_t_js = []
        r_t_js = []
        for input in inputs:
            a_t_js.append(input[0])
            d_t_js.append(input[1])
            e_t_js.append(input[2])
            r_t_js.append(input[3])
        return a_t_js, d_t_js, e_t_js, r_t_js
    def employ_schedule(self, ideal_schedule, energy_requests, time_horizon, num_of_vehicles):
        costs = {}
        energies = {}
        charging_evs = {}
        updated_energy_requests = energy_requests.copy()
        decisions = np.zeros((time_horizon + 1, num_of_vehicles))
        for t in range(time_horizon + 1):
            costs[t] = 0
            charging_evs[t] = 0
            energies[t] = 0
        overall_costs = 0
        for key in ideal_schedule.keys():
            value = ideal_schedule[key]
            if value != 0:
                starting_time = value[0]
                ending_time = value[1]
                remaining_time = ending_time - starting_time
                provided_energy = energy_requests[key]
                updated_energy_requests[key] -= provided_energy
                decisions[starting_time][key] = provided_energy
                energies[starting_time] += provided_energy
                costs[starting_time] += self.min_function(starting_time) * provided_energy
                overall_costs += self.min_function(starting_time) * provided_energy
                charging_evs[starting_time] += 1
        return costs, energies, overall_costs, charging_evs, decisions, updated_energy_requests

    def get_state(self, time_intervals, arrival_times, departure_times, energy_requests):
        x_t_states = {}
        for time in time_intervals:
            x_t_states = x_t_states.get(time, [])
            for car_id in range(len(arrival_times)):
                if arrival_times[car_id] <= time <= departure_times[car_id]:
                    x_t_states[time].append(departure_times[car_id],energy_requests[car_id])
    def my_train(self, inputs,rj, T = 10):
        (_, count) = inputs.shape
        loss_fn = nn.MSELoss()
        new_inputs = self.process_inputs(inputs,rj)
        a_t_js, d_t_js, e_t_js, r_t_js = self.get_other(new_inputs)
        arrival_times,\
            departure_times,\
            energy_requests,\
            peak_charging_rates = a_t_js, d_t_js, e_t_js, r_t_js
        time_horizon = T
        time_intervals = np.arange(time_horizon + 1)
        num_evs = len(new_inputs)
        interval_num_evs = np.arange(num_evs + 1)

        N = num_evs

        all_schedules = []

        ideal_schedule = {}
        for interval in time_intervals:
            chargers_per_time = np.ones(time_horizon + 1) * self.num_of_chargers
            schedule, energy_delivered, charging_ev = self.LLF(arrival_times, departure_times, energy_requests,
                                                               peak_charging_rates, interval, time_horizon,
                                                               chargers_per_time)
            all_schedules.append(schedule)
            for key in schedule.keys():
                value = schedule[key]
                if value != 0:
                    # giving more energy in less than hour is ineffective because of min function
                    if value[1] - value[0] >= 1:
                        ideal_schedule[key] = value
            # costs += self.min_function(interval) * energy_delivered
            # costs_array.append(costs)
            # charging_evs.append(charging_ev)

            # edfs = self.EDF(d_t_js, a_t_js, e_t_js, rj, interval,time_horizon)
            # for time_unit in range(interval)
        costs, energies, all_costs, charging_evs, decisions, updated_energy_requests = self.employ_schedule(ideal_schedule, energy_requests, T, num_evs)
        mpe1 = self.mpe(T, N, decisions, energy_requests)
        mpe2 = self.mpe_updated(T, N, decisions, energy_requests)
        print('Basic version of MPE = ',mpe1)
        print('Optimalized version of MPE =',mpe2)
        histogram_costs(time_intervals, costs.values(), time_intervals)
        histogram_energy(time_intervals, energies.values(),time_intervals)
        # histogram_charging_evs(time_intervals, charging_evs.values(), time_intervals, interval_num_evs)


    def LLF(self,
            arrival_times,
            departure_times,
            energy_requests,
            peak_charging_rate,
            t,
            T,
            chargers_per_time):
        '''
        arrival_times = a_t(j) j = 1...N
        departure_times = d_t(j) j = 1...N
        energy requested = e_t(j) j = 1...N
        t = current time from {1,...,T} at each time t je provide energy
        T = time horizon
        chargers_per_time = available chargers at each time t
        '''
        # assert len(djs) == len(ejs), 'Three arrays djs, ejs, rjs must have same length.'
        n = len(departure_times)
        lax_arr = []
        for car_id in range(n):
            # laxity is even minus number possibly, but mostly from R^+
            laxity = departure_times[car_id] - energy_requests[car_id] / peak_charging_rate[car_id]
            lax_arr.append([car_id, laxity, arrival_times[car_id], departure_times[car_id],energy_requests[car_id]])
        sorted_lax_array = sorted(lax_arr, key=lambda x: x[1])
        schedule = {}
        energy_delivered = 0
        current_time = t
        charging_ev = 0
        while sorted_lax_array:
            car_id, least_laxity_task, arrival_time, departure_time, energy_request = sorted_lax_array.pop(0)
            # if car has not arrived yet
            if arrival_time > current_time:
                schedule[car_id] = 0
                continue
            # if car has already left
            if departure_time < current_time:
                schedule[car_id] = 0
                continue
            # if user of car got all energy requested
            if energy_request == 0:
                schedule[car_id] = None
                continue
            # if there is no charger for car left
            if chargers_per_time[current_time] <= 0:
                schedule[car_id] = 0
                continue

            start_time = t
            end_time = departure_time
            schedule[car_id] = [start_time, end_time]
            energy_delivered += energy_request
            charging_ev += 1
            chargers_per_time[current_time] -= 1
            # schedule.append([car_id, start_time, end_time])

        return schedule, energy_delivered, charging_ev

    def EDF(self,
            arrival_times,
            departure_times,
            energy_requests,
            peak_charging_rate,
            t,
            T,
            chargers_per_time
            ):
        # assert len(djs) == len(ejs), 'Three arrays djs, ejs, rjs must have same length.'
        n = len(departure_times)
        deadline_arr = []
        for car_id in range(n):
            deadline_arr.append([car_id, arrival_times[car_id], departure_times[car_id], energy_requests[car_id]])
        sorted_deadline_arr = sorted(deadline_arr,key=lambda x: x[1])
        current_time = t
        schedule = {}
        charging_ev = 0
        energy_delivered = 0
        while sorted_deadline_arr:
            car_id, arrival_time, departure_time, energy_request = deadline_arr.pop(0)
            if arrival_time > current_time:
                schedule[car_id] = 0
                continue
            if departure_time > current_time:
                schedule[car_id] = 0
                continue
            if energy_request == 0:
                schedule[car_id] = 0
                continue
            if chargers_per_time[current_time] <= 0:
                schedule[car_id] = 0
                continue
            start_time = t
            end_time = departure_time
            energy_delivered += energy_request
            charging_ev += 1
            schedule[car_id] = [start_time, end_time]
            chargers_per_time[current_time] -= 1

        return schedule, energy_delivered, charging_ev

