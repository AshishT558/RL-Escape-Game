import time
import pickle
import numpy as np
from vis_gym import *


gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g




# update Qtable according to Bellman equation
# Parameters:
# - the Q table at t-1
# - the updates matrix
# - the previous observation state (unhashed)
# - the action taken from the previous observation state
# - the reward for that action
# - the new observation state (unhased)
# - the discount factor gamma
def update_Qtable(curr_Q_table, updates_matrix, old_obs, action_taken, reward, new_obs, gamma):
	# print("ACTION TAKEN:", action_taken)

	# hash the old state 
	curr_obs_key = hash(old_obs)
	# calulate learning rate and increment the count of updates for this state action pair
	updates_matrix[curr_obs_key][action_taken] += 1
	learning_rate = 1 / (1 + updates_matrix[curr_obs_key][action_taken])

	### Equation terms
	#Q-opt for (obs, action) before this update -> initialize to all 0s if entry doesn't exist
	if curr_obs_key not in curr_Q_table:
		curr_Q_table[curr_obs_key] = np.zeros(6)
		old_q_opt = 0
	else:
		old_q_opt = curr_Q_table[curr_obs_key][action_taken]
	# print("OLD: ", curr_Q_table[curr_obs_key])

	#V-opt for (new_obs) before this update -> initialize to all 0s if entry doesn't exist
	next_obs_key = hash(new_obs)
	if next_obs_key not in curr_Q_table:
		curr_Q_table[next_obs_key] = np.zeros(6)
		next_state_opt = 0
	else:
		next_state_opt = np.max(curr_Q_table[next_obs_key])


	# calculate q-opt from bellman ford equation
	new_q_opt = (1 - learning_rate) * old_q_opt + learning_rate * (reward + (gamma * next_state_opt))
	# print(new_q_opt)
	# update q-table
	curr_Q_table[curr_obs_key][action_taken] = new_q_opt
	# print("NEW: ", curr_Q_table[curr_obs_key])
	return curr_Q_table

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {}
	updates_matrix = np.zeros((375, 6))

	# play all episodes, keep track of 
	for episode in range(num_episodes):
		if episode % 10000 == 0:
			print("Episode #", episode / num_episodes)
			print("Epsilon: ", epsilon)
			# print("Q-table", Q_table)
		
		
		#reset prior to each episode
		obs, reward, done, info = env.reset()

		# while episode has not terminated
		while not done:

			# hash current state 
			obs_key = hash(obs)

			# choose action based on epsilon probability -> either random action or argmax of Q-table for the current state(hashed)

			rand_probability = random.random()

			# explore with random action
			if rand_probability < epsilon:
				# print("random")
				action = env.action_space.sample()

				
			# exploit with argmax of qtable for obs_key
			else:
				# print("qtable")
				if obs_key not in Q_table:
					Q_table[obs_key] = np.zeros(6)
				action = np.argmax(Q_table[obs_key])

			# execute new action
			new_obs, reward, done_status, info = env.step(action)
			
			# if gui_flag:
			# 	refresh(new_obs, reward, done_status, info)

			# update q table 
			Q_table = update_Qtable(Q_table, updates_matrix, obs, action, reward, new_obs, gamma)
			
			# update done to done_status
			done = done_status
			obs = new_obs
		# update epsilon after episode finishes
		epsilon = epsilon * decay_rate

	return Q_table

decay_rate = 0.999999

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning
# Q_table = Q_learning(num_episodes=100000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
