import gym
import gym_ple
import tensorflow as tf
import numpy as np
import cv2
import random

# we will use the experience replay method to play this game...

# define some functions that will be used...
def image_processing(image):
	# convert the image to the gray scale..,
	image = image[:, :, (2, 1, 0)]
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, image_final = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
	image_resize = cv2.resize(image_final, (80, 80))
	return image_resize

def choose_the_action(action_value, action_space, greedy_ratio):
	if random.random() < greedy_ratio:
		action_index = random.randrange(len(action_space))
		selected_action = action_space[action_index]
	else:
		action_index = np.argmax(action_value)
		selected_action = action_space[action_index]

	return selected_action

def transfer_to_one_hot_encoding(action):
	if action == 0:
		transfer_action = [1, 0]
	else:
		transfer_action = [0, 1]

	return transfer_action

# define the weight variables and layers of the tensorflow...

def weight_variables(weight_size):
	weight = tf.Variable(tf.truncated_normal(weight_size, stddev = 0.01)) # size should be [height, weights, in, out]
	return weight

def bias_variables(bias_size):
	bias = tf.Variable(tf.constant(0.01, shape = bias_size))
	return bias

def conv2D(input_feature, kernel, strides, conv_type = "VALID"):
	result = tf.nn.conv2d(input_feature, kernel, [1, strides[0], strides[1], 1], conv_type) # input should be NHWC
	return result

def maxPooling(input_feature, kernel_size, strides, pool_type = "VALID"):
	result = tf.nn.max_pool(input_feature, [1, kernel_size[0], kernel_size[1], 1], [1, strides[0], strides[1], 1], pool_type)
	return result

################################ Build up the Q - Value Network ##########################################
# define the variables
input_feature = tf.placeholder(tf.float32, [None, 80, 80, 4])
input_target_action_value = tf.placeholder(tf.float32, [None])
input_action = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_variables([8, 8, 4, 32])
b_conv1 = bias_variables([32])

W_conv2 = weight_variables([4, 4, 32, 64])
b_conv2 = bias_variables([64])

W_conv3 = weight_variables([3, 3, 64, 64])
b_conv3 = bias_variables([64])

W_fc1 = weight_variables([1600, 512])
b_fc1 = bias_variables([512])

W_fc2 = weight_variables([512, 2])
b_fc2 = bias_variables([2])

# define the network architecture
conv1_out = tf.nn.relu(conv2D(input_feature, W_conv1, [4, 4], "SAME") + b_conv1)
max1_out = maxPooling(conv1_out, [2, 2], [2, 2], "SAME")

conv2_out = tf.nn.relu(conv2D(max1_out, W_conv2, [2, 2], "SAME") + b_conv2)
#max2_out = maxPooling(conv2_out, [2, 2])

conv3_out = tf.nn.relu(conv2D(conv2_out, W_conv3, [1, 1], "SAME") + b_conv3)
conv3_out_reshape = tf.reshape(conv3_out, [-1, 1600]) # 5 * 5 * 64

fc1_out = tf.nn.relu(tf.matmul(conv3_out_reshape, W_fc1) + b_fc1)

predicted_value = tf.matmul(fc1_out, W_fc2) + b_fc2

selected_action_value = tf.reduce_sum(tf.multiply(predicted_value, input_action), reduction_indices = 1)

loss = tf.reduce_mean(tf.square(input_target_action_value - selected_action_value))

train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

# ################################## Q-TARGET NETWORK!!!! ################################################

# # Start to build the Target - Q Value Network!!!
# input_feature_T = tf.placeholder(tf.float32, [None, 80, 80, 4])

# W_conv1_T = weight_variables([8, 8, 4, 32])
# b_conv1_T = bias_variables([32])

# W_conv2_T = weight_variables([4, 4, 32, 64])
# b_conv2_T = bias_variables([64])

# W_conv3_T = weight_variables([3, 3, 64, 64])
# b_conv3_T = bias_variables([64])

# W_fc1_T = weight_variables([1600, 512])
# b_fc1_T = bias_variables([512])

# W_fc2_T = weight_variables([512, 2])
# b_fc2_T = bias_variables([2])

# # define the network architecture
# conv1_out_T = tf.nn.relu(conv2D(input_feature_T, W_conv1_T, [4, 4], "SAME") + b_conv1_T)
# max1_out_T = maxPooling(conv1_out_T, [2, 2], [2, 2], "SAME")

# conv2_out_T = tf.nn.relu(conv2D(max1_out_T, W_conv2_T, [2, 2], "SAME") + b_conv2_T)
# #max2_out = maxPooling(conv2_out, [2, 2])

# conv3_out_T = tf.nn.relu(conv2D(conv2_out_T, W_conv3_T, [1, 1], "SAME") + b_conv3_T)
# conv3_out_T_reshape = tf.reshape(conv3_out_T, [-1, 1600]) # 5 * 5 * 64

# fc1_out_T = tf.nn.relu(tf.matmul(conv3_out_T_reshape, W_fc1_T) + b_fc1_T)

# predicted_value_T = tf.matmul(fc1_out_T, W_fc2_T) + b_fc2_T

# ###############################################################################################

# initialize
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver.restore(sess, "MyModel/save_net_-1670000") # 2800000 last time ...

##############################################################################################3

# make the environment
ACTION_SPACE = [0, 1]
REPLAY_MEMORY = 50000
OBSERVE = 10000
EXPLORE = 3000000
GAMMA = 0.99
FINAL_RATIO = 0.0001
INITIAL_RATIO = 0.1
BATCH_SIZE = 32
#UPDATE_STEP = 300

time_step = 0 
greedy_ratio = INITIAL_RATIO
new_start = True
reward_total = 0
episode_number = 0
pipe_number = 0

brain_memory = []

# start to set_up the environment!
env = gym.make("FlappyBird-v0")
observation = env.reset()


while True:
	#env.render()
	if new_start == True:
		observation, _, _, _ = env.step(0) # we need to skip the first frame of the game, because it is black!
		#observation, _, _ = env.frame_step([1, 0])
		img_processed = image_processing(observation)
		state_current = np.stack((img_processed, img_processed, img_processed, img_processed), axis = 2)
		new_start = False

	action_value = sess.run(predicted_value, feed_dict = {input_feature: [state_current]})
	selected_action = choose_the_action(action_value, ACTION_SPACE, greedy_ratio)
	transfer_action = transfer_to_one_hot_encoding(selected_action)
	#observation, reward, done = env.frame_step(transfer_action)
	observation, reward, done, _ = env.step(selected_action)

	if reward >= 0.9:
		pipe_number += 1
	#transfer_action = transfer_to_one_hot_encoding(selected_action)
	# change the reward !!!
	#reward_new = change_the_reward(reward)
	reward_total += reward

	# process the greedy ratio
	if greedy_ratio > FINAL_RATIO and time_step > OBSERVE:
		greedy_ratio -= (INITIAL_RATIO - FINAL_RATIO) / EXPLORE

	# process the next state ...
	image_processed = image_processing(observation).reshape([80, 80, 1])
	state_next = np.append(image_processed, state_current[:, :, :3], axis = 2)

	# start to store the information
	brain_memory.append((state_current, reward, transfer_action, state_next, done))

	if len(brain_memory) > REPLAY_MEMORY:
		brain_memory.pop(0)

	if time_step > OBSERVE:
		minibatch = random.sample(brain_memory, BATCH_SIZE)

		state_current_batch = [element[0] for element in minibatch]
		reward_batch = [element[1] for element in minibatch]
		action_batch = [element[2] for element in minibatch]
		state_next_batch = [element[3] for element in minibatch]
		done_batch = [element[4] for element in minibatch]

		target_action_value_temp = []

		# predict the next_state_value
		target_action_value_next = sess.run(predicted_value, feed_dict = {input_feature: state_next_batch})

		for index in range(BATCH_SIZE):
			if done_batch[index] == True:
				target_action_value = reward_batch[index]
			else:
				target_action_value = reward_batch[index] + GAMMA * np.max(target_action_value_next[index])

			target_action_value_temp.append(target_action_value)


		sess.run(train_step, feed_dict = {input_feature: state_current_batch, input_target_action_value: target_action_value_temp, input_action: action_batch})

	time_step += 1
	state_current = state_next

	if done:
		new_start = True
		episode_number += 1
		print 'The episode number is ', episode_number, ' and the reward total is ', reward_total, ' and the time step is ', time_step, ' and the length of memory', len(brain_memory),' and pipe numbers is ', pipe_number
		reward_total = 0
		pipe_number = 0
		env.reset()

	if time_step % 10000 == 0:
		save_path = saver.save(sess, "MyModel/save_net" , global_step = time_step)
		print '######################Model has been saved!!!######################'

	# if time_step % UPDATE_STEP == 0:

	# 	sess.run(tf.assign(W_conv1_T, W_conv1))
	# 	sess.run(tf.assign(b_conv1_T, b_conv1))

	# 	sess.run(tf.assign(W_conv2_T, W_conv2))
	# 	sess.run(tf.assign(b_conv2_T, b_conv2))

	# 	sess.run(tf.assign(W_conv3_T, W_conv3))
	# 	sess.run(tf.assign(b_conv3_T, b_conv3))

	# 	sess.run(tf.assign(W_fc1_T, W_fc1))
	# 	sess.run(tf.assign(b_fc1_T, b_fc1))

	# 	sess.run(tf.assign(W_fc2_T, W_fc2))
	# 	sess.run(tf.assign(b_fc2_T, b_fc2))

	# 	print 'Update the TARGET NETWORK !!!'






		













