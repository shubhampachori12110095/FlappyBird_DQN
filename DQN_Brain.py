import tensorflow as tf
import numpy as np
import cv2
import random

# In this programe, I use the experience replay method which can be found in 
class Experience_Replay:

	def __init__(self, env, action_space = [0, 1], REPLAY_MEMORY = 50000, OBSERVE = 10000, FINAL_RATIO = 0.0001, INITIAL_RATIO = 0.1, EXPLORE = 3000000, GAMMA = 0.99, BATCH_SIZE = 32, TEST_MODE = False):
		self.action_space = action_space
		self.REPLAY_MEMORY = REPLAY_MEMORY
		self.OBSERVE = OBSERVE
		self.FINAL_RATIO = FINAL_RATIO
		self.INITIAL_RATIO = INITIAL_RATIO
		self.EXPLORE = EXPLORE
		self.BATCH_SIZE = BATCH_SIZE
		self.TEST_MODE = TEST_MODE
		self.GAMMA = GAMMA

		self.env = env
		self.env.reset()

		self.time_step = 0
		self.check_state = True
		self.reward_total = 0
		self.pipe_number = 0
		self.episode_number = 0
		self.greedy_ratio = INITIAL_RATIO
		self.brain_memory = []

		self.build_up_network()

		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	####################################################################################################################
	
	# define the weight and CNN layers of tensorflow

	def weight_variables(self, weight_size):
		weight = tf.Variable(tf.truncated_normal(weight_size, stddev = 0.01)) # size should be [height, weights, in, out]
		return weight

	def bias_variables(self, bias_size):
		bias = tf.Variable(tf.constant(0.01, shape = bias_size))
		return bias

	def conv2D(self, input_feature, kernel, strides, conv_type = "VALID"):
		result = tf.nn.conv2d(input_feature, kernel, [1, strides[0], strides[1], 1], conv_type) # input should be NHWC
		return result

	def maxPooling(self, input_feature, kernel_size, strides, pool_type = "VALID"):
		result = tf.nn.max_pool(input_feature, [1, kernel_size[0], kernel_size[1], 1], [1, strides[0], strides[1], 1], pool_type)
		return result

	#####################################################################################################################

	# build up the network!!! the network architecture is used yenchenlin's work.
	def build_up_network(self):		
		self.input_feature = tf.placeholder(tf.float32, [None, 80, 80, 4])
		self.input_target_action_value = tf.placeholder(tf.float32, [None])
		self.input_action = tf.placeholder(tf.float32, [None, 2])

		W_conv1 = self.weight_variables([8, 8, 4, 32])
		b_conv1 = self.bias_variables([32])

		W_conv2 = self.weight_variables([4, 4, 32, 64])
		b_conv2 = self.bias_variables([64])

		W_conv3 = self.weight_variables([3, 3, 64, 64])
		b_conv3 = self.bias_variables([64])

		W_fc1 = self.weight_variables([1600, 512])
		b_fc1 = self.bias_variables([512])

		W_fc2 = self.weight_variables([512, 2])
		b_fc2 = self.bias_variables([2])

		# define the network architecture
		conv1_out = tf.nn.relu(self.conv2D(self.input_feature, W_conv1, [4, 4], "SAME") + b_conv1)
		max1_out = self.maxPooling(conv1_out, [2, 2], [2, 2], "SAME")

		conv2_out = tf.nn.relu(self.conv2D(max1_out, W_conv2, [2, 2], "SAME") + b_conv2)
		#max2_out = maxPooling(conv2_out, [2, 2])

		conv3_out = tf.nn.relu(self.conv2D(conv2_out, W_conv3, [1, 1], "SAME") + b_conv3)
		conv3_out_reshape = tf.reshape(conv3_out, [-1, 1600]) # 5 * 5 * 64

		fc1_out = tf.nn.relu(tf.matmul(conv3_out_reshape, W_fc1) + b_fc1)

		self.predicted_value = tf.matmul(fc1_out, W_fc2) + b_fc2

		selected_action_value = tf.reduce_sum(tf.multiply(self.predicted_value, self.input_action), reduction_indices = 1)

		loss = tf.reduce_mean(tf.square(self.input_target_action_value - selected_action_value))

		self.train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

	# this function is used to pre-processing the input images
	def image_processing(self, image):
		# convert the image to the gray scale..,
		image = image[:, :, (2, 1, 0)]
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, image_final = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
		image_resize = cv2.resize(image_final, (80, 80))
		return image_resize

	def load_the_models(self, model_path):
		self.saver.restore(self.sess, model_path) # 2800000 last time ...

	# this is used to choose the action 
	def choose_the_action(self, action_value, action_space, greedy_ratio):

		if random.random() < greedy_ratio:
			action_index = random.randrange(len(action_space))
			selected_action = action_space[action_index]
		else:
			action_index = np.argmax(action_value)
			selected_action = action_space[action_index]

		return selected_action

	def transfer_to_one_hot_encoding(self, action):
		if action == 0:
			transfer_action = [1, 0]
		else:
			transfer_action = [0, 1]

		return transfer_action


	def start_the_trainning_process(self):

		while True:
			if self.TEST_MODE:
				self.env.render()

			if self.check_state == True:
				observation, _, _, _ = self.env.step(0) # we need to skip the first frame of the game, because it is black!
				img_processed = self.image_processing(observation)
				state_current = np.stack((img_processed, img_processed, img_processed, img_processed), axis = 2)
				self.check_state = False

			action_value = self.sess.run(self.predicted_value, feed_dict = {self.input_feature: [state_current]})
			selected_action = self.choose_the_action(action_value, self.action_space, self.greedy_ratio)
			transfer_action = self.transfer_to_one_hot_encoding(selected_action) # transfer to the one-hot mode, easy for use in the behind
			
			observation, reward, done, _ = self.env.step(selected_action)

			image_processed = self.image_processing(observation).reshape([80, 80, 1])
			state_next = np.append(image_processed, state_current[:, :, :3], axis = 2)

			# process some other information
			self.reward_total += reward
			if reward >= 0.9:
				#print reward
				self.pipe_number += 1

			# need to process the greedy, we dont need too much random actions in the behind...
			if self.greedy_ratio > self.FINAL_RATIO and self.time_step > self.OBSERVE:
				self.greedy_ratio -= (self.INITIAL_RATIO - self.FINAL_RATIO) / self.EXPLORE

			if self.TEST_MODE:
				state_current = state_next
				if done:
					self.check_state = True
					print 'The episode number is ', self.episode_number, ', the reward is ', self.reward_total, ' and its score is ', self.pipe_number
					self.pipe_number = 0
					self.episode_number += 1
					self.env.reset()

			else:
				self.brain_memory.append((state_current, reward, transfer_action, state_next, done))
				state_current = state_next

				if len(self.brain_memory) > self.REPLAY_MEMORY:
					self.brain_memory.pop(0)

				if self.time_step > self.OBSERVE:
					minibatch = random.sample(self.brain_memory, self.BATCH_SIZE)

					state_current_batch = [element[0] for element in minibatch]
					reward_batch = [element[1] for element in minibatch]
					action_batch = [element[2] for element in minibatch]
					state_next_batch = [element[3] for element in minibatch]
					done_batch = [element[4] for element in minibatch]

					target_action_value_temp = []

					# predict the next_state_value
					target_action_value_next = self.sess.run(self.predicted_value, feed_dict = {self.input_feature: state_next_batch})

					for index in range(self.BATCH_SIZE):
						if done_batch[index] == True:
							target_action_value = reward_batch[index]
						else:
							target_action_value = reward_batch[index] + self.GAMMA * np.max(target_action_value_next[index])

						target_action_value_temp.append(target_action_value)

					self.sess.run(self.train_step, feed_dict = {self.input_feature: state_current_batch, self.input_target_action_value: target_action_value_temp, self.input_action: action_batch})

				self.time_step += 1

				if done:
					self.check_state = True
					self.episode_number += 1
					print 'The episode number is ', self.episode_number, ' and the reward total is ', self.reward_total, ' and the time step is ', self.time_step, ' and the length of memory', len(self.brain_memory),' and pipe numbers is ', self.pipe_number
					self.reward_total = 0
					self.pipe_number = 0
					self.env.reset()

				if self.time_step % 10000 == 0:
					self.saver.save(self.sess, "MyModel/save_net" , global_step = self.time_step)
					print '######################Model has been saved!!!######################'














