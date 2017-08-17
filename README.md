# FlappyBird_DQN
This program uses the Deep Reinforcement Learning to play the Flappy Bird, the origin paper of this method is ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 

## Pre-Requisite
* tensorflow 1.1.0
* python 2.7.x
* OpenAI gym
* OpenAI gym_ple
* OpenCV-python

## Instruction to run the code:
### Sub

### Train the DQN
The training process is just following the below:
```bash
cd root_of_this_code/
python train.py
```
The trainning parameter is use the default setting, you can also set the parameters as you want when you train some different games. The parameter list could be found in the DQN_Brain.py. Another thing needs to be noticed is, if you want to train some different games, you need to re-write the function image_processing() as you needed. Or you can also just change the code like following. However, it may takes longer time to be converged:
```python
def image_processing(self, image):
		# convert the image to the gray scale..,
		image = image[:, :, (2, 1, 0)]
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#_, image_final = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
		image_resize = cv2.resize(gray_image, (80, 80))
		return image_resize
```



