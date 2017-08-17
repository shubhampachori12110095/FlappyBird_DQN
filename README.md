# FlappyBird_DQN
This program uses the Deep Reinforcement Learning to play the Flappy Bird, the origin paper of this method is ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 

## Acknowledgement:
This project is based on the work of https://github.com/yenchenlin/DeepLearningFlappyBird, Many thanks to his inspired. 

## Pre-Requisite
* tensorflow 1.1.0
* python 2.7.x
* OpenAI gym
* OpenAI gym_ple
* OpenCV-python

## Instruction to run the code:
### 1. Substitute the background of the OpenAI gym and change the reward
Firstly,  we need to substitute the background image of OpenAI gym. The substitute images are in the `assets_substitute` folder. The background images in the openai gym should in the path: `PyGame-Learning-Environment-master/ple/games/flappybird/assets`.  
You can also revise the source code of the game to get the same result, the file is in the path: `PyGame-Learning-Environment-master/ple/games/flappybird/__init__.py`. However, you should also change the background to the black image!
  
Secondly, we need to change the reward of the game, the file is in the path: `PyGame-Learning-Environment-master/ple/games/base/pygamewrapper.py`. Then change the reward as follows:
```python
def __init__(self, width, height, actions={}):

        # Required fields
        self.actions = actions  # holds actions

        self.score = 0.0  # required.
        self.lives = 0  # required. Can be 0 or -1 if not required.
        self.screen = None  # must be set to None
        self.clock = None  # must be set to None
        self.height = height
        self.width = width
        self.screen_dim = (width, height)  # width and height
        self.allowed_fps = None  # fps that the game is allowed to run at.
        self.NOOP = K_F15  # the noop key
        self.rng = None

        self.rewards = {
            "positive": 0.9,
            "negative": -1.0,
            "tick": 0.1,
            "loss": -1.1,
            "win": 5.0
        }
```

### 2. Train the DQN
The training process is just following the below:
```bash
cd root_of_this_code/
python train.py
```
The trainning parameter is use the default setting, you can also set the parameters as you want when you train some different games. The parameter list could be found in the `DQN_Brain.py`. Another thing needs to be noticed is, if you want to train some different games, you need to re-write the function `image_processing()` as you needed. Or you can also just change the code like following. However, it may takes longer time to be converged:
```python
def image_processing(self, image):
	# convert the image to the gray scale..,
	image = image[:, :, (2, 1, 0)]
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#_, image_final = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
	image_resize = cv2.resize(gray_image, (80, 80))
	return image_resize
```
I trained the network by using the `GTX1080 Ti` for about `12 hours`(about `4500000` time steps) and the highest score of the results is `264`. It was much better than what I played.

### 3. Test the DQN
The test process is just following the below:
```bash
cd root_of_this_code/
python test.py
```
### 4. Related Papers
[1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).  

[2] K. Arulkumaran, M. Deisenroth, M. Brundage, and A. A. Bharath, “A Brief Survey of Deep Reinforcement Learning,” IEEE Signal Processing Magazine, 2017. 




