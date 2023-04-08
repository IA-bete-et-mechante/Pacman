


1/ Change the my_config.yaml in the config directory
In particular, need to choose the environment, ie the game

NOTE: sometimes if the game has not been used yet, you need to 
add the mapping to play the game in pygame in the file human_play.py
For that, need to go in documentation for this game on gym website, and find all the possible actions. 

2/ cd code

3/python3 human_play.py to play as human


4/python3 train.py to to train.

5/python3 AI_play.py to launch a trained model.
Make sure to 
A/ select the correct path to the saved model
B/ select the correct model type, 
	for example: PPO in  PPO.model()


To see the nice charts during training, just open another terminal and the type the command:
tensorboard --logdir=logs
