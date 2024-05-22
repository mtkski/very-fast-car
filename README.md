# INFO-H-410 Project

Based on the gymnasium library, and more precisely on the car game from the "box2d" environnement.
https://gymnasium.farama.org/environments/box2d/car_racing/

## Action

action ['+0.00', '+1.00', '+0.00']
The first action value is steering (-1 for full left, +1 for full right). Second one is acceleration, third one is breaking.

## Reward

The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.


To run the project, you need to install the gymnasium library. You can do it by running the following command:


To try the best model just run the following command:

```
python3 racing_car_robot_driving.py
```