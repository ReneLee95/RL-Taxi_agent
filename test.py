import gym
import os
import pygame

pygame.init()
pygame.display.list_modes()

os.environ["SDL_VIDEODRIVER"] = "dummy"

env = gym.make('CartPole-v1')
env.reset()

for _ in range(1000):
	env.render()
	env.step(env.action_space.sample())

env.close()

