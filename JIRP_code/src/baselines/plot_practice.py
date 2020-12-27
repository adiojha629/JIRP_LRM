#Assume we have 200 times, and the rewards were 0 for the fist 100 and 200 for the second 100
#need a list of length 200, 100 0s then 100 1's
import numpy as np
import random
reward_list = []
for i in range(100):
    reward_list.append(0)
for i in range(100):
    reward_list.append(1)
#check
#print(reward_list[0])
#print(reward_list[100])
#print(reward_list[99])
#print(reward_list[199])

#make dictionary with time steps as the keys and the reward as the values
reward_dict = {}
for b in range(10):
    for i in range(200):
        key = i + 1
        if key in reward_dict.keys():
            reward_dict[key].append(reward_list[i])
        else:
            reward_dict[i+1] = [reward_list[i]]


print(reward_dict[8])
#print(reward_dict[1])
#print(reward_dict[200])

#while True:
 #   a = int(input("Time Step?"))
  #  if( a >= 0 and a <= 200):
   #     print(reward_dict[a])

plot_dict = reward_dict
#create a bunch of empty lists
prc_25 = list()
prc_50 = list()
prc_75 = list()
rewards_plot = list()
steps_plot = list()
current_step = list()
current_25 = list()
current_50 = list()
current_75 = list()
steps_plot = list()

for step in plot_dict.keys():
    if len(current_step) < 10: #if current step has less than 10 elements
        current_25.append(np.percentile(np.array(plot_dict[step]),25))#get the precentiles of values for this step size
        current_50.append(np.percentile(np.array(plot_dict[step]),50))
        current_75.append(np.percentile(np.array(plot_dict[step]),75))
        current_step.append(sum(plot_dict[step])/len(plot_dict[step]))#append the average value to current step
        #I think that the dictionary holds the values from all 10 trials
    else:#if current step has 10 or more entries, then you remove the last values
        current_step.pop(0)
        current_25.pop(0)
        current_50.pop(0)
        current_75.pop(0)
        current_25.append(np.percentile(np.array(plot_dict[step]),25))
        current_50.append(np.percentile(np.array(plot_dict[step]),50))
        current_75.append(np.percentile(np.array(plot_dict[step]),75))
        current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

    rewards_plot.append(sum(plot_dict[step])/len(plot_dict[step]))
    prc_25.append(sum(current_25)/len(current_25))
    prc_50.append(sum(current_50)/len(current_50))
    prc_75.append(sum(current_75)/len(current_75))
    steps_plot.append(step)
import matplotlib.pyplot as plt
plt.plot(prc_25)
plt.ylabel('25th percentile')
plt.show()

plt.plot(prc_50)
plt.ylabel('50th percentile')
plt.show()

plt.plot(prc_75)
plt.ylabel('75th percentile')
plt.show()
