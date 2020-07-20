import os
import pickle
import numpy as np
'''
plot_dict ={}
#for loop here
for trial in range(10):
    file_name = "../results/LRM/lrm-qrm/trail_"+str(trial)+"/officeworld/lrm-lrm-qrm-0_rewards_over_time.txt"
    file = open(file_name)
    lines = file.readlines()
    file.close()
    lines = lines[1:]
    lines = [line.replace("\n","").replace("\t","|") for line in lines]
    reward_list = [int(line[line.find("|")+1:]) for line in lines] #get just the rewards
    #print(reward_list[-1])
    print("rewards for trail "+str(trial)+" obtained")
    if 2 in reward_list: #get rewards per step
        print("editing file")
        list_new = []
        last_num = 0
        for num in reward_list:
            list_new.append(num-last_num)
            last_num = num
        reward_list = list_new.copy()
    #at this point reward list has rewards per step
    print("updateing plot_dict")
    for step in range(int(2e6)):
        reward_at_step = reward_list[step]
        if step in plot_dict.keys():
            plot_dict[step].append(reward_at_step)
        else:
            plot_dict[step] = [reward_at_step]
    #now plot_dict is updated
    print("plot_dict updated for trail #" + str(trial))
print(plot_dict[1234])
'''
#save plot_dict
'''
folder = '../results/jul_16_LRM_Officeworld_10_trails'
if not os.path.exists(folder): os.makedirs(folder)
path_file = folder + "/plot_dict.txt"
if not os.path.exists(path_file):
    f = open(path_file, 'wb')
    pickle.dump(plot_dict,f)
    f.close()
    print("\nPlot_dict uploaded")
#plot_dict = pickle.load(path_file)
'''
if __name__ == "__main__":
    print("Loading plot_dict from file")
    f = open('../results/jul_16_LRM_Officeworld_10_trails/plot_dict.txt','rb')
    plot_dict = pickle.load(f)
    f.close()
    #code to plot data
    print("calculating percentiles")
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
    #now use these functions to plot the results
    print("now plotting")
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}

    matplotlib.rc('font', **font)
    axis_font_size = 20

def plot_performance(steps,p25,p50,p75,title,algo): #this is the function I need to replicate
    fig, ax = plt.subplots() #Next three lines set the height and width of figure
    fig.set_figheight(6)
    fig.set_figwidth(8)
    ax.plot(steps, p25, alpha=0) #make 25 percentile transparent
    ax.plot(steps, p50, color='black',label = algo) #put the 50th percentile in black
    ax.plot(steps, p75, alpha=0)
    ax.grid()
    ax.legend()
    plt.fill_between(steps, p50, p25, color='grey', alpha=0.25)#fill in area between p50 and p25
    plt.fill_between(steps, p50, p75, color='grey', alpha=0.25)#fill in area between p50 p75
    plt.title(title)
    plt.xlabel("time step",fontsize = axis_font_size)
    plt.ylabel("reward",fontsize = axis_font_size)
    plt.show()

def plot_this(a1,a2,title,algo):
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    ax.plot(a1,a2, color='blue',label = algo)
    ax.grid()
    ax.legend()
    plt.title(title)
    plt.xlabel("time step",fontsize = axis_font_size)
    plt.ylabel("reward",fontsize = axis_font_size)
    plt.show()


