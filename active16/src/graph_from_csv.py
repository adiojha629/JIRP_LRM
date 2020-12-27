import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_performance(steps,p25,p50,p75,title,algo): #this is the function I need to replicate
    fig, ax = plt.subplots() #Next three lines set the height and width of figure
    fig.set_figheight(6)
    fig.set_figwidth(8)
    ax.plot(steps, p25, alpha=0) #make 25 percentile transparent
    ax.plot(steps, p50, color='black',label = algo) #put the 50th percentile in black
    ax.plot(steps, p75, alpha=0)
    ax.grid()
    plt.fill_between(steps, p50, p25, color='grey', alpha=0.25)#fill in area between p50 and p25
    plt.fill_between(steps, p50, p75, color='grey', alpha=0.25)#fill in area between p50 p75
    ax.set_xlabel('number of steps', fontsize=22)
    ax.set_ylabel('reward', fontsize=22)#changing the font of axis labels
    plt.locator_params(axis='x', nbins=5)
    plt.gcf().subplots_adjust(bottom=0.15) #adjusts the plot
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.savefig('figure_1.png', dpi=600)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_this(a1,a2,title,algo):
    fig, ax = plt.subplots()
    ax.plot(a1,a2, color='blue',label = algo)
    ax.grid()
    ax.set(xlabel='number of steps', ylabel='reward')
    plt.title(title)
    plt.legend()
    plt.show()

file = "../plotdata/officeworld8aqrm.csv"
plot_dict = {}
test_freq = 7000
counter = 0
with open(file) as csv_file:
    reader = csv.reader(csv_file)
    for step in reader:
        values = [float(value) for value in step]
        counter+=test_freq
        plot_dict[counter] = values
prc_25 = list()
prc_50 = list()
prc_75 = list()

# Buffers for plots
current_step = list()
current_25 = list()
current_50 = list()
current_75 = list()
steps_plot = list()
rewards_plot = list()
for step in plot_dict.keys():
    if len(current_step) < 10:
        current_25.append(np.percentile(np.array(plot_dict[step]),25))
        current_50.append(np.percentile(np.array(plot_dict[step]),50))
        current_75.append(np.percentile(np.array(plot_dict[step]),75))
        current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
    else:
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

algo = "Active"
task_label = "T10"
plot_performance(steps_plot,prc_25,prc_50,prc_75,title = "Reward v Time Step",algo = algo)
plot_this(steps_plot,rewards_plot,title = "Avg Rewards v Time Step",algo = algo)
while(True):
    #ask for max x limit
    x_limit = int(int(float(eval(input("What is X max limit?"))))/ test_freq)
    #graph it
    plot_performance(steps_plot[0:x_limit],prc_25[0:x_limit],prc_50[0:x_limit],prc_75[0:x_limit],"Rewards vs. Time Step",algo+' '+task_label+' task')
    #repeat
