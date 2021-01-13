## Created Dec. 5th 2020
## Author: Adi Ojha
##Purpose:
## To graph data for publication of Active Learning Paper.

## Section 0: Libraries used
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
import matplotlib.font_manager as font_manager
font_manager._rebuild() ##for fixing issues with fonts Dec 5th 20:24
## Section 1: Functions:

##Function Name: plot_performance
##inputs: steps list, 25,50,75 percentiles list, title and algorithm
##outputs: none
## Prints a graph of rewards vs. time with shading between the 75th,50th and 25th percentiles
def plot_performance(steps,p25,p50,p75,title,algo,total_time): #this is the function I need to replicate
    axis_font_size = 20
    fig, ax = plt.subplots() #Next three lines set the height and width of figure
    fig.set_figheight(6)
    fig.set_figwidth(8)
    ax.plot(steps, p25, alpha=0) #make 25 percentile transparent
    #select color
    color = None
    if(algo == "LRM"):
        color = 'red'
    elif(algo == 'AFRAI-RL'):
        color = 'green'
    elif(algo == 'JIRP'):
        color = 'blue'
    ax.plot(steps, p50, color=color,label = algo) #put the 50th percentile in black
    ax.plot(steps, p75, alpha=0)
    ax.grid()
    ax.legend(fontsize = 20,loc='center right')
    plt.fill_between(steps, p50, p25, color='dark'+color, alpha=0.25)#fill in area between p50 and p25
    plt.fill_between(steps, p50, p75, color='dark'+color, alpha=0.25)#fill in area between p50 p75
    #plt.title(title)
    plt.xlabel("Number of Training Steps",fontsize = axis_font_size)
    plt.ylabel("Reward",fontsize = axis_font_size)
    if(total_time == 6000000):
        plt.xticks([0,2000000,4000000,6000000],["0","2000000","4000000","6000000"]) ##get axis labels correct Dec. 5th 20:21
    elif(total_time == 1000000):
        plt.xticks([0,250000,500000,750000,1000000],["0","250000","500000","750000","1000000"])
        plt.xlim(0,int(1e6))
    elif(total_time == 400000):
        plt.xticks([0,100000,200000,300000,400000],["0","100000","200000","300000","400000"])
        plt.xlim(0,int(4e5))
    elif(total_time == 250000):
        plt.xticks([0,50000,100000,150000,200000,250000],["0","10000","100000","150000","200000","250000"])
        plt.xlim(0,int(25e4))
    else: #assume 2e6
        plt.xticks([0,500000,1000000,1500000,2000000],["0","500000","1000000","1500000","2000000"])
    plt.tick_params(axis = 'both', which = 'major',labelsize = 20)
    #loc = plticker.MultipleLocator(base=.1) # this locator puts ticks at regular intervals
    #ax.yaxis.set_major_locator(loc)
    plt.ylim(-0.05,1.05)
    plt.show()

##Function Name: plot_this
##inputs: x and y (a1,a2) title and algorithm
##outputs: none
## plots two variables a2 vs a1, and puts a title and algoithm
def plot_this(a1,a2,title,algo):
    axis_font_size = 20
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

##Function Name: Cal_Percentiles
##inputs: plot_dict (type dictionary)
##outputs: steps_plot, 25th,50th,75th percentile and rewards plot (list of numbers)
##Takes raw data and extracts the percentiles information
def Cal_Percentiles(plot_dict):
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
    return steps_plot,prc_25,prc_50,prc_75,rewards_plot


##Function Name: plot_active
##inputs: task: the task whose performance one wishes to view
##outputs: 0
##Plots the graphs for the Active Learning Algorithm
def plot_active(task):
    ##Get data
    file = None #set below
    task_label = None
    test_freq = None
    total_time = None
    if(task == 't6'):
        file = "../results/active/Sword_Shield/craftworldt6aqrm.csv"
    elif(task == 't10' or task.lower() == 'cbabca'):
        file = "../results/active/cbabca_office&craft/officeworld10aqrm_800_6e6.csv" ##file name
        task_label = 'cbabca'
        test_freq = 800
        total_time = int(6e6)
    elif(task == 't9' or task.lower() == 'bcabca'):
        file = "../results/active/office_bcabca/officeworld9aqrm.csv"
        task_label = 'bcabca'
        test_freq = 800
        total_time = int(2e6)
    elif(task == 't7' or task.lower() == 'abac'):
        file = "../results/active/office_abac/officeworld7aqrm.csv"
        task_label = 'abac'
        test_freq = 200
        total_time = int(1e6)
    elif(task == 't11' or task.lower() == 'spear'):
        file = "../results/active/craft_spear/craftworld11aqrm.csv"
        task_label = 'spear'
        test_freq = 400
        total_time = int(25e4)
    elif(task == 't8' or task.lower() == 'hammer'):
        file = "../results/active/craft_hammer/craftworldt8aqrm.csv"
        task_label = 'hammer'
        test_freq = 400
        total_time = int(4e5)
    plot_dict = {}
    counter = 0
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        for step in reader:
            if(counter <= total_time): #in case trail was 4e6, but we only want the first 2e6 steps
                values = [float(value) for value in step]
                counter+=test_freq
                plot_dict[counter] = values
    ##Calculate Percentiles
    steps_plot,prc_25,prc_50,prc_75,rewards_plot = Cal_Percentiles(plot_dict)
    ##Plot data
    print("now plotting AFRAI-RL "+str(task_label))
    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards vs. Time Step",'AFRAI-RL',total_time=total_time)
    #plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'Active'+task_label+' task')
    return 0
##Function Name: plot_JIRP
##inputs: task: the task whose performance one wishes to view
##outputs: 0
##Plots the graphs for the JIRP  Algorithm
def plot_JIRP(task):
    ##Get data
    test_freq = None #set below
    file = None
    task_label = None
    total_time = None
    if(task == 't7' or task.lower() == "abac"):
        file = "../results/JIRP/Office_abac/officeworld7jirpsat.csv"
        test_freq = 200
        task_label = 'abac'
        total_time = int(1e6)
    elif(task == 't9' or task.lower() == "bcabca"):
        file = "../results/JIRP/office_bacbac/officeworld9jirpsat.csv" ##file name
        test_freq = 800
        task_label = 'bcabca'
        total_time = int(2e6)
    elif(task == 't10' or task.lower() == "cbabca"):
        file = "../results/JIRP/office_cbabca/officeworld10jirpsat_800_6e6.csv"
        test_freq = 800
        task_label = 'cbabca'
        total_time = int(6e6)
    elif(task == 't11' or task.lower() == "spear"):
        file = "../results/JIRP/craft_spear/craftworld11jirpsat.csv"
        test_freq = 400
        task_label = 'spear'
        total_time = int(25e4)
    elif(task == 't8' or task.lower() == "hammer"):
        file = "../results/JIRP/craft_hammer/craftworld8jirpsat.csv"
        test_freq = 400
        task_label = 'hammer'
        total_time = int(4e5)
    plot_dict = {}
    counter = 0
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        for step in reader:
            values = [float(value) for value in step]
            counter+=test_freq
            plot_dict[counter] = values
    ##Calculate Percentiles
    steps_plot,prc_25,prc_50,prc_75,rewards_plot = Cal_Percentiles(plot_dict)
    ##Plot data
    print("now plotting: JIRP "+str(task_label))
    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards vs. Time Step",'JIRP',total_time = total_time)
    #plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'Active'+task_label+' task')
    return 0

##Function Name: plot_LRM
##inputs: task: the task whose performance one wishes to view
##outputs: 0
##Plots the graphs for the LRM  Algorithm
def plot_LRM(task):
    ##Get data
    file = None #file name
    test_freq = None #testing frequency
    total_time = None
    task_label = None
    if(task == "t9" or task.lower() == "bcabca"):
        file = "../results/LRM/lrm-qrm/"+"officeworld_active"+"/task_"+"t9"+"/plot_data/LRM_officeworld_active_t9.csv" ##file name
        test_freq = 800
        total_time = int(2e6)
        task_label = "bcabca"
    elif(task == "t7" or task.lower() == "abac"):
        file = "../results/LRM/lrm-qrm/officeworld_active/task_t7/plot_data/LRM_officeworld_active_t7.csv"
        test_freq = 200
        total_time = int(1e6)
        task_label = "abac"
    elif(task == 't10' or task.lower() == "cbabca"):
        file = "../results/LRM/lrm-qrm/"+"officeworld_active"+"/task_"+"10"+"/plot_data/LRM_officeworld_active_10.csv" ##file name
        test_freq = 800
        total_time = int(6e6)
        task_label = "cbabca"
    elif(task == 't11' or task.lower() == "spear"):
        file = "../results/LRM/lrm-qrm/craftworld/task_t11/plot_data/LRM_craftworld_t11.csv"
        test_freq = 400
        task_label = 'spear'
        total_time = int(25e4)
    elif(task == 't8' or task.lower() == "hammer"):
        file = "../results/LRM/lrm-qrm/craftworld/task_t8/plot_data/LRM_craftworld_t8.csv"
        test_freq = 400
        task_label = 'hammer'
        total_time = int(4e5)
    plot_dict = {}
    counter = 0
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        for step in reader:
            values = [float(value) for value in step]
            counter+=test_freq
            plot_dict[counter] = values
    ##Calculate Percentiles
    steps_plot,prc_25,prc_50,prc_75,rewards_plot = Cal_Percentiles(plot_dict)
    ##Plot data
    print("now plotting LRM "+task_label)

    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards vs. Time Step",'LRM',total_time=total_time)
    #plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'Active'+task_label+' task')
    return 0


#Main
if __name__ == '__main__':
    offie_tasks = ['t7','t9','t10']
    craft_tasks = ['t8','t11']
    for task in craft_tasks:
        plot_LRM(task)
        plot_JIRP(task)
        plot_active(task)
    #for task in tasks:
        #plot_JIRP(task)

    tasks = ['t7','t9','t10']
    for task in tasks:
        plot_JIRP(task)
        #plot_LRM(task)
        #plot_active(task)
    #plot_active('t6')
