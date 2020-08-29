import os
import pickle
import numpy as np
import csv
import matplotlib.ticker as plticker
#from run_lrm import print_results
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
def plot_performance(steps,p25,p50,p75,title,algo): #this is the function I need to replicate
    font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 12}

    matplotlib.rc('font', **font)
    axis_font_size = 20
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
    #loc = plticker.MultipleLocator(base=.1) # this locator puts ticks at regular intervals
    #ax.yaxis.set_major_locator(loc)
    plt.show()

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

import matplotlib.pyplot as plt
import matplotlib
if __name__ == "__main__":
    debug = "debug_get_files"
    if(debug == "results to plot_dict"):
        task = input("What task (ie t9 t7) is this for?")
        world = input("What world (office craft etc) ?")
        plot_dict ={}
        #for loop here
        print("Putting Data into Plot_dict")
        for trial in range(10):
            file_name = "../results/LRM/lrm-qrm/"+world+"/task_"+task+"/trail_"+str(trial)+"/lrm-lrm-qrm-0_rewards_over_time.txt"
            file = open(file_name)
            lines = file.readlines()
            file.close()
            lines = lines[1:]
            lines = [line.replace("\n","").replace("\t","|") for line in lines]
            reward_list = [(int(float(line[:line.find("|")])),int(float(line[line.find("|")+1:]))) for line in lines] #get just the rewards
            '''Reward List is a list of tuples
            tuple[0] = test step
            tuple[1] = reward for that test
            8/9/2020
            '''
            print("rewards for trail "+str(trial)+" obtained")
            #at this point reward list has rewards per step
            print("updateing plot_dict")
            for step,reward in reward_list:
                if step in plot_dict.keys():
                    plot_dict[step].append(reward)
                else:
                    plot_dict[step] = [reward]
            #now plot_dict is updated
            print("plot_dict updated for trail #" + str(trial))
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
        plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards vs. Time Step",'LRM-qrm bacbac task')
        plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'LRM-qrm LRM-qrm bacbac task')
        print("saving prcs")
        folder = '../results/LRM/lrm-qrm/'+world+'task_'+task
        if not os.path.exists(folder): os.makedirs(folder)
        for text,value in zip(["prc25","prc50","prc75"],[prc_25,prc_50,prc_75]):
            path_file = folder + "/"+text+".txt"
            if not os.path.exists(path_file):
                f = open(path_file, 'wb')
                pickle.dump(value,f)
                f.close()
                print("\n"+text+" uploaded")
            else:
                print(path_file+' already exists')
        print("Done saving\nCheck out files at this location:\n"+folder)

        print("Saving CSV files for reward plot and average rewards")
        filename = folder+"/plot_data/"
        if not os.path.exists(filename): os.makedirs(filename)
        filename = folder+"/plot_data/LRM_"+world+"_"+task+".csv"
        with open(filename, 'w') as f:#copied from Active-15. 8.9.2020
            wr = csv.writer(f)
            wr.writerows(list(plot_dict.values()))
        filename = folder+"/plot_data/LRM_avgreward"+task+".txt"
        with open(filename, 'w') as f:
            f.write("%s\n" % str(sum(rewards_plot) / len(rewards_plot)))
            for item in rewards_plot:
                f.write("%s\n" % item)
        print("You can find csv files at:\n"+folder+"/plot_data/")
    elif debug == "debug_get_files":
        #use this code to debug how you get data from the files and format it into a plot dict
        task = input("What task (ie t9 t7) is this for?")
        world = input("What world (office craft etc) ?")
        print("Debugging file data extraction")
        file1 = "../results/LRM/lrm-qrm/"+world+"/task_"+task+"/trail_"+str(3)+"/lrm-lrm-qrm-0_rewards_over_time.txt"
        file = open(file1)
        lines = file.readlines()
        file.close()
        lines = lines[1:]
        lines = [line.replace("\n","").replace("\t","|") for line in lines]
        #print(lines[0])
        reward_list = [(int(float(line[:line.find("|")])),int(float(line[line.find("|")+1:]))) for line in lines] #get just the rewards
        '''Reward List is a list of tuples
        tuple[0] = test step
        tuple[1] = reward for that test
        8/9/2020
        '''
        print(reward_list[0:10])
    elif debug == 3:
        folder = '../results/jul_30_LRM_3symoffice_debug'
        print("Getting prc's from folders")
        file1 = open(folder + "/prc25.txt",'rb')
        prc_25 = pickle.load(file1)
        file1.close()
        file2 =open(folder + "/prc50.txt",'rb')
        prc_50 = pickle.load(file2)
        file2.close()
        file3 = open(folder + "/prc75.txt",'rb')
        prc_75 = pickle.load(file3)
        file3.close()
        steps_plot = range(len(prc_25))
        #print("Plotting prc's")
        #plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards (percentiles) vs. Time Step",'LRM-qrm 3 symbol OfficeWorld')
        '''
        for value in [prc_25,prc_50,prc_75]:
            print(np.max(value))
            print(np.min(value))
            print(np.std(value))
            print(np.mean(value))
            print("\n")
        print(np.std(prc_50)*2)'''
        #value_history = []
        #counter = [1 for value in prc_50 if value == 0.05]
        #counter = sum(counter)
        #print(counter)
        counter = 0
        gap_list = []
        #prc_50 = [0,0,0,0,0,0,0,0.05,0.05,0,0,0,0.05,0,0,0] a test case; gap_list should be [7,0,3]
        for value in prc_50:
            if value == 0.05:
                gap_list.append(counter)
                counter = 0
            else:
                counter+=1
        print("gap list is \n"+str(gap_list))
        print("Gap list stats are:\n")
        print("Max is %d"%np.max(gap_list))
        print("Min is %d"%np.min(gap_list))
        print("Standard Deviation is %d"%np.std(gap_list))
        print("Average is %d"%np.mean(gap_list))
        print("Length of Gap list is %d"%len(gap_list))
        print("\n")
        plt.hist(gap_list[0:200])
        plt.show()
        plt.hist(gap_list[200:400])
        plt.show()
        '''
        value_history = []
        for value in gap_list:
            if not(value in value_history):
                value_history.append(value)
        print(value_history)'''
    else:
        test_freq = 1000
        testing_step = 0
        is_test_values = [1,0,0,1]
        testing_reward_values = [0,0,1,1]
        is_test_values = testing_reward_values
        plot_dict = dict()
        for is_test,testing_reward in zip(is_test_values,testing_reward_values):
            testing_step += test_freq
            if testing_step in plot_dict.keys():
                plot_dict[testing_step].append(testing_reward)
            else:
                plot_dict[testing_step] = [testing_reward]
        rewards_plot = list()
        prc_25 = list()
        prc_50 = list()
        prc_75 = list()
        # Buffers for plots
        current_step = list()
        current_25 = list()
        current_50 = list()
        current_75 = list()
        steps_plot = list()
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
        plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards (percentiles) vs. Time Step",'LRM-qrm 3 symbol OfficeWorld')
        plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'LRM-qrm LRM-qrm 3 symbol OfficeWorld')
"""Rand_Action_Run1 Code"""
if(False):
    #parameters for testing agent throughout training: 8/7/20
    test_frq = 1000 #how often we test agent 8/7/20
    test_epi_length = 1000 #how long we test agent 8/7/20
    plot_dict = dict() #used for plotting rewards over time from tests 8/7/20
    test_step = 0 #used by plot_dict (variable above) 8/7/20
    test_env = Game(env_params,label="test") #environment used for testing 8/7/2020
    #parameters to study while testing
    num_of_suc = 0 #how many times does the agent complete an episode in testing 8.7.20
    time_to_suc = [] #track how long it took agent to complete episode 8.7.20
    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < lp.rm_init_steps:
        # running an episode using a random policy
        env.restart("train")
        trace = [(env.get_events(),0.0)]
        for _ in range(lp.episode_horizon):
            # executing a random action
            a = random.choice(actions)
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            reward_total += reward
            reward_list.append(reward)
            trace.append((o2_events,reward))
            step += 1
            '''Code for testing agent performance'''
            if step % test_frq == 0: #We test the model if a test_frq number of time steps have passed
                #below we reset the environment, reward, and done variables
                test_env.restart("test")
                test_reward = 0
                test_done = False
                for test_trail in range(test_epi_length): #this is the testing loop
                    if not(test_done): #if an episode isn't complete
                        act = random.choice(actions) #choose a random action
                        test_reward,test_done = test_env.execute_action(act) #execute that action
                    else:#if an episode was completed:
                        num_of_suc += 1 #increment number of successes but agent to complete an episode 8.7.20
                        time_to_suc.append(test_trail) #record how long it took to complete an episode 8.7.20
                        break #break out of for loop
                test_step += test_frq #increment test_step by the test_frq
                if test_step in plot_dict.keys(): #this if/else updates plot_dict
                    plot_dict[test_step].append(test_reward) #this code was copied from final-i-caps
                else:
                    plot_dict[test_step] = [test_reward]
            # Testing (LRM testing, not what we need to compare with JIRP)
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
            # checking if the episode finishes
            if done or lp.rm_init_steps <= step:
                if done:rm.add_terminal_observations(o2_events)
                break
        # adding this trace to the set of traces that we use to learn the rm
        rm.add_trace(trace)
    print("Done with random action")
    assert len(plot_dict) == 200 #we check that the agent was tested 2e5/test_frq = 2e5/1e3 = 2e2 = 200 times
    assert len(time_to_suc) == num_of_suc #the number of completed episodes should match how many 'times to completed episodes' we recorded
    #first we graph percentile and raw rewards
    #the below code creates the percentiles for graphing
    rewards_plot = list()
    prc_25 = list()
    prc_50 = list()
    prc_75 = list()
    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()
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

    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards (percentiles) vs. Time Step",'LRM-qrm 3 symbol OfficeWorld Random Action')
    plot_this(steps_plot,rewards_plot,"Raw Reward vs. Time Step",'LRM-qrm 3 symbol OfficeWorld Random Action')
    plt.plot(time_to_suc,label = "Number of steps to complete episode with random action")
    plt.title("Steps to complete episode\nNumber of Successes = "+str(num_of_suc)+"/200 tests\nAverage number of steps = "+str(np.average(time_to_suc)))
    plt.show()
    adi = input("Continue?")
