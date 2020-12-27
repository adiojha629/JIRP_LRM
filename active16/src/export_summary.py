import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, argparse
import csv
from run1 import get_params_office_world, get_params_traffic_world, get_params_craft_world

def smooth(y, buffer_size):
    current_batch = list()
    y_smooth = list()

    for value in y:
        if len(current_batch)<buffer_size:
            current_batch.append(value)
        else:
            current_batch.pop(0)
            current_batch.append(value)

        y_smooth.append(sum(current_batch)/len(current_batch))

    return y_smooth

def export_results_traffic_world(task_id, algorithm, bsize, lower, upper):
    files = os.listdir("../plotdata/")

    step_unit = get_params_traffic_world('../experiments/traffic/tests/ground_truth.txt')[0].num_steps
    max_step = get_params_traffic_world('../experiments/traffic/tests/ground_truth.txt')[3].total_steps

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        files_of_interest = list()
        for file in files:
            if (("traffic" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row,lower))
                        p50_q.append(np.percentile(row,50))
                        p75_q.append(np.percentile(row,upper))
            elif 'hrl' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row,lower))
                        p50_hrl.append(np.percentile(row,50))
                        p75_hrl.append(np.percentile(row,upper))
            else:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row,lower))
                        p50.append(np.percentile(row,50))
                        p75.append(np.percentile(row,upper))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()


        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

        files_of_interest

    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]

        files_dict = dict()
        for file in files:
            if (("traffic" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

        for task in files_dict:
            for file in files_dict[task]:
                file_str = ("../plotdata/") + file
                if 'qlearning' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)

                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_qdict:
                                p25_qdict[step].append(np.percentile(row, lower))
                                p50_qdict[step].append(np.percentile(row, 50))
                                p75_qdict[step].append(np.percentile(row, upper))
                            else:
                                p25_qdict[step] = [np.percentile(row, lower)]
                                p50_qdict[step] = [np.percentile(row, 50)]
                                p75_qdict[step] = [np.percentile(row, upper)]

                elif 'hrl' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_hrldict:
                                p25_hrldict[step].append(np.percentile(row, lower))
                                p50_hrldict[step].append(np.percentile(row, 50))
                                p75_hrldict[step].append(np.percentile(row, upper))
                            else:
                                p25_hrldict[step] = [np.percentile(row, lower)]
                                p50_hrldict[step] = [np.percentile(row, 50)]
                                p75_hrldict[step] = [np.percentile(row, upper)]


                else:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25dict:
                                p25dict[step].append(np.percentile(row, lower))
                                p50dict[step].append(np.percentile(row, 50))
                                p75dict[step].append(np.percentile(row, upper))
                            else:
                                p25dict[step] = [np.percentile(row, lower)]
                                p50dict[step] = [np.percentile(row, 50)]
                                p75dict[step] = [np.percentile(row, upper)]



        for step in steps:
            p25_q.append(sum(p25_qdict[step])/len(p25_qdict[step]))
            p50_q.append(sum(p50_qdict[step])/len(p50_qdict[step]))
            p75_q.append(sum(p75_qdict[step])/len(p75_qdict[step]))
            p25_hrl.append(sum(p25_hrldict[step])/len(p25_hrldict[step]))
            p50_hrl.append(sum(p50_hrldict[step])/len(p50_hrldict[step]))
            p75_hrl.append(sum(p75_hrldict[step])/len(p75_hrldict[step]))
            p25.append(sum(p25dict[step])/len(p25dict[step]))
            p50.append(sum(p50dict[step])/len(p50dict[step]))
            p75.append(sum(p75dict[step])/len(p75dict[step]))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()


        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()


def export_results_office_world(task_id, algorithm, bsize, lower, upper):
    files = os.listdir("../plotdata/")

    step_unit = get_params_office_world('../experiments/office/tests/ground_truth.txt')[0].num_steps
    max_step = get_params_office_world('../experiments/office/tests/ground_truth.txt')[3].total_steps

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        files_of_interest = list()
        for file in files:
            if (("office" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file and '#' not in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row,lower))
                        p50_q.append(np.percentile(row,50))
                        p75_q.append(np.percentile(row,upper))
            elif 'hrl' in file and '#' not in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row,lower))
                        p50_hrl.append(np.percentile(row,50))
                        p75_hrl.append(np.percentile(row,upper))
            elif 'jirp' in file and '#' not in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_)>1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row,lower))
                        p50.append(np.percentile(row,50))
                        p75.append(np.percentile(row,upper))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)



        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()



        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

        files_of_interest

    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]

        files_dict = dict()
        for file in files:
            if (("office" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

        for task in files_dict:
            for file in files_dict[task]:
                file_str = ("../plotdata/") + file
                if 'qlearning' in file and '#' not in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)

                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_qdict:
                                p25_qdict[step].append(np.percentile(row, lower))
                                p50_qdict[step].append(np.percentile(row, 50))
                                p75_qdict[step].append(np.percentile(row, upper))
                            else:
                                p25_qdict[step] = [np.percentile(row, lower)]
                                p50_qdict[step] = [np.percentile(row, 50)]
                                p75_qdict[step] = [np.percentile(row, upper)]

                elif 'hrl' in file and '#' not in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_hrldict:
                                p25_hrldict[step].append(np.percentile(row, lower))
                                p50_hrldict[step].append(np.percentile(row, 50))
                                p75_hrldict[step].append(np.percentile(row, upper))
                            else:
                                p25_hrldict[step] = [np.percentile(row, lower)]
                                p50_hrldict[step] = [np.percentile(row, 50)]
                                p75_hrldict[step] = [np.percentile(row, upper)]


                else:
                    if '#' in file:
                        continue
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25dict:
                                p25dict[step].append(np.percentile(row, lower))
                                p50dict[step].append(np.percentile(row, 50))
                                p75dict[step].append(np.percentile(row, upper))
                            else:
                                p25dict[step] = [np.percentile(row, lower)]
                                p50dict[step] = [np.percentile(row, 50)]
                                p75dict[step] = [np.percentile(row, upper)]



        for step in steps:
            p25_q.append(sum(p25_qdict[step])/len(p25_qdict[step]))
            p50_q.append(sum(p50_qdict[step])/len(p50_qdict[step]))
            p75_q.append(sum(p75_qdict[step])/len(p75_qdict[step]))
            p25_hrl.append(sum(p25_hrldict[step])/len(p25_hrldict[step]))
            p50_hrl.append(sum(p50_hrldict[step])/len(p50_hrldict[step]))
            p75_hrl.append(sum(p75_hrldict[step])/len(p75_hrldict[step]))
            p25.append(sum(p25dict[step])/len(p25dict[step]))
            p50.append(sum(p50dict[step])/len(p50dict[step]))
            p75.append(sum(p75dict[step])/len(p75dict[step]))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()


        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.3), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

        files_dict


def export_results_craft_world(task_id, algorithm, bsize, lower, upper):
    files = os.listdir("../plotdata/")

    step_unit = get_params_craft_world('../experiments/craft/tests/ground_truth.txt')[0].num_steps
    #max_step = get_params_craft_world('../experiments/craft/tests/ground_truth.txt')[3].total_steps
    max_step = 600000

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        files_of_interest = list()
        for file in files:
            if (("craft" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row,lower))
                        p50_q.append(np.percentile(row,50))
                        p75_q.append(np.percentile(row,upper))
            elif 'hrl' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row,lower))
                        p50_hrl.append(np.percentile(row,50))
                        p75_hrl.append(np.percentile(row,upper))
            else:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row,lower))
                        p50.append(np.percentile(row,50))
                        p75.append(np.percentile(row,upper))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)

        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()


        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()


    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]

        files_dict = dict()
        for file in files:
            if (("craft" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

        for task in files_dict:
            for file in files_dict[task]:
                file_str = ("../plotdata/") + file
                if 'qlearning' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)

                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_qdict:
                                p25_qdict[step].append(np.percentile(row, lower))
                                p50_qdict[step].append(np.percentile(row, 50))
                                p75_qdict[step].append(np.percentile(row, upper))
                            else:
                                p25_qdict[step] = [np.percentile(row, lower)]
                                p50_qdict[step] = [np.percentile(row, 50)]
                                p75_qdict[step] = [np.percentile(row, upper)]

                elif 'hrl' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_hrldict:
                                p25_hrldict[step].append(np.percentile(row, lower))
                                p50_hrldict[step].append(np.percentile(row, 50))
                                p75_hrldict[step].append(np.percentile(row, upper))
                            else:
                                p25_hrldict[step] = [np.percentile(row, lower)]
                                p50_hrldict[step] = [np.percentile(row, 50)]
                                p75_hrldict[step] = [np.percentile(row, upper)]


                else:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25dict:
                                p25dict[step].append(np.percentile(row, lower))
                                p50dict[step].append(np.percentile(row, 50))
                                p75dict[step].append(np.percentile(row, upper))
                            else:
                                p25dict[step] = [np.percentile(row, lower)]
                                p50dict[step] = [np.percentile(row, 50)]
                                p75dict[step] = [np.percentile(row, upper)]



        for step in steps:
            p25_q.append(sum(p25_qdict[step])/len(p25_qdict[step]))
            p50_q.append(sum(p50_qdict[step])/len(p50_qdict[step]))
            p75_q.append(sum(p75_qdict[step])/len(p75_qdict[step]))
            p25_hrl.append(sum(p25_hrldict[step])/len(p25_hrldict[step]))
            p50_hrl.append(sum(p50_hrldict[step])/len(p50_hrldict[step]))
            p75_hrl.append(sum(p75_hrldict[step])/len(p75_hrldict[step]))
            p25.append(sum(p25dict[step])/len(p25dict[step]))
            p50.append(sum(p50dict[step])/len(p50dict[step]))
            p75.append(sum(p75dict[step])/len(p75dict[step]))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm=="jirp" or algorithm=="all":
            p25 = smooth(p25, bsize)
            p50 = smooth(p50, bsize)
            p75 = smooth(p75, bsize)
            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP')
            ax.plot(steps, p75, alpha=0)


        if algorithm=="qlearning" or algorithm=="all":
            p25_q = smooth(p25_q, bsize)
            p50_q = smooth(p50_q, bsize)
            p75_q = smooth(p75_q, bsize)
            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)


        if algorithm=="hrl" or algorithm=="all":
            p25_hrl = smooth(p25_hrl, bsize)
            p50_hrl = smooth(p50_hrl, bsize)
            p75_hrl = smooth(p75_hrl, bsize)
            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

        ax.grid()

        if algorithm=="jirp" or algorithm=="all":
            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP', '', '', 'QAS', '', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

if __name__ == "__main__":


    # EXAMPLE: python3 export_summary.py --world="craft"

    # Getting params
    worlds     = ["office", "craft", "traffic"]

    print("Note: ensure that runs correspond with current parameters for curriculum.total_steps and testing_params.num_steps!")
    print("")

    parser = argparse.ArgumentParser(prog="export_summary", description='After running the experiments, this algorithm computes a summary of the results.')
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which world to solve.')
    parser.add_argument('--algorithm', default='all', type=str,
                        help='This parameter indicated which algorithm to solve. Set to "all" to graph all methods.')
    parser.add_argument('--task', default=1, type=int,
                        help='This parameter indicates which task to display. Set to zero to graph all tasks.')

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")

    
    lower_percentile = 25
    upper_percentile = 75


    # Computing the experiment summary
    world = args.world
    if world == "office":
        export_results_office_world(args.task, args.algorithm, 10, lower_percentile, upper_percentile)
    if world == "craft":
        export_results_craft_world(args.task, args.algorithm, 10, lower_percentile, upper_percentile)
    if world == "traffic":
        export_results_traffic_world(args.task, args.algorithm, 10, lower_percentile, upper_percentile)
