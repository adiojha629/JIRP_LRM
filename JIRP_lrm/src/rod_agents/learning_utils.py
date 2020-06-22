import random, time, os

def save_results(rewards, scores, rm_info, game_type, alg, rl, seed):
    folder = '../results/%s'%game_type
    if not os.path.exists(folder): os.makedirs(folder)
    root_file = '%s/%s-%s-%d'%(folder,alg,rl,seed)

    if rewards is not None:
        # saving the training rewards
        path_file = root_file + "_reward.txt"
        f = open(path_file, 'w')
        for e,r in rewards:
            # Training step \t reward
            f.write("%d\t%0.1f\n"%(e,r))
        f.close()
    
    if scores is not None:
        # saving the scores of the learned reward machines
        path_file = root_file + "_scores.txt"
        f = open(path_file, 'w')
        for step,total,examples,n_traces,perfect,current in scores:
            # Training step \t number of traces \t trace total \t training examples \t score perfect rm \t score found rm
            f.write("%d\t%d\t%d\t%d\t%0.2f\t%0.2f\n"%(step,total,examples,n_traces,perfect,current))
        f.close()
    
    if rm_info is not None:
        # saving the training rewards
        path_file = root_file + "_rm.txt"
        f = open(path_file, 'w')
        for line in rm_info:
            # Training step \t reward
            f.write(line + "\n")
        f.close()
