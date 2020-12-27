q_history = [-1] * 10
print(q_history)

def find_first(num_list):
    for index in range(len(num_list)):
        if num_list[index] == -1:
            return index
    return -1

def add_to_history(state):
    #State is the State of the MDP, as in a value between 0 and 107
    q_history.insert(0,state) #insert the new state to the beginning of the history
    q_history.pop(-1)#remove the last state
    return q_history


for i in range(10):
    add_to_history(i)
print(q_history)
print(add_to_history(100))
"""
for i in range(10):
    q_history[find_first(q_history)] = 3
print(q_history)
print(find_first(q_history))"""
