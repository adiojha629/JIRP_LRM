#get file
file = open("../results/LRM/lrm-qrm/trail_0/officeworld/lrm-lrm-qrm-0_rewards_over_time.txt")
for line in file.readlines():
    
    if "18" in line:
        print(line)
        break
"""
for i in range(10):
    file = open("../results/LRM/lrm-qrm/trail_"+str(i)+"/officeworld/lrm-lrm-qrm-0_rewards_over_time.txt")
    isone = False
    istwo = False
    for line in file.readlines():
        num = int(line.replace(" ","").replace("\n","")[-1])
        if(num == 1):
            isone = True
            #print(num)
        if(num == 8):
            istwo = True
            #print(num)
    file.close()
    #print(isone)
    print("for trail " + str(i)+ " Is their an 8: " + str(istwo))
"""
"""
list = []
for i in range(10):
    for _ in range(10):
        list.append(i)
list_new = []
last_num = 0
for num in list:
    list_new.append(num-last_num)
    last_num = num
print(list_new)
"""
