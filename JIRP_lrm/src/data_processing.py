#get file
file = open("../results/LRM/lrm-qrm/trail_5/officeworld/lrm-lrm-qrm-0_rewards_over_time.txt")
isone = False
for line in file.readlines():
    num = int(line.replace(" ","").replace("\n","")[-1])
    if(num == 1):
        isone = True
        print(num)
file.close()
print(isone)
