f = open('tuning.txt', 'r')
lines = f.readline()
pn = []
mean_acc = []
std = []
acc = []
while lines:
    line = line.split()
    pn.append(line[0][3:])
    mean_acc.append(line[1][5:11])
    std.append(line[2][4:11])
    acc.append(line[3][4:])
print('pn:{}'.format(pn))