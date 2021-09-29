f = open('tuning.txt', 'r')
lines = f.readline()
pn = []
mean_acc = []
std = []
acc = []
while lines:
    lines = lines.split()
    pn.append(lines[0][3:])
    mean_acc.append(lines[1][5:11])
    std.append(lins[2][4:11])
    acc.append(lines[3][4:])
print('pn:{}'.format(pn))