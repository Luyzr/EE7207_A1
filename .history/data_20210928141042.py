f = open('tuning.txt', 'r')
lines = f.readlines()
print(lines)
pn = []
mean_acc = []
std = []
acc = []
for line in lines:
    print(line)

    pn.append(lines[0][3:])
    mean_acc.append(lines[1][5:11])
    std.append(lines[2][4:11])
    acc.append(lines[3][4:])
print('pn:{}'.format(pn))