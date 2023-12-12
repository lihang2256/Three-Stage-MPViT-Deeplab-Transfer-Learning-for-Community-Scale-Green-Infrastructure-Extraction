# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

log0_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/loss/log-mp.txt"
log1_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/loss/log1.txt"
log2_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/loss/log2.txt"
log3_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/loss/log3.txt"
log4_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/loss/log4.txt"

log_paths = [log0_path, log1_path, log2_path, log3_path, log4_path]
log_name = ['MPViT-D', 'MPViT-D-T', 'MPViT-D-FM', 'MPViT-D-FD', 'MPViT-D-FA']
save_name = 'cccc'

# 测试输出正确率前的关键字
split_loss = "Loss="
# 根据solver.prototxt文件中的test_interval调整
loss_interval = 10
# 根据solver.prototxt文件中的test_interval调整
acc_interval = 100

plt_point = []
p = 0

plt.figure(figsize=(8, 6))
for log_path in log_paths:
    a_n = 0
    acc_x = []
    acc_y = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            # if len(line.split(split_loss)) == 2:
            #     if a_n % 100 == 0:
            #         acc_y.append(float(line.split(split_acc)[1]))
            #         acc_x.append(a_n)
            #     a_n += loss_interval
            if len(line.split(split_loss)) == 2:
                if a_n % 100 == 0:
                    tmp = line.split(split_loss)[1]
                    if a_n >=1500 and float(tmp.split(" ")[0]) >= 0.1:
                        continue
                    else:
                        acc_y.append(float(tmp.split(" ")[0]))
                        acc_x.append(a_n)
                a_n += loss_interval
            if a_n >= 3000:
                break
    plt.plot(acc_x, acc_y, '', label=log_name[p])
    p = p + 1

plt.title('Loss of networks')
plt.legend(loc='upper right')  # 设置label标记的显示位置
plt.xlabel('iter')
plt.ylabel('Loss')

plt.grid()  # 生成网格线
plt.savefig(save_name)
# plt.show()
