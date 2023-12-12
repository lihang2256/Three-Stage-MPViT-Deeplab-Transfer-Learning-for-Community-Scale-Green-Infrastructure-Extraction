# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

'''
在该代码文件所在目录下执行python plt_test.py log_path save_name
log_path 为你的caffe训练日志文件的路径，save_name为你需要保存的图片名字
split_acc 、split_loss、test_interval和display的值需要根据自己的实际情况来进行设置
'''

log_path = "D:/BaiduNetdiskDownload/code/paper/【paper baseline】/【】csgi/【】mpvit-csgi-land3-tl-冻de-2/log.txt"
TYPE = ["Overall Acc", "Mean Acc", "FreqW Acc", "Mean IoU"]
CLASS = ["class1", "class2", "class3", "class4", "class5"]


for acc_type in TYPE:

    # acc_type = "Overall Acc"
    # acc_type = "Mean Acc"
    # acc_type = "FreqW Acc"
    # acc_type = "Mean IoU"
    save_name = '' + acc_type

    # 测试输出正确率前的关键字
    split_acc = acc_type + ": "
    # 测试输出loss前的关键字
    split_loss = "Loss="
    # 根据solver.prototxt文件中的test_interval调整
    loss_interval = 10
    acc_interval = 100

    l_n = 0
    a_n = 0
    loss_x = []
    loss_y = []
    acc_x = []
    acc_y = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if len(line.split(split_acc)) == 2:
                acc_y.append(float(line.split(split_acc)[1]))
                acc_x.append(a_n)
                a_n += acc_interval
            if len(line.split(split_loss)) == 2:
                tmp = line.split(split_loss)[1]
                loss_y.append(float(tmp.split(" ")[0]))
                loss_x.append(l_n)
                l_n += loss_interval

    acc_y = np.array(acc_y)  # 后面np.where找最大值的位置需要用到
    loss_y = np.array(loss_y)  # 后面np.where找最小值的位置需要用到

    def draw_point(x, y):
        plt.plot(x, y, "ro")
        # s为需要显示的字符串，xy为箭头需要指向的位置，xytext为文本的位置（个人理解是xy的相对位置）
        plt.annotate(s="({},{})".format(x, round(y, 4)), xy=(x, y), xycoords="data", xytext=(+10, +30),
                     textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.figure(figsize=(8, 6))
    plt.plot(loss_x, loss_y, '', label="Loss")
    plt.plot(acc_x, acc_y, '', label=acc_type)
    plt.title('Loss vs ' + acc_type)
    plt.legend(loc='upper right')  # 设置label标记的显示位置
    plt.xlabel('iter')
    plt.ylabel('')
    pos_a = np.where(acc_y == max(acc_y))[0] * acc_interval
    pos_l = np.where(loss_y == min(loss_y))[0] * loss_interval
    # 可能存在多个最大或最小值
    for x in pos_a:
        draw_point(x, max(acc_y))
    for x in pos_l:
        draw_point(x, min(loss_y))

    plt.grid()  # 生成网格线
    plt.savefig(save_name)
    # plt.show()
