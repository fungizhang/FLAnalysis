import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.backends.backend_pdf import PdfPages
import os
# plt.style.use('classic')


fig = plt.figure(figsize=(5,3.5))
totalgraph = 1
num = 1
main_list = []
backdoor_list = []

# for defend_method in defend_methods:
for i in range(1):
    filename="/home/user/zfj/noniidAnalysis/tmp.csv"
    # filename="/home/user/zfj/backdoorTAIL/language-tasks-fl/out/sent140-greek-director-backdoor-2-10-190-lr-adapt/stats_{}_1-10.csv".format(defend_method)
    with open(filename,'r')as file:
        #1.创建阅读器对象
        reader=csv.reader(file)
        #2.读取文件头信息
        header_row=next(reader)
        for index, column_header in enumerate(header_row):
            print(index, column_header)

        # 3.保存最高气温数据
        fedavg_no=[]
        fedavg = []
        krum = []
        multi_krum = []
        ndc = []
        rfa = []
        rsa = []
        xmam = []
        xmam1 = []
        xmam2 = []
        for row in reader:
            # 4.将字符串转换为整型数据
            print("row",row)
            ### attack success rate
            fedavg_no.append(float(row[0]))
            fedavg.append(float(row[1]))
            krum.append(float(row[2]))
            multi_krum.append(float(row[3]))
            ndc.append(float(row[4]))
            rfa.append(float(row[5]))
            rsa.append(float(row[6]))
            xmam.append(float(row[7]))
            xmam1.append(float(row[8]))
            xmam2.append(float(row[9]))


            ### testing error rate
            # fedavg_no.append(1-float(row[0]))
            # fedavg.append(1-float(row[1]))
            # krum.append(1-float(row[2]))
            # multi_krum.append(1-float(row[3]))
            # ndc.append(1-float(row[4]))
            # rfa.append(1-float(row[5]))
            # rsa.append(1-float(row[6]))
            # xmam.append(1-float(row[7]))

        # print(main)
        # print(backdoor)

    # main_list.append(main)
    # backdoor_list.append(backdoor)
    # print("=========",num)

    aaaa = "fedavg_analysis"
    pdf = PdfPages('plot_pdf/{}.pdf'.format(aaaa))
    ax = fig.add_subplot(1, totalgraph, num)
    # ax.plot(main, marker="o", markerfacecolor="blue", markersize=5, markevery=20)
    ax.plot(fedavg_no)
    ax.plot(fedavg)
    ax.plot(krum)
    ax.plot(multi_krum)
    ax.plot(ndc)
    ax.plot(rfa)
    ax.plot(rsa)
    ax.plot(xmam)
    ax.plot(xmam1)
    ax.plot(xmam2)
    #ax.plot(reverse_loss, label="distance Loss based weights")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Test accuracy", fontsize=16)
    # plt.ylabel("Testing error rate")
    # plt.ylim(-0.05, 1.05)
    plt.tick_params(labelsize=15)
    plt.legend(labels=["class:0", "class:1", "class:2", "class:3", "class:4", "class:5", "class:6", "class:7", "class:8", "class:9"], loc='center left')
    # plt.legend(labels=["FedAvg", "Krum", "Multi-Krum", "NDC", "RFA", "RSA", "XMAM"])
    # plt.legend(labels=["FedAvg*", "Krum", "Multi-Krum", "NDC", "RFA", "RSA", "XMAM"], fontsize=6, loc='center', bbox_to_anchor=(0.4, 1.2),ncol=7, frameon=False)
    plt.tight_layout()
    # plt.legend(fontsize=15)
    # num += 1

    pdf.savefig()
    plt.close()
    pdf.close()
