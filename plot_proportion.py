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
        proportion = []
        fedavg_no=[]
        fedavg = []
        krum = []

        for row in reader:
            # 4.将字符串转换为整型数据
            print("row",row)
            ## attack success rate
            # proportion.append(float(row[0]))
            # fedavg_no.append(float(row[1]))
            # fedavg.append(float(row[2]))
            # krum.append(float(row[3]))

            ## test error rate
            # proportion.append(row[0])
            fedavg_no.append(1-float(row[0]))
            fedavg.append(1-float(row[1]))
            krum.append(1-float(row[2]))

        # print(main)
        # print(backdoor)

    # main_list.append(main)
    # backdoor_list.append(backdoor)
    # print("=========",num)

    aaaa = "proportion_xmam-attack"
    pdf = PdfPages('plot_pdf/{}.pdf'.format(aaaa))
    ax = fig.add_subplot(1, totalgraph, num)
    # ax.plot(main, marker="o", markerfacecolor="blue", markersize=5, markevery=20)

    ax.plot(fedavg_no, marker='^' )
    ax.plot(fedavg, marker='s')
    ax.plot(krum, marker='o')
    plt.xticks(
        [0,1,2,3,4,5,6,7,8,9,10],
        # [r'$2^{-11}$',r'$2^{-12}$',r'$2^{-13}$',r'$2^{-14}$',r'$2^{-15}$',r'$2^{-16}$',r'$2^{-17}$',r'$2^{-18}$',r'$2^{-19}$',r'$2^{-20}$'],
        [0,5,10,15,20,25,30,35,40,45,50],
        )

    # ax.plot(multi_krum, linestyle='--')
    # ax.plot(ndc, linestyle='--')
    # ax.plot(rfa, linestyle='--')
    # ax.plot(rsa, linestyle='--')
    # ax.plot(xmam)
    #ax.plot(reverse_loss, label="distance Loss based weights")
    plt.xlabel("Fraction of malicious clients (%)", fontsize=16)
    # plt.xlabel("$\lambda$ in adaptive attacks")
    # plt.ylabel("Euclidean distance")
    plt.ylabel("Attack success rate", fontsize=16)
    plt.tick_params(labelsize=15)
    # plt.ylim(-0.05, 1.05)
    # plt.tick_params(labelsize=17)
    # plt.legend(labels=["malicious update", "malicious SLOU"])
    plt.legend(labels=["FedAvg*", "XMAM", "FedAvg"], fontsize=14)
    # plt.legend(labels=["Trigger", "Semantic", "Edge-case"], fontsize=14)
    # plt.legend(labels=["FedAvg", "Krum", "Multi-Krum", "NDC", "RFA", "RSA", "XMAM"])
    # plt.legend(labels=["FedAvg*", "Krum", "Multi-Krum", "NDC", "RFA", "RSA", "XMAM"], fontsize=6, loc='center', bbox_to_anchor=(0.4, 1.2),ncol=7, frameon=False)
    plt.tight_layout()

    # plt.legend(fontsize=15)
    # num += 1

    pdf.savefig()
    # plt.savefig('aaaa.png')
    plt.close()
    pdf.close()
