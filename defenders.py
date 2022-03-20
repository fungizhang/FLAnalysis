import torch
import numpy as np
import copy
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import hdbscan
import sys
from PIL import Image
import scipy.misc
import imageio
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import time

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight_diff[index_bias:index_bias + p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorized_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorized_diff).item()
        clipped_weight_diff = vectorized_diff / max(1, weight_diff_norm / self.norm_bound)

        print("The Norm of Weight Difference between received global model and updated client model: {}".format(weight_diff_norm))
        print("The Norm of weight (updated part) after clipping: {}".format(torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None

class RSA(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, global_model, flround, *args, **kwargs):

        for net_index, net in enumerate(client_model):
            whole_aggregator = []
            for p_index, p in enumerate(client_model[0].parameters()):
                params_aggregator = 0.00005 * 0.998 ** flround * torch.sign(list(net.parameters())[p_index].data
                    - list(global_model.parameters())[p_index].data) + list(global_model.parameters())[p_index].data

                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(net.parameters()):
                p.data = whole_aggregator[param_index]

        return None


class WeakDPDefense(Defense):
    """
        deprecated: don't use!
        according to literature, DPDefense should be applied
        to the aggregated model, not invidual models
        """

    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net / max(1, weight_norm / self.norm_bound)
        dp_weight = clipped_weight + torch.randn(
            vectorized_net.size(), device=self.device) * self.stddev

        load_model_weight(client_model, clipped_weight)
        return None


class AddNoise(Defense):
    def __init__(self, stddev, *args, **kwargs):
        self.stddev = stddev

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        gaussian_noise = torch.randn(vectorized_net.size(),
                                     device=self.device) * self.stddev
        dp_weight = vectorized_net + gaussian_noise
        load_model_weight(client_model, dp_weight)
        print("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))

        return None


class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """

    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, global_model_pre, num_dps, g_user_indices, device, *args, **kwargs):

        ######################################################################## separate model to get updated part
        whole_aggregator = []
        client_models_copy = copy.deepcopy(client_models)
        for i in range(len(client_models_copy)):
            for p_index, p in enumerate(client_models_copy[i].parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                params_aggregator = params_aggregator + (list(client_models_copy[i].parameters())[p_index].data -
                                                         list(global_model_pre.parameters())[p_index].data)
                # params_aggregator = torch.sign(params_aggregator)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(client_models_copy[i].parameters()):
                p.data = whole_aggregator[param_index]

            whole_aggregator = []

        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j) ** 2))  # let's change this to pytorch version
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            print("===starr===:", i_star)
            print("===scoree===:", scores)
            print("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)),
                                                                                           g_user_indices[scores.index(
                                                                                               min(scores))]))
            aggregated_model = client_models[i]  # slicing which doesn't really matter
            # load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            neo_net_list = [aggregated_model]
            print("Norm of Aggregated Model: {}".format(
                torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, i_star
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score + 2)[:nb_in_score + 2]

            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]

            print("===scores===", scores)
            print("Num data points: {}".format(num_dps))
            print("Num selected data points: {}".format(selected_num_dps))
            print("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
                                                                    [g_user_indices[ti] for ti in topk_ind]))
            # aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0)
            aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq,
                                         axis=0).astype(np.float32)
            aggregated_model=[]
            for i in range(len(topk_ind)):
                aggregated_model.append(client_models[topk_ind[i]])  # slicing which doesn't really matter
            # load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            neo_net_list = aggregated_model
            # print("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, topk_ind

class XMAM(Defense):

    def __init__(self, *args, **kwargs):
        pass

    def abs_avg(self, SLPD):
        SLPD_abssum = 0
        for i in range(len(SLPD)):
            SLPD_abssum += abs(SLPD[i])

        for i in range(len(SLPD)):
            SLPD[i] = abs(SLPD[i]) / SLPD_abssum
        return SLPD

    def label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    #### 2. 获取第k层的特征图
    '''
    args:
    k:定义提取第几层的feature map
    x:图片的tensor
    model_layer：是一个Sequential()特征层
    '''

    def get_k_layer_feature_map(self, model_layer, k, x):
        with torch.no_grad():
            for index, layer in enumerate(model_layer):  # model的第一个Sequential()是有多层，所以遍历
                x = layer(x)  # torch.Size([1, 64, 55, 55])生成了64个通道
                if k == index:
                    return x

    #  可视化特征图
    def show_feature_map(self, feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
        # feature_map[2].shape     out of bounds
        feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])
        feature_map_num = feature_map.shape[0]  # 返回通道数
        row_num = np.ceil(np.sqrt(feature_map_num))  # 8
        plt.figure()
        for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出

            plt.subplot(row_num, row_num, index)
            plt.imshow(feature_map[index - 1].cpu(), cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
            plt.axis('off')
            imageio.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1].cpu())
        plt.show()

    def exec(self, client_models, x_ray_loader, global_model_pre, g_user_indices, device, malicious_ratio, *args, **kwargs):

        # for data, target in x_ray_loader[1]:
        if malicious_ratio == 0:
            for data, target in x_ray_loader:
                x_ray = data[0:1]
                break
        else:
            for data, target in x_ray_loader[1]:
                x_ray = data[0:1]
                break

        x_ray = x_ray.to(device)
        x_ray = torch.ones_like(x_ray)

        # plt.figure(figsize=(1, 1))
        # plt.imshow(np.transpose(x_ray_allone[0], (1, 2, 0)))
        # plt.show()
        # x_ray_gassian =  torch.rand_like(x_ray_allone)  # Gassian noise
        # plt.figure(figsize=(1, 1))
        # plt.imshow(np.transpose(x_ray_gassian[0], (1, 2, 0)))
        # plt.show()

        client_num = len(client_models)

        ################################################################################## feature visualization
        # for id in range(len(client_models)):
        #     model_layer = list(client_models[id].children())
        #     for k in range(len(model_layer)):
        #         print("model {} , layer {}".format("$malicous" if g_user_indices[id] < 0.2*200 else "@benign" , k))
        #         feature_map = self.get_k_layer_feature_map(model_layer, k, x_ray)
        #         print("**************", feature_map)
        #         # self.show_feature_map(feature_map)


        ######################################################################## separate model to get updated part
        whole_aggregator = []
        client_models_copy = copy.deepcopy(client_models)
        for i in range(len(client_models_copy)):
            for p_index, p in enumerate(client_models_copy[i].parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                params_aggregator = params_aggregator + 1*(list(client_models_copy[i].parameters())[p_index].data -
                                                         list(global_model_pre.parameters())[p_index].data)
                # params_aggregator = torch.sign(params_aggregator)
                whole_aggregator.append(params_aggregator)
                # print("{}--{}:{}".format(i, p_index, p))

            for param_index, p in enumerate(client_models_copy[i].parameters()):
                p.data = whole_aggregator[param_index]

            whole_aggregator = []

        # for name, module in client_models[0]._modules.items():
        #     print(name, " : ", module)
        ########################################### generate new x-ray: from global update reverse a or a set of data
        # # calculate the global update, which as the original gradient original_dy_dx
        # original_dy_dx = []
        # client_models_copy = copy.deepcopy(client_models)
        #
        # for p_index, p in enumerate(client_models_copy[0].parameters()):
        #     params_aggregator = torch.zeros(p.size()).to(device)
        #     for i in range(len(client_models_copy)):
        #         params_aggregator += params_aggregator + (list(client_models_copy[i].parameters())[p_index].data -
        #                                                  list(global_model_pre.parameters())[p_index].data)
        #     params_aggregator = 1/len(client_models) * params_aggregator
        #     original_dy_dx.append(params_aggregator)

        # fedavged_model = copy.deepcopy(global_model_pre)
        # whole_aggregator = []
        # client_models_copy = copy.deepcopy(client_models)
        #
        # for p_index, p in enumerate(client_models_copy[0].parameters()):
        #     params_aggregator = torch.zeros(p.size()).to(device)
        #     for i in range(len(client_models_copy)):
        #         params_aggregator += params_aggregator + list(client_models_copy[i].parameters())[p_index].data
        #     params_aggregator = 1 / len(client_models) * params_aggregator
        #     whole_aggregator.append(params_aggregator)
        #
        # for param_index, p in enumerate(fedavged_model.parameters()):
        #         p.data = whole_aggregator[param_index]
        #
        #
        # dummy_data = torch.randn(x_ray.size()).to(device).requires_grad_(True)
        # x_ray_label[0] = 0
        # print("=====",x_ray_label)
        # x_ray_label = self.label_to_onehot(x_ray_label, 10)
        #
        # dummy_label = x_ray_label.to(device).requires_grad_(True)
        #
        # # dummy_label = torch.randn(x_ray_label.size()).to(device).requires_grad_(True)
        # tt = torchvision.transforms.ToPILImage()
        # plt.imshow(tt(dummy_data[0].cpu()))
        # # plt.show()
        # optimizer = torch.optim.LBFGS([dummy_data])
        #
        # def cross_entropy_for_onehot(pred, target):
        #     return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
        # criterion = cross_entropy_for_onehot
        #
        # history = []
        # for iters in range(300):
        #     def closure():
        #         optimizer.zero_grad()
        #
        #         dummy_pred = fedavged_model(dummy_data)
        #         dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        #         # dummy_onehot_label = dummy_label
        #         dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        #         # dummy_dy_dx = torch.autograd.grad(dummy_loss, global_model_pre.parameters(), create_graph=True)
        #         #
        #         # grad_diff = 0
        #         # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        #         #     grad_diff += ((gx - gy) ** 2).sum()
        #         # grad_diff.backward()
        #         dummy_loss.backward()
        #
        #         # return grad_diff
        #         return dummy_loss
        #
        #     # pdb.set_trace()
        #     optimizer.step(closure)
        #     if iters % 10 == 0:
        #         current_loss = closure()
        #         print(iters, "%.4f" % current_loss.item())
        #         history.append(tt(dummy_data[0].cpu()))
        #
        # plt.figure(figsize=(12, 8))
        # for i in range(30):
        #     plt.subplot(3, 10, i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')
        #
        # plt.show()

        # output_2 = fedavged_model(dummy_data)
        # softmax_output_2 = F.softmax(output_2, dim=1)
        # print("===softmax_output_2====", softmax_output_2)
        # print("===dummy_label====", dummy_label)

        ############################necessary###################################### calculate clients' SLPDs currently

        #################################
        # for i in range(len(client_models_copy)):
        #     for p_index, p in enumerate(client_models_copy[i].parameters()):
        #         print("{}==={}:{}".format(i,p_index,p))
        #################################


        # model_layer = list(client_models[0].children())
        # feature_map = self.get_k_layer_feature_map(model_layer, 0, x_ray)
        # print("*******clean_data*******", feature_map)

        # for i in range(len(client_models_copy)):
        #     for p_index, p in enumerate(client_models_copy[i].parameters()):
        #         print("{}=={}:{}".format(i, p_index, p))


        client_SLPDs = []
        Temperature = 1

        for net_index, net in enumerate(client_models_copy):
            SLPD_now = net(x_ray)
            # SLPD_now = net(x_ray_gassian)
            SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
            # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
            SLPD_now = SLPD_now.detach().cpu().numpy()
            SLPD_now = SLPD_now[0]
            # SLPD_now = self.abs_avg(SLPD_now)
            client_SLPDs.append(SLPD_now)

        client_SLPDs = np.array(client_SLPDs)

        # # ########################################################################  delete

        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('clean_data'), index=False)
        # print("Wrote results!!!")

        # for data, target in x_ray_loader[2]:
        #     x_ray = data[0:1]
        #     break
        #
        # x_ray = x_ray.to(device)
        #
        # client_SLPDs = []
        # Temperature = 1
        #
        # # model_layer = list(client_models[0].children())
        # # feature_map = self.get_k_layer_feature_map(model_layer, 1, x_ray)
        # # print("*******backdoor_data*******", feature_map[0][0:10])
        #
        # for net_index, net in enumerate(client_models_copy):
        #     SLPD_now = net(x_ray)
        #     # SLPD_now = net(x_ray_gassian)
        #     # SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
        #     # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
        #     SLPD_now = SLPD_now.detach().cpu().numpy()
        #     SLPD_now = SLPD_now[0]
        #     # SLPD_now = self.abs_avg(SLPD_now)
        #     client_SLPDs.append(SLPD_now)
        #
        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('backdoor_data'), index=False)
        # print("Wrote results!!!")
        #
        #
        #
        #
        #
        #
        # for data, target in x_ray_loader[2]:
        #     x_ray = data[0:1]
        #     break
        #
        # x_ray = x_ray.to(device)
        # x_ray = torch.ones_like(x_ray)
        #
        # client_SLPDs = []
        # Temperature = 1
        #
        # # model_layer = list(client_models[0].children())
        # # feature_map = self.get_k_layer_feature_map(model_layer, 1, x_ray)
        # # print("*******all_one_data*******", feature_map[0][0:10])
        #
        # for net_index, net in enumerate(client_models_copy):
        #     SLPD_now = net(x_ray)
        #     # SLPD_now = net(x_ray_gassian)
        #     # SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
        #     # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
        #     SLPD_now = SLPD_now.detach().cpu().numpy()
        #     SLPD_now = SLPD_now[0]
        #     # SLPD_now = self.abs_avg(SLPD_now)
        #     client_SLPDs.append(SLPD_now)
        #
        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('all_one_data'), index=False)
        # print("Wrote results!!!")
        #
        #
        #
        #
        # for data, target in x_ray_loader[2]:
        #     x_ray = data[0:1]
        #     break
        #
        # x_ray = x_ray.to(device)
        # x_ray = torch.ones_like(x_ray)
        #
        # client_SLPDs = []
        # Temperature = 1
        #
        # # model_layer = list(client_models[0].children())
        # # feature_map = self.get_k_layer_feature_map(model_layer, 1, x_ray)
        # # print("*******all_one_data*******", feature_map[0][0:10])
        #
        # for net_index, net in enumerate(client_models_copy):
        #     SLPD_now = net(x_ray)
        #     # SLPD_now = net(x_ray_gassian)
        #     SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
        #     # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
        #     SLPD_now = SLPD_now.detach().cpu().numpy()
        #     SLPD_now = SLPD_now[0]
        #     # SLPD_now = self.abs_avg(SLPD_now)
        #     client_SLPDs.append(SLPD_now)
        #
        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('all_one_data_softmax'), index=False)
        # print("Wrote results!!!")
        #
        #
        #
        #
        #
        # for data, target in x_ray_loader[2]:
        #     x_ray = data[0:1]
        #     break
        #
        # x_ray = x_ray.to(device)
        # x_ray = torch.rand_like(x_ray)
        #
        # client_SLPDs = []
        # Temperature = 1
        #
        # # model_layer = list(client_models[0].children())
        # # feature_map = self.get_k_layer_feature_map(model_layer, 1, x_ray)
        # # print("*******random_data*******", feature_map[0][0:10])
        #
        # for net_index, net in enumerate(client_models_copy):
        #     SLPD_now = net(x_ray)
        #     # SLPD_now = net(x_ray_gassian)
        #     # SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
        #     # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
        #     SLPD_now = SLPD_now.detach().cpu().numpy()
        #     SLPD_now = SLPD_now[0]
        #     # SLPD_now = self.abs_avg(SLPD_now)
        #     client_SLPDs.append(SLPD_now)
        #
        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('random_data'), index=False)
        # print("Wrote results!!!")
        #
        #
        #
        #
        # for data, target in x_ray_loader[2]:
        #     x_ray = data[0:1]
        #     break
        #
        # x_ray = x_ray.to(device)
        # x_ray = torch.zeros_like(x_ray)
        #
        # client_SLPDs = []
        # Temperature = 1
        #
        # # model_layer = list(client_models[0].children())
        # # feature_map = self.get_k_layer_feature_map(model_layer, 1, x_ray)
        # # print("*******zero_data*******", feature_map[0][0:10])
        #
        #
        # for net_index, net in enumerate(client_models_copy):
        #     SLPD_now = net(x_ray)
        #     # SLPD_now = net(x_ray_gassian)
        #     # SLPD_now = F.softmax(SLPD_now / Temperature, dim=1)
        #     # softmax_output_list = SLPD_now.detach().cpu().numpy().tolist()
        #     SLPD_now = SLPD_now.detach().cpu().numpy()
        #     SLPD_now = SLPD_now[0]
        #     # SLPD_now = self.abs_avg(SLPD_now)
        #     client_SLPDs.append(SLPD_now)
        # # print(client_SLPDs)
        # df = pd.DataFrame({'mali_client0': client_SLPDs[0],
        #                    'mali_client1': client_SLPDs[1],
        #                    'mali_client2': client_SLPDs[2],
        #                    'mali_client3': client_SLPDs[3],
        #                    'client4': client_SLPDs[4],
        #                    'client5': client_SLPDs[5],
        #                    'client6': client_SLPDs[6],
        #                    'client7': client_SLPDs[7],
        #                    'client8': client_SLPDs[8],
        #                    'client9': client_SLPDs[9],
        #                    'client10': client_SLPDs[10],
        #                    'client11': client_SLPDs[11],
        #                    'client12': client_SLPDs[12],
        #                    'client13': client_SLPDs[13],
        #                    'client14': client_SLPDs[14],
        #                    'client15': client_SLPDs[15],
        #                    'client16': client_SLPDs[16],
        #                    'client17': client_SLPDs[17],
        #                    'client18': client_SLPDs[18],
        #                    'client19': client_SLPDs[19],
        #                    })
        # df.to_csv('result/{}.csv'.format('zero_data'), index=False)
        # print("Wrote results!!!")
        # ########################################################################  delete



        # #################################################################################  observe parameter/PCA graph
        # net_vec = []
        #
        #
        # for net_index, net in enumerate(client_models_copy):
        #     vec = vectorize_net(net)
        #     vec = vec.detach().cpu().numpy()
        #     net_vec.append(vec)
        # net_vec = np.array(net_vec)
        ############################################################################ distance contrast
        # net_dis_list = []
        # for j in range(len(net_vec)):
        #     net_dis = 0
        #     for i in range(len(net_vec)):
        #         net_dis += np.linalg.norm(net_vec[j]-net_vec[i])
        #     net_dis_list.append(net_dis)
        # # print("net_dis::::::::",net_dis)
        #
        # df = pd.DataFrame({'1': [net_dis_list[0]],
        #                    '2': [net_dis_list[1]],
        #                    '3': [net_dis_list[2]],
        #                    '4': [net_dis_list[3]],
        #                    '5': [net_dis_list[4]],
        #                    '6': [net_dis_list[5]],
        #                    '7': [net_dis_list[6]],
        #                    '8': [net_dis_list[7]],
        #                    '9': [net_dis_list[8]],
        #                    '10': [net_dis_list[9]],
        #                    '11': [net_dis_list[10]],
        #                    '12': [net_dis_list[11]],
        #                    '13': [net_dis_list[12]],
        #                    '14': [net_dis_list[13]],
        #                    '15': [net_dis_list[14]],
        #                    '16': [net_dis_list[15]],
        #                    '17': [net_dis_list[16]],
        #                    '18': [net_dis_list[17]],
        #                    '19': [net_dis_list[18]],
        #                    '20': [net_dis_list[19]],
        #                    })
        #
        # results_filename = 'net_dis'
        # df.to_csv('result/{}.csv'.format(results_filename), index=False, mode='a')
        # print("Wrote accuracy results to: {}".format(results_filename))
        #
        # slpd_dis_list = []
        # for i in range(len(client_SLPDs)):
        #     slpd_dis = 0
        #     for i in range(len(client_SLPDs)):
        #         slpd_dis += np.linalg.norm(client_SLPDs[j] - client_SLPDs[i])
        #     slpd_dis_list.append(slpd_dis)
        # # print("slpd_dis::::::::", slpd_dis)

        # df = pd.DataFrame({'1': [slpd_dis_list[0]],
        #                    '2': [slpd_dis_list[1]],
        #                    '3': [slpd_dis_list[2]],
        #                    '4': [slpd_dis_list[3]],
        #                    '5': [slpd_dis_list[4]],
        #                    '6': [slpd_dis_list[5]],
        #                    '7': [slpd_dis_list[6]],
        #                    '8': [slpd_dis_list[7]],
        #                    '9': [slpd_dis_list[8]],
        #                    '10': [slpd_dis_list[9]],
        #                    '11': [slpd_dis_list[10]],
        #                    '12': [slpd_dis_list[11]],
        #                    '13': [slpd_dis_list[12]],
        #                    '14': [slpd_dis_list[13]],
        #                    '15': [slpd_dis_list[14]],
        #                    '16': [slpd_dis_list[15]],
        #                    '17': [slpd_dis_list[16]],
        #                    '18': [slpd_dis_list[17]],
        #                    '19': [slpd_dis_list[18]],
        #                    '20': [slpd_dis_list[19]],
        #                    })
        #
        #
        # results_filename = 'slpd_dis'
        # df.to_csv('result/{}.csv'.format(results_filename), index=False, mode='a')
        # print("Wrote accuracy results to: {}".format(results_filename))

        # for i in range(len(net_vec)):
        #     aaa = np.linalg.norm(net_vec[0])
        #     print("===net_vec[{}]===: {}".format(i, net_vec[i]))
        # for i in range(len(client_SLPDs)):
        #     aaa = np.linalg.norm(client_SLPDs[0])
        #     print("===client_SLPDs[{}]===: {}".format(i, client_SLPDs[i]))

        # ########### observer net parameter
        # # for i in range(len(net_vec)):
        # #     print("net {} : {}".format("$malicious" if g_user_indices[i] < 40 else "@benign", net_vec[i][10000:10100]))
        #
        # ############ observe histogram of model parameter
        # # plt.figure()
        # # for i in range(10):
        # #     plt.subplot(2, 5, i+1)
        # #     plt.hist(net_vec[i], bins=10000)
        # #     plt.xlim(-0.00008, 0.00008)
        # #     plt.ylim(0, 8e5)
        # #     plt.xlabel("$malicious" if g_user_indices[i] < 0.3*200 else "@benign")
        # #     print("done!!!")
        # # plt.show()
        #
        ####################################### PCA according to net parameter
        # start = time.time()
        #
        # mali_num = int(100 * 0.2)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        #
        # pca = PCA(n_components=3)
        # pca.fit(net_vec)
        # X_new = pca.transform(net_vec)
        #
        # cluster_labels = clusterer.fit_predict(X_new)
        # print("cluster_labels", cluster_labels)
        #
        # majority = Counter(cluster_labels)
        # print("majority", majority)
        # majority = majority.most_common()[0][0]
        # print("majority", majority)
        #
        # end = time.time()
        # runtime = end - start
        # print("mapping net:", runtime)
        ### 3D
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(X_new[:mali_num, 0], X_new[:mali_num, 1], X_new[:mali_num, 2], c='red', depthshade=False)
        # ax.scatter(X_new[mali_num:, 0], X_new[mali_num:, 1], X_new[mali_num:, 2], c='blue', depthshade=False)
        # plt.xlabel('parameters real')
        # plt.show()
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=cluster_labels, depthshade=False)
        # plt.xlabel('parameters after cluster')
        # plt.show()

        ### 2D
        # plt.figure(figsize=(3.5,3.5))
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # # plt.subplot(2,2,1)
        # plt.scatter(X_new[:mali_num, 0], X_new[:mali_num, 1], c='red', marker='x')
        # plt.scatter(X_new[mali_num:, 0], X_new[mali_num:, 1], c='blue', marker='v')
        # # plt.xlabel('parameters real')
        # plt.tight_layout()
        # # plt.legend(labels=["malicious update", "benign update"])
        # plt.legend(labels=["malicious", "benign"])
        # plt.savefig('png/update_real_{}.png'.format(time.time()))
        # plt.close()
        #
        # # # plt.subplot(2, 2, 2)
        # plt.scatter(X_new[:, 0], X_new[:, 1], c=cluster_labels)
        # # plt.xlabel('parameters after cluster')
        # plt.savefig('png/update_cluster_{}.png'.format(time.time()))
        # plt.close()

        ##################################################### PCA according to net SLPDs
        # start = time.time()
        #
        # mali_num = int(100 * 0.2)
        #
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        #
        # pca = PCA(n_components=3)
        # pca.fit(client_SLPDs)
        # X_new = pca.transform(client_SLPDs)
        #
        # cluster_labels = clusterer.fit_predict(X_new)
        # print("cluster_labels", cluster_labels)
        #
        # majority = Counter(cluster_labels)
        # print("majority", majority)
        # majority = majority.most_common()[0][0]
        # print("majority", majority)
        #
        # end = time.time()

        # runtime = end - start + runtime_mapping
        # print("mapping SLPD:", runtime)

        #### 3D
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(X_new[:mali_num, 0], X_new[:mali_num, 1], X_new[:mali_num, 2], c='red', depthshade=False)
        # ax.scatter(X_new[mali_num:, 0], X_new[mali_num:, 1], X_new[mali_num:, 2], c='blue', depthshade=False)
        # plt.xlabel('SLPDs real')
        # plt.show()
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=2 * cluster_labels, depthshade=False)
        # plt.xlabel('SLPDs after cluster')
        # plt.show()

        # ### 2D
        # plt.subplot(2, 2, 3)
        # plt.figure(figsize=(3.5, 3.5))
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.scatter(X_new[:mali_num, 0], X_new[:mali_num, 1], c='red', marker='x')
        # plt.scatter(X_new[mali_num:, 0], X_new[mali_num:, 1], c='blue', marker='v')
        # plt.tight_layout()
        # # plt.xlabel('SLPDs real')
        # # plt.legend(labels=["malicious update", "benign update"])
        # plt.legend(labels=["malicious", "benign"])
        # plt.savefig('png/SLPD_real_{}.png'.format(time.time()))
        # plt.close()
        #
        # # plt.subplot(2, 2, 4)
        # plt.scatter(X_new[:, 0], X_new[:, 1], c=cluster_labels)
        # # plt.xlabel('SLPDs after cluster')
        # plt.savefig('png/SLPD_cluster_{}.png'.format(time.time()))
        # plt.close()


        ####################################################################################### train one class 's model
        # client_SLPDs = []
        # Temperature = 1

        # for net_index, net in enumerate(client_models_copy):
        #     net_vec = vectorize_net(net).detach().cpu()
        #     net_vec = np.array(net_vec)
        #     net_vec_abs = np.abs(net_vec)
        #     net_vec_sort = np.argsort(net_vec_abs)
        #     net_vec_sort_jiang = net_vec_sort[::-1]
        #     top_k = net_vec_sort_jiang[0:100]
        #
        #     for i in range(len(net_vec)):
        #         if i in top_k:
        #             pass
        #         else:
        #             net_vec[i] = 0
        #
        #     plt.plot([i for i in range(len(net_vec))], net_vec)
        #     plt.savefig("png/net_{}.png".format(net_index))
        #     plt.close()
        #
        # sys.exit()



        # ####################################################################### attempt SLPD_now - SLPD_original
        # ### calculate global model's SLPD
        # client_SLPD_diffs = []
        # SLPD_ori = global_model_pre(x_ray)
        # SLPD_ori = F.softmax(SLPD_ori, dim=1)
        # SLPD_ori = SLPD_ori.detach().cpu().numpy()
        # SLPD_ori = SLPD_ori[0]
        # for i in range(client_num):
        #     AS = np.linalg.norm(client_SLPDs[i] - SLPD_ori)
        #     print("client {} 's SLPD l2-norm diff: {}".format(g_user_indices[i], AS))
        #     client_SLPD_diff = client_SLPDs[i] - SLPD_ori
        #     client_SLPD_diffs.append(client_SLPD_diff)
        # client_SLPD_diffs = np.array(client_SLPD_diffs)

        ################################################## the first screening: for abnormal SLPD value like nan etc.
        client_models_nonan = []
        client_SLPDs_nonan = []
        jjj = 0
        for i in range(client_num):
            for j in range(len(client_SLPDs[i])):
                jjj = j
                if np.isnan(client_SLPDs[i][j]):
                    print("********delete client {}'s model for nan********".format(i))
                    # del client_SLPDs[j]
                    # del client_models[j]
                    # break
                    break

            if jjj == len(client_SLPDs[i])-1:
                client_models_nonan.append(client_models[i])
                client_SLPDs_nonan.append(client_SLPDs[i])

        client_num_remain = len(client_models_nonan)

        ############################################### the second screening: for abnormal SLPDs compared with others
        # non_similarity_sum = [0.0 for i in range(len(client_models))]
        # non_similarity = []
        # for k in range(client_num_remain):
        #     sum = 0
        #     for j in range(client_num_remain):
        #         sum += np.linalg.norm(np.array(client_SLPDs[k]) - np.array(client_SLPDs[j]), ord=2)
        #     non_similarity.append(sum)
        #
        # non_similarity_sum += np.array(non_similarity)
        # print("=======non_similarity_sum=======", non_similarity_sum)
        #
        # threshold = np.mean(non_similarity_sum)
        # client_models_remain = []
        # g_user_indices_remain = []
        # for i in range(client_num_remain):
        #     if non_similarity_sum[i] <= threshold:
        #         client_models_remain.append(client_models[i])
        #         g_user_indices_remain.append(i)
        #
        # print("======remain models & number======", g_user_indices_remain, len(g_user_indices_remain))

        # return client_models_remain, g_user_indices_remain

        ######################################################################### the second screening: cluster SLPDs
        start = time.time()

        pca = PCA(n_components=3)
        X_new = pca.fit_transform(client_SLPDs_nonan)
        # X_new = pca.fit_transform(net_vec)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(X_new)
        majority = Counter(cluster_labels)
        majority = majority.most_common()[0][0]

        client_models_remain = []
        g_user_indices_remain = []
        for i in range(client_num_remain):
            if cluster_labels[i] == majority:
                client_models_remain.append(client_models_nonan[i])
                g_user_indices_remain.append(i)
        print("======models selected=====:", g_user_indices_remain)
        end = time.time()
        run_time2 = end - start
        # print("************************************************", run_time1, run_time2, run_time1+run_time2)
        return client_models_remain, g_user_indices_remain

class RFA(Defense):
    """
    we implement the robust aggregator at:
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation:
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, maxiter=4, eps=1e-5, ftol=1e-6, device=torch.device("cuda"), *args, **kwargs):
        """
        Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        net_freq = [0.1 for i in range(len(client_model))]
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_model]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        print('Starting Weiszfeld algorithm')
        print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            print("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        # print("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_model[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        # tot_weights = np.sum(weights)
        # weighted_updates = [np.zeros_like(v) for v in points[0]]
        # for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        # return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])





