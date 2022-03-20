import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from flTrainer import *
import copy
from model import *
import torchvision
from model.vgg import get_vgg_model

def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='parameter board')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.98, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--local_training_epoch', type=int, default=1, help='number of local training epochs')
    parser.add_argument('--malicious_local_training_epoch', type=int, default=1, help='number of malicious local training epochs')
    parser.add_argument('--num_nets', type=int, default=10, help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=5, help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=50, help='total number of FL round to conduct')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataname', type=str, default='cifar10', help='dataset to use during the training process')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--datadir', type=str, default='./dataset/', help='the directory of dataset')
    parser.add_argument('--partition_strategy', type=str, default='hetero-dir', help='dataset iid(homo) or non-iid(hetero-dir)')
    parser.add_argument('--dir_parameter', type=float, default=0.8, help='the parameter of dirichlet distribution')
    parser.add_argument('--model', type=str, default='vgg9', help='model to use during the training process')
    parser.add_argument('--load_premodel', type=bool_string, default=False, help='whether load the pre-model in begining')
    parser.add_argument('--save_model', type=bool_string, default=False, help='whether save the intermediate model')
    parser.add_argument('--client_select', type=str, default='fix-pool', help='the strategy for PS to select client: fix-frequency|fix-pool')

    # parameters for backdoor attacker
    parser.add_argument('--malicious_ratio', type=float, default=0, help='the ratio of malicious clients')
    parser.add_argument('--trigger_ori_label', type=int, default=5, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--trigger_tar_label', type=int, default=7, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--semantic_label', type=int, default=2, help='The NO. of semantic label (int, range from 0 to 9, default: 2)')
    parser.add_argument('--poisoned_portion', type=float, default=0.5, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--attack_mode', type=str, default="none", help='attack method used: none|stealthy|pgd|replacement')
    parser.add_argument('--pgd_eps', type=float, default=5e-2, help='the eps of pgd')
    parser.add_argument('--backdoor_type', type=str, default="trigger", help='backdoor type used: none|trigger|semantic|edge-case|')
    parser.add_argument('--model_scaling', type=float, default=1, help='model replacement technology')

    # parameters for untargeted attacker
    parser.add_argument('--untargeted_type', type=str, default="none", help='untargeted type used: none|label-flipping|sign-flipping|same-value|krum-attack|xmam-attack|')

    # parameters for defenders
    parser.add_argument('--defense_method', type=str, default="none",help='defense method used: none|krum|multi-krum|xmam|ndc|rsa|rfa|weak-dp|rlr|har')

    #############################################################################
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    device = torch.device(args.device if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    ###################################################################################### select networks
    if args.model == "lenet":
        if args.load_premodel==True:
            net_avg = LeNet().to(device)
            with open("savedModel/mnist_lenet.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")

            net_avg_clean = LeNet().to(device)
            with open("savedModel/mnist_lenet.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg_clean.load_state_dict(ckpt_state_dict)
            logger.info("Loading clean pre-model successfully ...")


        else:
            net_avg = LeNet().to(device)
    elif args.model in ("vgg9", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"):
        if args.load_premodel==True:
            net_avg = get_vgg_model(args.model, args.num_class).to(device)
            if args.model == 'vgg9':
                with open("savedModel/cifar10_vgg9_trigger5to0Backdoored.pt", "rb") as ckpt_file:
                # with open("savedModel/cifar10_vgg9.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            elif args.model == 'vgg11':
                with open("savedModel/cifar100_vgg11_500round.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = get_vgg_model(args.model, args.num_class).to(device)

    ############################################################################ adjust data distribution
    if args.backdoor_type in ('none', 'trigger'):
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'semantic':
        net_dataidx_map = partition_data_semantic(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'edge-case':
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)

    ########################################################################################## load dataset
    train_data, test_data = load_init_data(dataname=args.dataname, datadir=args.datadir)

    ######################################################################################### create data loader
    if args.backdoor_type == 'none':
        test_data_ori_loader, _ = create_test_data_loader(args.dataname, test_data, args.trigger_ori_label, args.trigger_tar_label,
                                                    args.batch_size)
        test_data_backdoor_loader = test_data_ori_loader
    elif args.backdoor_type == 'trigger':
        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader(args.dataname, test_data, args.trigger_ori_label, args.trigger_tar_label,
                                                     args.batch_size)
    elif args.backdoor_type == 'semantic':
        with open('./backdoorDataset/green_car_transformed_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = args.semantic_label * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # green car -> label as bird

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)
    elif args.backdoor_type == 'edge-case':
        with open('./backdoorDataset/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = 9 * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # southwest airplane -> label as truck

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)


    # net_avg.eval()
    # for batch_idx, (batch_x, batch_y) in enumerate(test_data_ori_loader):
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #
    #     if batch_y == 5:
    #         _,_,_,_,_,_,_,_,_,_,batch_y_predict = net_avg(batch_x)
    #         batch_y_predict = torch.argmax(batch_y_predict, dim=1)
    #         batch_y_predict = batch_y_predict.item()
    #         if batch_y_predict == 5:
    #             zc5 = batch_x
    #             break
    # for batch_idx, (batch_x, batch_y) in enumerate(test_data_ori_loader):
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #     if batch_y == 7:
    #         _,_,_,_,_,_,_,_,_,_,batch_y_predict = net_avg(batch_x)
    #         batch_y_predict = torch.argmax(batch_y_predict, dim=1)
    #         batch_y_predict = batch_y_predict.item()
    #         if batch_y_predict == 7:
    #             zc7 = batch_x
    #             break
    # for batch_idx, (batch_x, batch_y) in enumerate(test_data_backdoor_loader):
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #     if batch_y == 7:
    #         _,_,_,_,_,_,_,_,_,_,batch_y_predict = net_avg(batch_x)
    #         batch_y_predict = torch.argmax(batch_y_predict, dim=1)
    #         batch_y_predict = batch_y_predict.item()
    #         if batch_y_predict == 7:
    #             d5 = batch_x
    #             break
    #
    #
    #
    #
    # zc5x1, zc5x2, zc5x3, zc5x4, zc5x5, zc5x6, zc5x7, zc5x8, zc5x9, zc5x10, zc5x11 = net_avg(zc5)
    # d5x1, d5x2, d5x3, d5x4, d5x5, d5x6, d5x7, d5x8, d5x9, d5x10, d5x11 = net_avg(d5)
    # zc7x1, zc7x2, zc7x3, zc7x4, zc7x5, zc7x6, zc7x7, zc7x8, zc7x9, zc7x10, zc7x11 = net_avg(zc7)
    #
    #
    # d5_zc5_dist1 = torch.norm(d5x1 - zc5x1).item()
    # d5_zc7_dist1 = torch.norm(zc5x1 - zc7x1).item()
    # d5_zc5_dist2 = torch.norm(d5x2 - zc5x2).item()
    # d5_zc7_dist2 = torch.norm(d5x2 - zc7x2).item()
    # d5_zc5_dist3 = torch.norm(d5x3 - zc5x3).item()
    # d5_zc7_dist3 = torch.norm(d5x3 - zc7x3).item()
    # d5_zc5_dist4 = torch.norm(d5x4 - zc5x4).item()
    # d5_zc7_dist4 = torch.norm(d5x4 - zc7x4).item()
    # d5_zc5_dist5 = torch.norm(d5x5 - zc5x5).item()
    # d5_zc7_dist5 = torch.norm(d5x5 - zc7x5).item()
    # d5_zc5_dist6 = torch.norm(d5x6 - zc5x6).item()
    # d5_zc7_dist6 = torch.norm(d5x6 - zc7x6).item()
    # d5_zc5_dist7 = torch.norm(d5x7 - zc5x7).item()
    # d5_zc7_dist7 = torch.norm(d5x7 - zc7x7).item()
    # d5_zc5_dist8 = torch.norm(d5x8 - zc5x8).item()
    # d5_zc7_dist8 = torch.norm(d5x8 - zc7x8).item()
    # d5_zc5_dist9 = torch.norm(d5x9 - zc5x9).item()
    # d5_zc7_dist9 = torch.norm(d5x9 - zc7x9).item()
    # d5_zc5_dist10 = torch.norm(d5x10 - zc5x10).item()
    # d5_zc7_dist10 = torch.norm(d5x10 - zc7x10).item()
    # d5_zc5_dist11 = torch.norm(d5x11 - zc5x11).item()
    # d5_zc7_dist11 = torch.norm(d5x11 - zc7x11).item()
    #
    #
    # print("poisoned model   毒5-干净5        毒5-干净7")
    # print("layer 1 : ", d5_zc5_dist1, d5_zc7_dist1)
    # print("layer 2 : ", d5_zc5_dist2, d5_zc7_dist2)
    # print("layer 3 : ", d5_zc5_dist3, d5_zc7_dist3)
    # print("layer 4 : ", d5_zc5_dist4, d5_zc7_dist4)
    # print("layer 5 : ", d5_zc5_dist5, d5_zc7_dist5)
    # print("layer 6 : ", d5_zc5_dist6, d5_zc7_dist6)
    # print("layer 7 : ", d5_zc5_dist7, d5_zc7_dist7)
    # print("layer 8 : ", d5_zc5_dist8, d5_zc7_dist8)
    # print("layer 9 : ", d5_zc5_dist9, d5_zc7_dist9)
    # print("layer 10 : ", d5_zc5_dist10, d5_zc7_dist10)
    # print("layer 11 : ", d5_zc5_dist11, d5_zc7_dist11)
    #
    # zc5x1, zc5x2, zc5x3, zc5x4, zc5x5, zc5x6, zc5x7, zc5x8, zc5x9, zc5x10, zc5x11 = net_avg_clean(zc5)
    # d5x1, d5x2, d5x3, d5x4, d5x5, d5x6, d5x7, d5x8, d5x9, d5x10, d5x11 = net_avg_clean(d5)
    # zc7x1, zc7x2, zc7x3, zc7x4, zc7x5, zc7x6, zc7x7, zc7x8, zc7x9, zc7x10, zc7x11 = net_avg_clean(zc7)
    #
    # d5_zc5_dist1 = torch.norm(d5x1 - zc5x1).item()
    # d5_zc7_dist1 = torch.norm(zc5x1 - zc7x1).item()
    # d5_zc5_dist2 = torch.norm(d5x2 - zc5x2).item()
    # d5_zc7_dist2 = torch.norm(d5x2 - zc7x2).item()
    # d5_zc5_dist3 = torch.norm(d5x3 - zc5x3).item()
    # d5_zc7_dist3 = torch.norm(d5x3 - zc7x3).item()
    # d5_zc5_dist4 = torch.norm(d5x4 - zc5x4).item()
    # d5_zc7_dist4 = torch.norm(d5x4 - zc7x4).item()
    # d5_zc5_dist5 = torch.norm(d5x5 - zc5x5).item()
    # d5_zc7_dist5 = torch.norm(d5x5 - zc7x5).item()
    # d5_zc5_dist6 = torch.norm(d5x6 - zc5x6).item()
    # d5_zc7_dist6 = torch.norm(d5x6 - zc7x6).item()
    # d5_zc5_dist7 = torch.norm(d5x7 - zc5x7).item()
    # d5_zc7_dist7 = torch.norm(d5x7 - zc7x7).item()
    # d5_zc5_dist8 = torch.norm(d5x8 - zc5x8).item()
    # d5_zc7_dist8 = torch.norm(d5x8 - zc7x8).item()
    # d5_zc5_dist9 = torch.norm(d5x9 - zc5x9).item()
    # d5_zc7_dist9 = torch.norm(d5x9 - zc7x9).item()
    # d5_zc5_dist10 = torch.norm(d5x10 - zc5x10).item()
    # d5_zc7_dist10 = torch.norm(d5x10 - zc7x10).item()
    # d5_zc5_dist11 = torch.norm(d5x11 - zc5x11).item()
    # d5_zc7_dist11 = torch.norm(d5x11 - zc7x11).item()
    #
    # print("clean model     毒5-干净5         毒5-干净7")
    # print("layer 1 : ", d5_zc5_dist1, d5_zc7_dist1)
    # print("layer 2 : ", d5_zc5_dist2, d5_zc7_dist2)
    # print("layer 3 : ", d5_zc5_dist3, d5_zc7_dist3)
    # print("layer 4 : ", d5_zc5_dist4, d5_zc7_dist4)
    # print("layer 5 : ", d5_zc5_dist5, d5_zc7_dist5)
    # print("layer 6 : ", d5_zc5_dist6, d5_zc7_dist6)
    # print("layer 7 : ", d5_zc5_dist7, d5_zc7_dist7)
    # print("layer 8 : ", d5_zc5_dist8, d5_zc7_dist8)
    # print("layer 9 : ", d5_zc5_dist9, d5_zc7_dist9)
    # print("layer 10 : ", d5_zc5_dist10, d5_zc7_dist10)
    # print("layer 11 : ", d5_zc5_dist11, d5_zc7_dist11)


    logger.info("Test the model performance on the entire task before FL process ... ")
    overall_acc = test_model(net_avg, test_data_ori_loader, device, print_perform=True)
    logger.info("Test the model performance on the backdoor task before FL process ... ")
    backdoor_acc = test_model(net_avg, test_data_backdoor_loader, device, print_perform=False)

    logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
    logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))

    # logger.info("Test the model performance on the entire task before FL process ... ")
    # overall_acc = test_model(net_avg, test_data_ori_loader, device, print_perform=True)
    # logger.info("Test the model performance on the backdoor task before FL process ... ")
    # test_model_backdoor(net_avg, test_data_backdoor_loader, device, print_perform=False)

    # logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
    # logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))




    arguments = {
        "net_avg": net_avg,
        "partition_strategy": args.partition_strategy,
        "dir_parameter": args.dir_parameter,
        "net_dataidx_map": net_dataidx_map,
        "num_nets": args.num_nets,
        "dataname": args.dataname,
        "num_class": args.num_class,
        "datadir": args.datadir,
        "model": args.model,
        "load_premodel":args.load_premodel,
        "save_model":args.save_model,
        "client_select":args.client_select,
        "part_nets_per_round": args.part_nets_per_round,
        "fl_round": args.fl_round,
        "local_training_epoch": args.local_training_epoch,
        "malicious_local_training_epoch": args.malicious_local_training_epoch,
        "args_lr": args.lr,
        "args_gamma": args.gamma,
        "batch_size": args.batch_size,
        "device": device,
        "test_data_ori_loader": test_data_ori_loader,
        "test_data_backdoor_loader": test_data_backdoor_loader,
        "malicious_ratio": args.malicious_ratio,
        "trigger_ori_label": args.trigger_ori_label,
        "trigger_tar_label": args.trigger_tar_label,
        "semantic_label": args.semantic_label,
        "poisoned_portion": args.poisoned_portion,
        "attack_mode": args.attack_mode,
        "pgd_eps": args.pgd_eps,
        "backdoor_type": args.backdoor_type,
        "model_scaling": args.model_scaling,
        "untargeted_type": args.untargeted_type,
        "defense_method": args.defense_method,
    }

    fl_trainer = FederatedLearningTrainer(arguments=arguments)
    fl_trainer.run()
