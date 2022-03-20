 
### mnist
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.01 \
--gamma 0.998 \
--seed 1234 \
--num_nets 50 \
--part_nets_per_round 50 \
--fl_round 50 \
--local_training_epoch 1 \
--malicious_local_training_epoch 1 \
--dataname mnist \
--num_class 10 \
--model lenet \
--load_premodel False \
--save_model False \
--partition_strategy homo \
--dir_parameter 0.5 \
--malicious_ratio 1 \
--backdoor_type trigger \
--untargeted_type none \
--trigger_ori_label 5 \
--trigger_tar_label 7 \
--poisoned_portion 0.2 \
--attack_mode pgd \
--pgd_eps 5e-2 \
--defense_method none \
--device cuda:1


### emnist
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.01 \
--gamma 0.998 \
--seed 1234 \
--num_nets 3383 \
--part_nets_per_round 20 \
--fl_round 100 \
--local_training_epoch 1 \
--malicious_local_training_epoch 1 \
--dataname emnist \
--num_class 10 \
--model lenet \
--load_premodel False \
--save_model True \
--partition_strategy hetero-dir \
--dir_parameter 0.5 \
--malicious_ratio 0.8 \
--backdoor_type trigger \
--untargeted_type none \
--trigger_label 0 \
--poisoned_portion 0.3 \
--attack_mode none \
--pgd_eps 5e-2 \
--defense_method xmam \
--device cuda:1

### cifar10
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.00036 \
--gamma 0.998 \
--seed 1234 \
--num_nets 100 \
--part_nets_per_round 10 \
--fl_round 50 \
--local_training_epoch 1 \
--malicious_local_training_epoch 1 \
--dataname cifar10 \
--num_class 10 \
--model vgg9 \
--load_premodel True \
--save_model False \
--partition_strategy hetero-dir \
--dir_parameter 0.5 \
--malicious_ratio 0.4 \
--backdoor_type none \
--untargeted_type krum-attack \
--trigger_ori_label 5 \
--trigger_tar_label 7 \
--semantic_label 2 \
--poisoned_portion 0.2 \
--attack_mode none \
--pgd_eps 5e-2 \
--model_scaling 1 \
--defense_method none \
--device cuda:0

### cifar100
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.00036 \
--gamma 0.998 \
--seed 1234 \
--num_nets 100 \
--part_nets_per_round 10 \
--fl_round 100 \
--local_training_epoch 1 \
--malicious_local_training_epoch 1 \
--dataname cifar100 \
--num_class 100 \
--model vgg11 \
--load_premodel True \
--partition_strategy homo \
--dir_parameter 0.9 \
--malicious_ratio 0.2 \
--trigger_label 0 \
--poisoned_portion 0.3 \
--attack_mode none \
--defense_method xmam \
--device cuda:1