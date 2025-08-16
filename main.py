import numpy as np
import torch
import pickle
import os
import argparse
from data import MNIST_PROCESS, CIFAR10_PROCESS, ImageNette_PROCESS
from ThresholdBasedSparsification import TDistributedSGD
from VarReducedSparsification import VDistributedSGD
import random


#----------------------------------------------------
def get_parameter():
  parser=argparse.ArgumentParser()
  parser.add_argument("-learning_rate",default=0.05,type=float,help="learning rate")
  parser.add_argument("-batch_size",default= 5,type=int,help="batch size")
  parser.add_argument("-myseed",default=0,type=int,help="seed")
  parser.add_argument("-num_epoch",default=0,type=int,help="number of epoch")
  parser.add_argument("-dataset",default='MNIST',type=str,help="dataset name")
  parser.add_argument("-datasplit",default='iid',type=str,help="data split across devices")
  parser.add_argument("-model_name",default='ResNet9',type=str,help="model name")
  parser.add_argument("-method",default='GSDS',type=str,help="method name")
  parser.add_argument("-kappa",default=0.5,type=float,help="sparsification parameter for all methods.")
  parser.add_argument("-epsilon",default=0.5,type=float,help=".")
  parser.add_argument("-delay_dist",default='uniform',type=str,help="per coordinate delay distribution")
  parser.add_argument("-alpha",default='0.8',type=float,help="alpha parameter for beta distribution if delay_dist is non-uniform")
  args= parser.parse_args()
  return args

#------------------------------------------------------------
def main():
  if torch.cuda.is_available():
    print("CUDA is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  else:
    print("CUDA is not available")

  
  args = get_parameter()
  learning_rate = args.learning_rate
  batch_size = args.batch_size
  num_epoch = args.num_epoch
  kappa= args.kappa
  dataset= args.dataset
  datasplit= args.datasplit
  model_name= args.model_name
  method= args.method
  myseed = args.myseed
  epsilon = args.epsilon
  delay_dist = args.delay_dist
  alpha = args.alpha


  #--------- Parameters ----------
  num_of_clients = int(10)### number of devices
  max_delay = 1e-4 ### maximum delay or d_max

  print('learning_rate: ', learning_rate)
  print('batch_size: ', batch_size)
  print('dataset: ', dataset)
  print('datasplit: ', datasplit)
  print('model: ', model_name)
  print('method: ', method)
  print('myseed: ', myseed)
  print('num_epoch: ', num_epoch) 
  print('kappa: ', kappa)
  print('epsilon: ', epsilon)
  print('delay_dist: ', delay_dist)
  print('alpha: ', alpha)

  torch.backends.cudnn.deterministic=True
  torch.cuda.manual_seed(0) # to get the reproducible results
  random.seed(0) # to get the reproducible results
  torch.manual_seed(0) # to get the reproducible results

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  if dataset== 'MNIST':
    train_loader, test_loader, loaders, clients_num_sample = MNIST_PROCESS(num_of_clients, batch_size, datasplit, myseed)
    num_classes = 10
  elif dataset== 'CIFAR10':
    train_loader, test_loader, loaders, clients_num_sample = CIFAR10_PROCESS(num_of_clients, batch_size, datasplit, myseed)
    num_classes= 10
  elif dataset== 'ImageNette':
    train_loader, test_loader, loaders, clients_num_sample = ImageNette_PROCESS(num_of_clients, batch_size, datasplit, myseed)
    num_classes = 10

  
  performance = dict()

  np.random.seed(myseed)
  torch.random.manual_seed(myseed)

  if method == 'varreduced':
    train_acc_list , test_acc_list , train_loss_list , test_loss_list, sparsification_matrix, delay_matrix, num_nonzeros_array = VDistributedSGD(loaders, train_loader, test_loader, model_name, num_epoch, 
                                                                                                          num_of_clients, clients_num_sample, learning_rate, batch_size, 
                                                                                                          kappa, num_classes, max_delay, delay_dist, alpha, device)
  else:
    train_acc_list , test_acc_list , train_loss_list , test_loss_list, sparsification_matrix, delay_matrix, num_nonzeros_array = TDistributedSGD(loaders, train_loader, test_loader, model_name, num_epoch, 
                                                                                                        num_of_clients, clients_num_sample, learning_rate, batch_size, method, 
                                                                                                        kappa, num_classes, epsilon, max_delay, delay_dist, alpha, device)

                                                                                                        
  performance['train_acc'] = train_acc_list
  performance['test_acc'] = test_acc_list
  performance['train_loss'] = train_loss_list
  performance['test_loss'] = test_loss_list
  performance['sparsification'] = sparsification_matrix
  performance['delay'] = delay_matrix
  performance['num_non_zeros'] = num_nonzeros_array


  file_name = dataset+method+'seed'+str(myseed)+'BS'+str(batch_size)+'LR'+str(learning_rate)+'NE'+str(num_epoch)+'kappa'+str(kappa)
  folder_name = 'epsilon'+str(epsilon)
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)

  file_path = os.path.join(folder_name, file_name)
  with open(file_path, 'wb') as f:
    pickle.dump(performance, f)
######################################

main()