
import torch
from torch import nn
import numpy as np
import sys
import time
from LearningFunctions import MainModelOptimizerCriterion, LoaderValidation, send_main_model_to_nodes_and_update_model_dict, create_model_optimizer_criterion_dict
from LearningFunctions import get_num_params, create_grads_dict, UpdateCalculator, UpdateMainModel, DelayCacluator
from delay_generation import PerCoordinateDelay

#-------------------------------------------------------------------
def DetermineProbVec(kappa, grad, model_size):
    prob = [torch.zeros_like(tensor) for tensor in grad]
    total_sum = sum(torch.sum(torch.abs(tensor)).item() for tensor in grad)
    for i, tensor in enumerate(grad):
      prob[i] = torch.minimum(torch.ones_like(tensor), kappa * model_size * (1 / total_sum) * torch.abs(tensor))
    for _ in range(2):
      indices = [(i, torch.where(prob[i] != 1)) for i in range(len(prob))]
      non_one_probs = torch.cat([prob[i][idx] for i, idx in indices])
      non_one_count = len(non_one_probs)
      c = (kappa * model_size - model_size + non_one_count) / torch.sum(non_one_probs).item()
      if abs(c-1) <= 1e-6:
        break
      for i, idx in indices:
        prob[i][idx] = torch.minimum(c * prob[i][idx], torch.ones_like(prob[i][idx]))

    return prob

#---------------------------------------------------------------------
def VarReducedSpar(kappa, model_size, grads_dict, num_of_clients):
    sparsed_grad_dict= dict()
    for j in range(num_of_clients):
      probabilities= DetermineProbVec(kappa, grads_dict['client'+str(j)], model_size)
      sparsed_grad= UnbiasedSparsifier(grads_dict['client'+str(j)], probabilities)
      sparsed_grad_dict.update({'client'+str(j) : sparsed_grad })

    return sparsed_grad_dict

#-------------------------------------------------------------------------
def UnbiasedSparsifier(grad, probabilities):
    sparse_gard = [torch.zeros_like(tensor) for tensor in grad]
    for i, tensor in enumerate(grad):
      rv= torch.bernoulli(probabilities[i])
      sparse_gard[i]= torch.where(probabilities[i] != 0, rv * tensor / probabilities[i], torch.zeros_like(tensor))

    return sparse_gard

#------------------------------------------------------------------------------------------------
def VDistributedSGD(loaders, train_loader, test_loader, model_name, num_epoch, 
                   num_of_clients, clients_num_sample, learning_rate, batch_size,  
                   kappa, num_classes, max_delay, delay_dist, alpha, device):
  
  num_of_comm_round_in_each_epoch = int(clients_num_sample[0]/batch_size)
  total_num_commu_rounds= num_of_comm_round_in_each_epoch*num_epoch
  print('total_num_commu_rounds is: ', total_num_commu_rounds)

  main_model, main_optimizer, main_criterion, flag = MainModelOptimizerCriterion(model_name, learning_rate, num_classes, device)

  model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(num_of_clients, learning_rate, model_name, num_classes, device)
  model_size= get_num_params(main_model)
  print('model size is: ', model_size)

  train_acc_list = np.zeros(num_epoch+1)
  test_acc_list = np.zeros(num_epoch+1)
  train_loss_list = np.zeros(num_epoch+1)
  test_loss_list = np.zeros(num_epoch+1)

  train_loss, train_acc= LoaderValidation(main_model, main_criterion, train_loader, flag, device)
  test_loss, test_acc= LoaderValidation(main_model, main_criterion, test_loader, flag, device)
  train_loss_list[0] = train_loss
  train_acc_list[0] = train_acc
  test_loss_list[0] = test_loss
  test_acc_list[0] = test_acc
  delay_matrix = np.zeros((total_num_commu_rounds, num_of_clients))
  sparsification_matrix = np.zeros((total_num_commu_rounds, num_of_clients))
  num_nonzeros_array = np.zeros((total_num_commu_rounds, num_of_clients))

  print("Epoch", str(0), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )

  for i in range(num_epoch):
    for j in range(num_of_comm_round_in_each_epoch):

      devices_percoordinate_delay =  PerCoordinateDelay(num_of_clients, max_delay, distribution = delay_dist, alpha= alpha)
      model_dict= send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_of_clients)
      grads_dict= create_grads_dict(model_dict, criterion_dict, optimizer_dict, loaders, device)
      sparsified_grads_dict = VarReducedSpar(kappa, model_size, grads_dict, num_of_clients)

      update = UpdateCalculator(main_model, sparsified_grads_dict, num_of_clients, device)
      main_model = UpdateMainModel(main_model, update, main_optimizer)


      delay_array, num_nonzeros = DelayCacluator(sparsified_grads_dict, num_of_clients, devices_percoordinate_delay)
      num_nonzeros_array[i*num_of_comm_round_in_each_epoch+j] = num_nonzeros
      delay_matrix[i*num_of_comm_round_in_each_epoch+j] = delay_array


    train_loss, train_acc= LoaderValidation(main_model, main_criterion, train_loader, flag, device)
    test_loss, test_acc= LoaderValidation(main_model, main_criterion, test_loader, flag, device)
    train_loss_list[i+1] = train_loss
    train_acc_list[i+1] = train_acc
    test_loss_list[i+1] = test_loss
    test_acc_list[i+1] = test_acc

    print("Epoch", str(i+1), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )


  return train_acc_list , test_acc_list , train_loss_list , test_loss_list, sparsification_matrix, delay_matrix, num_nonzeros_array

#---------------------------------------------------------------------------------------------------


        
    