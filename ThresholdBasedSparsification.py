
import torch
import numpy as np
import time
import sys
from scipy.stats import rankdata
from delay_generation import PerCoordinateDelay
from LearningFunctions import MainModelOptimizerCriterion, LoaderValidation, Initialize_errors_dict, send_main_model_to_nodes_and_update_model_dict, create_model_optimizer_criterion_dict
from LearningFunctions import get_num_params, create_grads_dict, ErrorAccumulation, ErrorDictUpdate, UpdateCalculator, UpdateMainModel, DelayCacluator


#-----------------------------------------------------------------------------------------------
def SparsificationQueueUpdate(current_queue, threshold_values, num_of_clients, epsilon):
  new_queue = np.zeros(num_of_clients)
  new_queue = current_queue+ threshold_values**2
  new_queue = np.maximum(new_queue - epsilon, 0)

  return new_queue
#-----------------------------------------------------------------------------------------------
def OptimalSparsisfication(method, errorplusgrads_dict, num_of_clients, sparsification_queue, devices_percoordinate_delay, epsilon, V):
  optimized_sparsification= np.zeros(num_of_clients)
  for i in range(num_of_clients):
    errorplusgrad = errorplusgrads_dict['client'+str(i)]
    optimized_sparsification[i] = OptimizingSparsification(method, errorplusgrad, sparsification_queue[i], devices_percoordinate_delay[i], epsilon, V)
    
  return optimized_sparsification

#-----------------------------------------------------------------------------------------------
def OptimizingSparsification(method, errorplusgrad, sparsification_queue, percoordinate_delay, epsilon, V):
  flat_tensors = torch.cat([tensor.flatten() for tensor in errorplusgrad])
  abs_flat_tensors = torch.abs(flat_tensors)

  abs_flat_tensors_cpu = abs_flat_tensors.cpu() 
  abs_np = abs_flat_tensors_cpu.numpy()

  abs_np = np.append(abs_np, 0.0)

  if method == 'AdaSparse':
    ranks = len(abs_np) - rankdata(abs_np, method='max')
    costs = CostFunc(ranks, abs_np, sparsification_queue, percoordinate_delay, epsilon, V)
    index= np.argmin(costs)
    optimal_thershold= abs_np[index]

  elif method == 'LCAdaSparse':
    B = 25
    bin_representatives, greater_elems= count_elements_in_intervals(abs_np, B)
    costs = CostFunc(greater_elems, bin_representatives, sparsification_queue, percoordinate_delay, epsilon, V)
    index= np.argmin(costs)
    optimal_thershold= bin_representatives[index]

  return optimal_thershold
#-----------------------------------------------------------------------------------------------
def count_elements_in_intervals(array, B):
  array = 1e5*array
  max_val = array.max()
  interval_length = max_val / B
  bin_representatives= interval_length*np.arange(1, B+1)
    
  bin_indices = np.ceil(array / interval_length).astype(np.int32)
  mask = (bin_indices != 0) 
  array= array[mask]
  bin_indices= bin_indices[mask]-1
  counts = np.bincount(bin_indices)

  cumsum_counts= np.cumsum(counts)
  greater_elems= len(array)-cumsum_counts

  return bin_representatives/1e5, greater_elems

#-----------------------------------------------------------------------------------------------
def CostFunc(num_non_zero_array, sparsification_thres, sparsification_queue, percoordinate_delay, epsilon, V):
  const1= sparsification_queue*np.power(sparsification_thres, 2)+ 0.5* np.power(sparsification_thres, 4)
  const2 = V*percoordinate_delay
  cost = const1+ const2*num_non_zero_array
  return cost

#-----------------------------------------------------------------------------------------------
def ApplyingSparsification(sparsification_threshold, u):
  sparsified_u = [torch.where(torch.abs(tensor) <= sparsification_threshold, torch.tensor(0.0), tensor) for tensor in u]
  return sparsified_u

#-----------------------------------------------------------------------------------------------
def TopK(K, errorplusgrads_dict, num_of_clients):
  sparsification_thresholds = np.zeros(num_of_clients)
  for j in range(num_of_clients):
    u_vec= np.absolute(torch.nn.utils.parameters_to_vector(errorplusgrads_dict['client'+str(j)]).cpu().numpy())
    kth_largest = np.partition(-1*u_vec, K-1)[K-1]
    sparsification_thresholds[j]= -1*kth_largest-1e-6

  return sparsification_thresholds
#-----------------------------------------------------------------------------------------------
def ThresholdSparsification(errorplusgrads_dict, sparsification_thresholds, num_of_clients):
  sparsified_grads_dict = dict()
  for j in range(num_of_clients):
    errorplusgrad= errorplusgrads_dict['client'+str(j)]
    sparsified_vec= ApplyingSparsification(sparsification_thresholds[j], errorplusgrad)
    sparsified_grads_dict.update({'client'+str(j) : sparsified_vec }) 

  return sparsified_grads_dict

#------------------------------------------------------------------------------------------------
def TDistributedSGD(loaders, train_loader, test_loader, model_name, num_epoch, 
                   num_of_clients, clients_num_sample, learning_rate, batch_size, method, 
                   kappa, num_classes, epsilon, max_delay, delay_dist, alpha, device):
  
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
  sparsification_matrix = np.zeros((total_num_commu_rounds, num_of_clients))
  delay_matrix = np.zeros((total_num_commu_rounds, num_of_clients))
  num_nonzeros_array = np.zeros((total_num_commu_rounds, num_of_clients))
  sparsification_queue = np.zeros(num_of_clients)

  print("Epoch", str(0), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )

  errors_dict = Initialize_errors_dict(main_model, num_of_clients, device)

  for i in range(num_epoch):
    for j in range(num_of_comm_round_in_each_epoch):

      devices_percoordinate_delay =  PerCoordinateDelay(num_of_clients, max_delay, distribution = delay_dist, alpha= alpha)
      model_dict= send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_of_clients)
      grads_dict= create_grads_dict(model_dict, criterion_dict, optimizer_dict, loaders, device)
      errorplusgrads_dict = ErrorAccumulation(grads_dict, errors_dict, learning_rate, num_of_clients)

      if method == 'AdaSparse':
        V = (kappa*num_epoch)/(max_delay*model_size)
        sparsification_thresholds = OptimalSparsisfication('AdaSparse', errorplusgrads_dict, num_of_clients, 
                                                           sparsification_queue, 
                                                           devices_percoordinate_delay, epsilon, V)

        sparsification_queue = SparsificationQueueUpdate(sparsification_queue, sparsification_thresholds, num_of_clients, epsilon)

      elif method == 'LCAdaSparse':
        V = (kappa*num_epoch)/(max_delay*model_size)
        sparsification_thresholds = OptimalSparsisfication('LCAdaSparse', errorplusgrads_dict, num_of_clients, 
                                                           sparsification_queue, 
                                                           devices_percoordinate_delay, epsilon, V)

        sparsification_queue = SparsificationQueueUpdate(sparsification_queue, sparsification_thresholds, num_of_clients, epsilon)

      elif method == 'UniSparse':
        sparsification_thresholds = kappa*np.ones(num_of_clients)

      elif method == 'withoutsparsification':
        sparsification_thresholds = np.zeros(num_of_clients)

      elif method == 'topk':
        K = int(kappa)
        if K > model_size:
          K = model_size
        sparsification_thresholds = TopK(K, errorplusgrads_dict, num_of_clients)

      sparsified_grads_dict = ThresholdSparsification(errorplusgrads_dict, sparsification_thresholds, num_of_clients)
      errors_dict = ErrorDictUpdate(num_of_clients, errorplusgrads_dict, sparsified_grads_dict, learning_rate)
      update = UpdateCalculator(main_model, sparsified_grads_dict, num_of_clients, device)
      main_model = UpdateMainModel(main_model, update, main_optimizer)

      sparsification_matrix[i*num_of_comm_round_in_each_epoch+j] = 1*sparsification_thresholds
      delay_array, num_nonzeros = DelayCacluator(sparsified_grads_dict, num_of_clients, devices_percoordinate_delay)
      num_nonzeros_array[i*num_of_comm_round_in_each_epoch+j] = num_nonzeros
      delay_matrix[i*num_of_comm_round_in_each_epoch+j] = delay_array

      del sparsified_grads_dict
      del grads_dict
      del errorplusgrads_dict
      torch.cuda.empty_cache()

    train_loss, train_acc= LoaderValidation(main_model, main_criterion, train_loader, flag, device)
    test_loss, test_acc= LoaderValidation(main_model, main_criterion, test_loader, flag, device)
    train_loss_list[i+1] = train_loss
    train_acc_list[i+1] = train_acc
    test_loss_list[i+1] = test_loss
    test_acc_list[i+1] = test_acc

    print("Epoch", str(i+1), "| Test loss: ", test_loss, "Test acc :", test_acc, "Train loss: " , train_loss, "Train acc :", train_acc )


  return train_acc_list , test_acc_list , train_loss_list , test_loss_list, sparsification_matrix, delay_matrix, num_nonzeros_array

#-----------------------------------------------------------------------------------------------

