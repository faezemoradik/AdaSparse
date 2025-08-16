
import torch
from torch import nn
import numpy as np
import sys
from models import ResNet, LogisticRegression, BasicBlock, ResNet9

#-----------------------------------------------------------------------------------------------
def DelayCacluator(sparsified_grads_dict, num_of_clients, devices_percoordinate_delay):
  num_non_zero_elements= np.zeros(num_of_clients)
  delay_array= np.zeros(num_of_clients)
  for i in range(num_of_clients):
    l_zero_norm= LZeroNorm(sparsified_grads_dict['client'+str(i)])
    delay_array[i] = devices_percoordinate_delay[i]*l_zero_norm
    num_non_zero_elements[i]= l_zero_norm
    # print('num of non-zero: ', LZeroNorm(sparsified_grads_dict['client'+str(i)]))

  return delay_array, num_non_zero_elements
#-----------------------------------------------------------------------------------------------
def LZeroNorm(u):
  count = 0
  for i in range(len(u)):
    count += torch.count_nonzero(u[i])
  return count
#-----------------------------------------------------------------------------------------------
def train_clients(model, inputs, targets, criterion, optimizer):
  model.train()
  output = model(inputs)
  loss = criterion(output, targets)
  optimizer.zero_grad()
  loss.backward()
  grad_vector=[]
  for param in model.parameters():
    grad_vector.append(param.grad) 

  return grad_vector
#-----------------------------------------------------------------------------------------------
def create_grads_dict(model_dict, criterion_dict , optimizer_dict, loaders, device):
  grads_dict = dict()
  for j, dataloader in enumerate(loaders):
    model= model_dict['model'+ str(j)]
    criterion= criterion_dict['criterion'+ str(j)]
    optimizer= optimizer_dict['optimizer'+ str(j)]

    data_iter = iter(dataloader)
    xt, yt = next(data_iter)
    xt= xt.to(device)
    yt= yt.to(device)

    grad= train_clients(model, xt,  yt, criterion, optimizer)

    grads_dict.update({'client'+str(j) : grad })

  return grads_dict
#-----------------------------------------------------------------------------------------------
def WeightedAddFunc(a, b, s1, s2):
  add=[]
  for j in range(len(a)):
    add.append(s1*a[j]+s2*b[j])
  return add
#-----------------------------------------------------------------------------------------------
def ErrorAccumulation(grads_dict, errors_dict, learning_rate, num_of_clients):
  errorplusgrads_dict = dict()
  for j in range(num_of_clients):
    errplusgrad= WeightedAddFunc(grads_dict['client'+str(j)], errors_dict['client'+str(j)], 1.0, 1.0/learning_rate)
    errorplusgrads_dict.update({'client'+str(j) : errplusgrad })

  return errorplusgrads_dict

#-----------------------------------------------------------------------------------------------
def ErrorDictUpdate(num_of_clients, errorplusgrads_dict, sparsified_grads_dict, learning_rate):
  errors_dict= dict()
  for j in range(num_of_clients):
    subtraction= WeightedAddFunc(errorplusgrads_dict['client'+str(j)], sparsified_grads_dict['client'+str(j)], learning_rate, -1*learning_rate)
    errors_dict.update({'client'+str(j) : subtraction }) 

  return errors_dict
#-----------------------------------------------------------------------------------------------
def l2_norm_calculator(u):
  return np.linalg.norm(torch.nn.utils.parameters_to_vector(u).cpu().numpy())
#-----------------------------------------------------------------------------------------------
def Initialize_errors_dict(main_model, num_of_clients, device):
  error = [torch.zeros_like(param).to(device) for param in main_model.parameters()]
  errors_dict= dict()
  for j in range(num_of_clients):
    errors_dict.update({'client'+str(j) :  error })

  return errors_dict

#-----------------------------------------------------------------------------------------------
def create_model_optimizer_criterion_dict(num_of_clients, learning_rate, model_name, num_classes, device):
  model_dict = dict()
  optimizer_dict= dict()
  criterion_dict = dict()
  for i in range(num_of_clients):
    if model_name == 'LogisticRegression':
      model_info = LogisticRegression().to(device)
      optimizer_info= torch.optim.SGD(model_info.parameters(), lr=learning_rate, weight_decay=0.0001)
      criterion_info = nn.CrossEntropyLoss().to(device)
        
    elif model_name == 'ResNet20':
      model_info = ResNet(BasicBlock, [3,3,3]).to(device) ##resnet20
      optimizer_info= torch.optim.SGD(model_info.parameters(), lr=learning_rate)
      criterion_info = nn.CrossEntropyLoss().to(device)   

    elif model_name == 'ResNet9':
      model_info = ResNet9(num_classes= num_classes, scale_norm=False, norm_layer= "group").to(device) ##resnet9
      optimizer_info = torch.optim.NAdam(model_info.parameters(), lr=learning_rate)  
      criterion_info = nn.CrossEntropyLoss().to(device)

    model_dict.update({"model"+str(i) : model_info })
    optimizer_dict.update({"optimizer"+str(i) : optimizer_info })
    criterion_dict.update({"criterion"+str(i) : criterion_info})
        
  return model_dict, optimizer_dict, criterion_dict 

#-----------------------------------------------------------------------------------------------
def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_of_clients):
  state_dict = main_model.state_dict()
  filtered_state_dict = {k: v for k, v in state_dict.items() if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}
  for i in range(num_of_clients):
    model_dict['model'+str(i)].load_state_dict(filtered_state_dict, strict=False)
    # model_dict['model'+str(i)].load_state_dict(state_dict)

  return model_dict
#----------------------------------------------------------------------
def UpdateCalculator(model, grads_dict, num_of_clients, device):
  update = [torch.zeros_like(param).to(device) for param in model.parameters()]

  for i in range(num_of_clients):
    update = WeightedAddFunc(grads_dict['client'+str(i)], update, 1.0/num_of_clients, 1.0)

  return update
#---------------------------------------------------------------------------
def UpdateMainModel(main_model, update_params, main_optimizer):
  main_model.train()
  i=0
  for p in main_model.parameters():
    p.grad = update_params[i]
    i +=1
  main_optimizer.step()
  # if lr_scheduler is not None:
  #   lr_scheduler.step()
  return main_model
#-----------------------------------------------------------------------------------------------
def validation(model, input, target, criterion, mode):
  model.train(mode= mode)
  with torch.no_grad():
    output = model(input)
    loss = criterion(output, target)
    if isinstance(criterion, nn.CrossEntropyLoss):
      prediction = output.argmax(dim=1, keepdim=True)
      correct = prediction.eq(target.view_as(prediction)).sum().item()
      acc= correct/len(input)
    elif isinstance(criterion, nn.BCELoss):
      prediction = (output > 0.5).float()
      correct = (prediction == target).sum().item()
      num_label = target.shape[1]
      acc= correct/(num_label*len(input))

  return loss.item(), acc

#----------------------------------------------------------------------------
def LoaderValidation(model, criterion, loader, mode, device):
  loader_loss=0
  loader_acc=0
  for j, (batch_data, batch_labels) in enumerate(loader):
    batch_data= batch_data.to(device)
    batch_labels = batch_labels.to(device)
    loss, acc= validation(model, batch_data, batch_labels, criterion, mode)
    loader_loss += loss*batch_data.size(0)
    loader_acc += acc*batch_data.size(0)

  loader_loss /= len(loader.dataset)
  loader_acc /= len(loader.dataset)

  return loader_loss, loader_acc
#-----------------------------------------------------------------------------------------------
def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#-----------------------------------------------------------------------------------------------
def MainModelOptimizerCriterion(model_name, learning_rate, num_classes, device):
  if model_name == 'LogisticRegression':
    main_model = LogisticRegression().to(device)
    main_optimizer= torch.optim.SGD(main_model.parameters(), lr=learning_rate, weight_decay=0.0001)
    main_criterion = nn.CrossEntropyLoss().to(device)
    flag = False
    
  elif model_name == 'ResNet20': 
    main_model = ResNet(BasicBlock, [3,3,3]).to(device) ### ResNet 20
    main_optimizer= torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    main_criterion = nn.CrossEntropyLoss().to(device)
    flag = False
  elif model_name == 'ResNet9':
    main_model = ResNet9(num_classes= num_classes, scale_norm=False, norm_layer= "group").to(device)
    main_optimizer = torch.optim.NAdam(main_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    main_criterion = nn.CrossEntropyLoss().to(device)
    flag = False

  return main_model, main_optimizer, main_criterion, flag

#-----------------------------------------------------------------------------------------------