
import numpy as np
import torch
import math
from torchvision import datasets
import torchvision.transforms as transforms
import sys
import random
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict


#---------------------------------------------------------------------------
def count_classes_in_dataloader(dataloader, num_classes=10):
  class_counts = np.zeros(num_classes, dtype=int) 

  for batch in dataloader:
    _, labels = batch
    labels = labels.numpy()
    batch_counts = np.bincount(labels, minlength=num_classes)
    class_counts += batch_counts

  return class_counts
#---------------------------------------------------------------------------
def IIDDistribution(train_data, num_of_clients, batch_size, seed):
  total_samples = len(train_data)
  print('len train: ', total_samples)
  data_fraction = (1/num_of_clients)*np.ones(num_of_clients)

  # Randomly split the dataset into groups
  generator1 = torch.Generator().manual_seed(seed)
  grouped_data = random_split(train_data, data_fraction, generator=generator1)
  dataloaders = [DataLoader(group, batch_size=batch_size, shuffle=True) for group in grouped_data]
  clients_num_sample = np.zeros(num_of_clients)
  for i, group in enumerate(grouped_data):
    clients_num_sample[i] = len(group)

  return dataloaders, clients_num_sample
#---------------------------------------------------------------------------
def NonIIDDistribution(train_data, num_of_clients, batch_size, seed, labels_per_client=2):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Access labels
  try:
    targets = train_data.targets
  except AttributeError:
    targets = train_data.labels
  if isinstance(targets, torch.Tensor):
    targets = targets.numpy()

  targets = np.array(targets)
  num_classes = len(np.unique(targets))

  # Build a dictionary: label -> list of indices
  label2indices = defaultdict(list)
  for idx, label in enumerate(targets):
    label2indices[label].append(idx)

  for label in label2indices:
    random.shuffle(label2indices[label])

  # Assign labels to clients
  client_indices = [[] for _ in range(num_of_clients)]

  m1= int(num_classes/labels_per_client) ## this should be an integer
  m2 = int(num_of_clients/(num_classes/labels_per_client)) ## this should be an integer

  label2clients = defaultdict(list)
  for i in range(m2):
    array_of_labels = np.arange(num_classes)
    for j in range(m1):
      assigned_labels = np.random.choice(array_of_labels, labels_per_client, replace=False)
      array_of_labels = np.setdiff1d(array_of_labels, assigned_labels)
      for label in assigned_labels:
        label2clients[label].append(j+i*m1)
         
  # Distribute examples
  for label, indices in label2indices.items():
    n_clients = len(label2clients[label])
    if n_clients == 0:
      continue
    chunks = np.array_split(indices, n_clients)
    for client_id, chunk in zip(label2clients[label], chunks):
      client_indices[client_id].extend(chunk.tolist())

  # Build dataloaders
  dataloaders = []
  clients_num_sample = np.zeros(num_of_clients)
  for i in range(num_of_clients):
    subset = Subset(train_data, client_indices[i])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    dataloaders.append(loader)
    clients_num_sample[i] = len(client_indices[i])

  return dataloaders, clients_num_sample

#---------------------------------------------------------------------------
def MNIST_PROCESS(num_of_clients, batch_size, datasplit, seed):
  transform = transforms.Compose([transforms.ToTensor(), torch.flatten])
  train_data = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform )
  test_data = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform )

  train_loader= DataLoader(train_data, batch_size= 100) 
  test_loader= DataLoader(test_data, batch_size= 100) 

  if datasplit == 'non-iid':
    dataloaders, clients_num_sample = NonIIDDistribution(train_data, num_of_clients, batch_size, seed, labels_per_client=2)
  elif datasplit == 'iid':
    dataloaders, clients_num_sample = IIDDistribution(train_data, num_of_clients, batch_size, seed)

  return train_loader, test_loader, dataloaders, clients_num_sample

#---------------------------------------------------------------------------
def CIFAR10_PROCESS(num_of_clients, batch_size, datasplit, seed):
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  train_data = datasets.CIFAR10('./CIFAR10', train=True, download=True, transform=transform_train)
  test_data = datasets.CIFAR10('./CIFAR10', train=False, download=True, transform=transform_test)

  train_loader= DataLoader(train_data, batch_size= 100) 
  test_loader= DataLoader(test_data, batch_size= 100) 


  if datasplit == 'non-iid':
    dataloaders, clients_num_sample = NonIIDDistribution(train_data, num_of_clients, batch_size, seed, labels_per_client=2)
  elif datasplit == 'iid':
    dataloaders, clients_num_sample = IIDDistribution(train_data, num_of_clients, batch_size, seed)

  for loader in dataloaders:
    print(count_classes_in_dataloader(loader, num_classes=10))
    print(len(loader.dataset))

  return train_loader, test_loader, dataloaders, clients_num_sample
#---------------------------------------------------------------------------
def ImageNette_PROCESS(num_of_clients, batch_size, datasplit, seed):
  transform = [transforms.Resize((160, 160)),
                    transforms.ToTensor(),]
       
  train_data = datasets.ImageFolder(root='./imagenette2-160/train', 
                                         transform=transforms.Compose(transform))
  test_data = datasets.ImageFolder(root='./imagenette2-160/val', 
                                        transform=transforms.Compose(transform))
  
  train_loader= DataLoader(train_data, batch_size= 32) 
  test_loader= DataLoader(test_data, batch_size= 32) 

  if datasplit == 'non-iid':
    dataloaders, clients_num_sample = NonIIDDistribution(train_data, num_of_clients, batch_size, seed, labels_per_client=2)
  elif datasplit == 'iid':  
    dataloaders, clients_num_sample = IIDDistribution(train_data, num_of_clients, batch_size, seed)

  for loader in dataloaders:
    print(count_classes_in_dataloader(loader, num_classes=10))
    print(len(loader.dataset))

  return train_loader, test_loader, dataloaders, clients_num_sample
#---------------------------------------------------------------------------