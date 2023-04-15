import torch
import torch.nn as nn
#import function call


class MoaDataset:
  def __init__(self, features, targets): #features should be changes to images in the long run
    self.features = features #both numpy arrays for now, should be the same for image
    self.targets = targets
  def __len__(self):
    return self.features.shape[0]
  def __getitem__(self, item):
    return {
        "x": torch.tensor(self.features[item, :], dtype = torch.float), #returns features for a given item, should return an image(s)? in the future
        "y": torch.tensor(self.targets[item, :], dtype = torch.float),
    }

class Engine:
  def __init__(self, model, optimizer, device):
    self.model = model
    self.device = device
    self.optimizer = optimizer

  @staticmethod
  def loss_fn(targets, outputs):
    return nn.BCEWithLogitsLoss()(outputs, targets)

  def train(self, data_loader):
    self.model.train()
    final_loss = 0
    for data in data_loader:
      self.optimizer.zero_grad()
      inputs = data["x"].to(self.device)
      targets = data["y"].to(self.device)
      outputs = self.model(inputs)
      loss = self.loss_fn(targets, outputs)
      loss.backward()
      self.optimizer.step()
      final_loss += loss.item()
    return final_loss / len(data_loader)

  def evaluate(self, data_loader):
    self.model.eval()
    final_loss = 0
    for data in data_loader:
      inputs = data["x"].to(self.device)
      targets = data["y"].to(self.device)
      outputs = self.model(inputs)
      loss = self.loss_fn(targets, outputs)
      final_loss += loss.item()
    return final_loss / len(data_loader)

class Model(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
      super().__init__()
      layers = []
      for _ in range(nlayers):
        if len(layers) == 0:
          layers.append(defconv(nfeatures, hidden_size))
        else:
          layers.append(downpass(hidden_size*(pow(2,_)), hidden_size*(pow(2,_+1)))
      dim = hidden_size*(pow(2,nlayers))
      layers.append(ConvLSTM(input_s=(8,16),
                             input_dim=dim,
                             hidden_dim=[dim, dim],
                             kernel_size=(2,2),
                             num_layers=2,
                             batch_first=False,
                             bias=True,
                             return_all_layers=False))

      for _ in range(nlayers):
        layers.append(uppass(2*(hidden_size*pow(2,nlayers-_)), hidden_size*pow(2,nlayers-(_+1)))

      layer.append(nn.Conv2d(hidden_size, ntargets, 1))

      layers.append(nn.Linear(ntargets, ntargets))
      self.model = nn.Sequential(*layers)

    def forward(self, x):
      return self.model(x)