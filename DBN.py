"""This file contains the implementation of a Deep Belief Network, stacking several Restricted Boltzmann Machines
implemented in RBM.py."""
import torch
import torch.nn as nn
from RBM import RBM


class DBN(nn.Module):
    """Class implementing a DBN using the basic RBM class."""
    def __init__(self,
                 visible_units=256,
                 hidden_units=[64, 100],
                 k=2,
                 learning_rate=1e-5,
                 learning_rate_decay=1.0,
                 weight_decay=.0002,
                 initial_momentum=.5,
                 final_momentum=.9,
                 xavier_init=False,
                 increase_to_cd_k=False,
                 use_gpu=False):
        super(DBN, self).__init__()

        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.n_layers = len(hidden_units)
        self.rbm_layers = nn.ModuleList()

        # Creating different RBM layers
        for i in range(self.n_layers):
            input_size = 0
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size,
                      hidden_units=hidden_units[i],
                      k=k,
                      learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      weight_decay=weight_decay,
                      initial_momentum=initial_momentum,
                      final_momentum=final_momentum,
                      xavier_init=xavier_init,
                      increase_to_cd_k=increase_to_cd_k,
                      use_gpu=use_gpu)

            self.rbm_layers.append(rbm)

        # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [
            nn.Parameter(self.rbm_layers[i].W.data.clone())
            for i in range(self.n_layers - 1)
        ]
        self.W_gen = [
            nn.Parameter(self.rbm_layers[i].W.data)
            for i in range(self.n_layers - 1)
        ]
        self.bias_rec = [
            nn.Parameter(self.rbm_layers[i].h_bias.data.clone())
            for i in range(self.n_layers - 1)
        ]
        self.bias_gen = [
            nn.Parameter(self.rbm_layers[i].v_bias.data)
            for i in range(self.n_layers - 1)
        ]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)

        for i in range(self.n_layers - 1):
            self.register_parameter('W_rec%i' % i, self.W_rec[i])
            self.register_parameter('W_gen%i' % i, self.W_gen[i])
            self.register_parameter('bias_rec%i' % i, self.bias_rec[i])
            self.register_parameter('bias_gen%i' % i, self.bias_gen[i])

    def forward(self, input_data):
        """running the forward pass
        do not confuse with training this just runs a forward pass

        :param input_data:
        """
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0], -1))  # flatten
            p_v, v = self.rbm_layers[i].to_hidden(v)
        return p_v, v

    def reconstruct(self, input_data):
        """go till the final layer and then reconstruct

        :param input_data:
        """
        p_h = input_data
        for i in range(len(self.rbm_layers)):
            p_h = p_h.view((p_h.shape[0], -1)).float().to(self.device)  # flatten
            p_h, h = self.rbm_layers[i].to_hidden(p_h)

        p_v = p_h
        for i in range(len(self.rbm_layers) - 1, -1, -1):
            p_v = p_v.view((p_v.shape[0], -1)).float().to(self.device)
            p_v, v = self.rbm_layers[i].to_visible(p_v)
        return p_v, v

    def train_static(self,
                     train_data,
                     train_loader,
                     num_epochs=50,
                     batch_size=10,
                     transform=None):
        """Greedy Layer By Layer training
        Keeping previous layers as static

        :param train_data: 
        :param train_loader: DataLoader for the first layer's training data
        :param num_epochs:  (Default value = 50)
        :param batch_size:  (Default value = 10)
        :param transform: optional callable (e.g. a torchvision transforms composition)
                          applied per-sample to the raw input during the first layer's
                          training. Subsequent layers train on activations and are not
                          augmented. (Default value = None)
        """
        tmp = train_data
        current_loader = train_loader

        for i in range(len(self.rbm_layers)):
            print("-" * 20)
            print("Training RBM layer {}".format(i + 1))

            layer_transform = transform if i == 0 else None
            self.rbm_layers[i].fit(current_loader, num_epochs, batch_size, layer_transform)

            v = tmp.view((tmp.shape[0], -1)).float()
            if self.rbm_layers[i].use_gpu:
                v = v.cuda()
            p_v, v = self.rbm_layers[i].forward(v)
            tmp = p_v.detach()

            # Build a new DataLoader for the next layer using transformed data.
            # Placeholder labels are required by DataLoader but unused during RBM training.
            if i < len(self.rbm_layers) - 1:
                placeholder_labels = torch.zeros(tmp.shape[0])
                next_dataset = torch.utils.data.TensorDataset(tmp.cpu(), placeholder_labels)
                current_loader = torch.utils.data.DataLoader(next_dataset,
                                                             batch_size=batch_size)
        return

    def train_ith(self, train_data, train_labels, num_epochs, batch_size,
                  ith_layer, transform=None):
        """taking ith layer at once
        can be used for fine tuning

        :param train_data: 
        :param train_labels: 
        :param num_epochs: 
        :param batch_size: 
        :param ith_layer:
        :param transform: optional callable (e.g. a torchvision transforms composition)
                          applied per-sample to the raw input before passing through
                          preceding layers. (Default value = None)
        """
        if (ith_layer - 1 > len(self.rbm_layers) or ith_layer <= 0):
            print("Layer index out of range")
            return
        ith_layer = ith_layer - 1

        if transform is not None:
            v = torch.stack([transform(sample) for sample in train_data]).float()
        else:
            v = train_data.float()
        v = v.view((v.shape[0], -1))

        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].forward(v)

        tmp = v
        tensor_x = tmp.float()  # transform to torch tensors
        tensor_y = train_labels.float()
        _dataset = torch.utils.data.TensorDataset(
            tensor_x, tensor_y)  # create your datset
        _dataloader = torch.utils.data.DataLoader(_dataset,
                                                  batch_size=batch_size,
                                                  drop_last=True)
        self.rbm_layers[ith_layer].fit(_dataloader, num_epochs, batch_size)
        return
