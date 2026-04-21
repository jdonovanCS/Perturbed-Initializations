import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
import helper_hpc as helper



class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None, diversity=None, lr=5e-5, bn=True, log_activations=False, use_scheduler=False):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes

        self.classnames = classnames
        self.diversity = diversity
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

        if bn:
            self.model = models.vgg16_bn(pretrained=False, num_classes=self.num_classes)
        else:
            self.model = models.vgg16(pretrained=False, num_classes=self.num_classes)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes)

        # self.activations = {}
        # for i in range(len(self.conv_layers)):
        #     self.activations[i] = []
        
        self.use_scheduler=use_scheduler
        
        self.log_activations = log_activations
        if self.log_activations:
            self.activations={}
            count = 0
            for i, m in enumerate(self.model.features.modules()):
                if isinstance(m, (torch.nn.Conv2d)):
                    self.model.features[i].register_forward_hook(self.get_activation(count))
                    self.activations[count] = []
                    count += 1

    def get_activation(self, name):
        def hook(model, input, output):
            # trying out detach to see if it allows for 
            # computation without CUDA out of memory
            # self.activations[name].append(output.detach().cpu())
            self.activations[name].append(output)
        return hook

    def forward(self, x, get_activations=False):
        # count = 0
        # prev_m = None
        # if get_activations:
        #     for m in self.model.modules():
        #         print('shape of input:', x.shape)
        #         print('is conv2d:', isinstance(m, (torch.nn.Conv2d)))
        #         print(m.named_parameters)
        #         x = m(x)
        #         if isinstance(prev_m, (torch.nn.Conv2d)) and (isinstance(m, (torch.nn.ReLU)) or m is None):
        #             self.activations[count].append(x)
        #             count += 1
        #         prev_m = m
        #     return x
                    
        # else:
        
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.log_activations:
            for i in range(len(self.activations)):
                self.activations[i] = []

        logits = self.forward(x, get_activations=self.log_activations)

        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(logits, y)

        # log loss and acc
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        if self.log_activations:
            for i in self.activations:
                self.log('activation_{}'.format(i+1), torch.stack(self.activations[i]).flatten().mean())
        if self.log_activations:
            for i in self.activations:
                cov_matrix = self.get_activation_covariance(torch.cat(self.activations[i]))
                C = cov_matrix.shape[0]
                off_diag = cov_matrix[~torch.eye(C, dtype=bool)]
                mean_cov = off_diag.mean().item()
                self.log('activation_map_covariance{}'.format(i+1), mean_cov)
                v = torch.diag(cov_matrix)
                stddev = torch.sqrt(v + 1e-8)
                corr_matrix = cov_matrix / stddev[:, None] / stddev[None, :]
                off_diag_corr = corr_matrix[~torch.eye(C, dtype=bool)].abs().mean().item()
                self.log('activation_map_correlation{}'.format(i+1), off_diag_corr)
        if self.log_activations:
            for i in self.activations:
                cosine_dist_matrix = self.get_activation_cosine_distance(torch.cat(self.activations[i]))
                C = cosine_dist_matrix.shape[0]
                off_diag = cosine_dist_matrix[~torch.eye(C, dtype=bool)]
                mean_cosine_distance = off_diag.mean().item()
                self.log('activation_map_cosine_distance{}'.format(i+1), mean_cosine_distance)
        batch_dictionary={
	            "train_loss": loss, "train_acc": self.train_acc, 'loss': loss
	        }
        if self.log_activations:
            for i in range(len(self.activations)):
                self.activations[i] = []
        return batch_dictionary

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        
        self.log('train_loss_epoch', avg_loss, sync_dist=True)
        self.log('train_acc_epoch', self.train_acc, sync_dist=True)
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            
            # I think the hook still happens in validation,
            # but I'm not clearing it out causing CUDA OOM(?)
            if self.log_activations:
                for i in range(len(self.activations)):
                    self.activations[i] = []
            
            logits = self.forward(x)

            loss = self.cross_entropy_loss(logits, y)

            self.valid_acc(logits, y)

            self.log('val_loss', loss)
            self.log('val_acc', self.valid_acc)
            batch_dictionary = {'val_loss': loss, 
                                'val_acc': self.valid_acc
                                }
        return batch_dictionary
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss, sync_dist=True)
        self.log('val_acc_epoch', self.valid_acc, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        with torch.no_grad():
            x, y = test_batch
            
            # I think the hook still happens in validation,
            # but I'm not clearing it out causing CUDA OOM(?)
            if self.log_activations:
                for i in range(len(self.activations)):
                    self.activations[i] = []
            
            logits = self.forward(x)

            loss = self.cross_entropy_loss(logits, y)

            self.test_acc(logits, y)

            self.log('test_loss', loss)
            self.log('test_acc', self.test_acc)
            batch_dictionary = {'test_loss': loss, 
                                'test_acc': self.test_acc
                                }
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},}
        else:
            return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)
    
    # def get_fitness(self, batch):
    #     with torch.no_grad():
    #         x, y = batch
    #         logits = self.forward(x, get_activations=True)
    #         novelty_score = self.compute_feature_novelty()
    #         # clear out activations
    #         for i in range(len(self.conv_layers)):
    #             self.activations[i] = []
    #     return novelty_score

    def set_filters(self, filters):
        count = 0
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv2d)):
                z = torch.tensor(filters[count])
                z = z.type_as(m.weight.data)
                m.weight.data = z
                count += 1

    def get_activation_covariance(self, activations):
        return helper.get_activation_covariance(activations)

    def get_activation_cosine_distance(self, activations):
        return helper.get_activation_cosine_distance(activations)
    
    # def get_filters(self, numpy=False):
    #     if numpy:
    #         return [m.weight.data.detach().cpu().numpy() for m in self.conv_layers]
    #     return [m.weight.data.detach().cpu() for m in self.conv_layers]

    # def get_features(self, numpy=False):
    #     if numpy:
    #         return [self.activations[a][0] for a in range(len(self.activations))]
    #     return [self.activations[a][0] for a in range(len(self.activations))]
    
    # def compute_activation_dist(self):
    #     activations = self.get_features(numpy=True)
    #     return helper.get_dist(activations)
    
    # def compute_weight_dist(self):
    #     weights = self.get_filters(True)
    #     return helper.get_dist(weights)

    # def compute_feature_novelty(self):
        
    #     # start = time.time()
    #     # layer_totals = {}
    #     # with torch.no_grad():
    #     #     # for each conv layer 4d (batch, channel, h, w)
    #     #     for layer in range(len(self.activations)):
    #     #         B = len(self.activations[layer][0])
    #     #         C = len(self.activations[layer][0][0])
    #     #         a = self.activations[layer][0]
    #     #         layer_totals[layer] = torch.abs(a.unsqueeze(2) - a.unsqueeze(1)).sum().item()
    #     # end = time.time()
    #     # print('gpu answer: {}'.format(sum(layer_totals.values())))
    #     # print('gpu time: {}'.format(end-start))
    #     # return(sum(layer_totals.values()))

    #         # layer_totals[layer] = np.abs(np.expand_dims(a, axis=2) - np.expand_dims(a, axis=1)).sum().item()

    #     if self.diversity == None:
    #         return 0
    #     l = []
    #     for i in self.activations:
    #         print(len(self.activations[i]))
    #         if len(self.activations[i]) == 0:
    #             continue
    #         if type(self.activations[i][0]) == torch.Tensor:
    #             self.activations[i][0] = self.activations[i][0].detach().cpu().numpy()
    #         if self.diversity['type']=='relative':
    #             l.append(helper.diversity_relative(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='original':
    #             l.append(helper.diversity_orig(self.activations[i], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='absolute':
    #             l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='cosine':
    #             l.append(helper.diversity_cosine_distance(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type'] == 'constant':
    #             l.append(helper.diversity_constant(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         else:
    #             l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))

    #     if self.diversity['ldop'] == 'sum':
    #         return(sum(l))
    #     elif self.diversity['ldop'] == 'mean':
    #         return(np.mean(l))
    #     elif self.diversity['ldop'] == 'w_mean':
    #         total_channels = 0
    #         for i in range(len(self.conv_layers)):
    #             total_channels+=self.conv_layers[i].out_channels
    #         return(np.sum([l[i]*(self.conv_layers[i].out_channels)/total_channels for i in range(len(l))]))
