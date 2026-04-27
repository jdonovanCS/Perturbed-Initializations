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
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.lr = lr
        self.global_step_count = 0

        if bn:
            self.model = models.vgg16_bn(pretrained=False, num_classes=self.num_classes)
        else:
            self.model = models.vgg16(pretrained=False, num_classes=self.num_classes)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes)

        # self.activations = {}
        # for i in range(len(self.conv_layers)):
        #     self.activations[i] = []
        
        self.use_scheduler=use_scheduler
        
        self.layer_metrics=nn.ModuleDict()
        self.log_activations = log_activations
        if self.log_activations:
            self.activations={}
            self.activations_after_nonlinearity={}
            count = 0
            for i, m in enumerate(self.model.features.modules()):
                if isinstance(m, (torch.nn.Conv2d)):
                    self.model.features[i].register_forward_hook(self.get_activation(count))
                    self.activations[count] = []
                    self.layer_metrics[f"activation_{count}"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_correlation_{count}"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_covariance_{count}"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_cosine_distance_{count}"] = torchmetrics.MeanMetric() 
                if isinstance(m, (torch.nn.ReLU)):
                    self.model.features[i].register_forward_hook(self.get_activation_after_nonlinearity(count))
                    self.activations_after_nonlinearity[count] = []
                    self.layer_metrics[f"activation_{count}_afterRELU"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_correlation_{count}_afterRELU"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_covariance_{count}_afterRELU"] = torchmetrics.MeanMetric()
                    self.layer_metrics[f"activation_cosine_distance_{count}_afterRELU"] = torchmetrics.MeanMetric() 
                    count+=1

    def get_activation(self, name):
        def hook(model, input, output):
            # trying out detach to see if it allows for 
            # computation without CUDA out of memory
            # self.activations[name].append(output.detach().cpu())
            self.activations[name].append(output)
        return hook
    
    def get_activation_after_nonlinearity(self, name):
        def hook(model, input, output):
            self.activations_after_nonlinearity[name].append(output)
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
                self.activations_after_nonlinearity[i] = []

        logits = self.forward(x, get_activations=self.log_activations)

        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(logits, y)
        self.train_loss(loss)

        # log loss and acc
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, batch_size=len(y))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, batch_size=len(y))
        self.log('step_count', self.global_step_count)
        self.global_step_count+=1
        if self.log_activations:
            for i in range(len(self.activations)):
                # --------BEFORE RELU--------
                # calculate and log activation map scalar
                mean_activation = torch.stack(self.activations[i]).flatten().mean()
                self.layer_metrics[f"activation_{i}"](mean_activation)
                self.log(f'activation_{i+1}', self.layer_metrics[f"activation_{i}"], on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map covariance
                cov_matrix, mean_cov = self.get_activation_covariance(torch.cat(self.activations[i]))
                self.layer_metrics[f"activation_covariance_{i}"](mean_cov)
                self.log(f'activation_map_covariance{i+1}', self.layer_metrics[f"activation_covariance_{i}"], on_step=True, on_epoch=True, batch_size=len(y))
                
                #calculate and log activation map pearson correlation
                off_diag_corr = self.get_activation_correlation(cov_matrix)
                self.layer_metrics[f"activation_correlation_{i}"](off_diag_corr)
                self.log(f'activation_map_correlation{i+1}', self.layer_metrics[f"activation_correlation_{i}"], on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map cosine distance
                mean_cosine_distance = self.get_activation_cosine_distance(torch.cat(self.activations[i]))
                self.layer_metrics[f"activation_cosine_distance_{i}"](mean_cosine_distance)
                self.log(f'activation_map_cosine_distance{i+1}', self.layer_metrics[f"activation_cosine_distance_{i}"], on_step=True, on_epoch=True, batch_size=len(y))

                #------AFTER RELU--------
                # calculate and log activation map scalar
                mean_activation_after_nonlinearity = torch.stack(self.activations_after_nonlinearity[i]).flatten().mean()
                self.layer_metrics[f"activation_{i}_afterRELU"](mean_activation_after_nonlinearity)
                self.log(f'activation_{i+1}_afterRELU', self.layer_metrics[f"activation_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map covariance
                cov_matrix, mean_cov = self.get_activation_covariance(torch.cat(self.activations_after_nonlinearity[i]))
                self.layer_metrics[f"activation_covariance_{i}_afterRELU"](mean_cov)
                self.log(f'activation_map_covariance{i+1}_afterRELU', self.layer_metrics[f"activation_covariance_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))
                
                #calculate and log activation map pearson correlation
                off_diag_corr = self.get_activation_correlation(cov_matrix)
                self.layer_metrics[f"activation_correlation_{i}_afterRELU"](off_diag_corr)
                self.log(f'activation_map_correlation{i+1}_afterRELU', self.layer_metrics[f"activation_correlation_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map cosine distance
                mean_cosine_distance = self.get_activation_cosine_distance(torch.cat(self.activations_after_nonlinearity[i]))
                self.layer_metrics[f"activation_cosine_distance_{i}_afterRELU"](mean_cosine_distance)
                self.log(f'activation_map_cosine_distance{i+1}_afterRELU', self.layer_metrics[f"activation_cosine_distance_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))

                return loss
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            
            # I think the hook still happens in validation,
            # but I'm not clearing it out causing CUDA OOM(?)
            if self.log_activations:
                for i in range(len(self.activations)):
                    self.activations[i] = []
                    self.activations_after_nonlinearity[i] = []
            
            logits = self.forward(x, get_activations=True)

            loss = self.cross_entropy_loss(logits, y)

            self.valid_acc(logits, y)
            self.valid_loss(loss)

            self.log('val_loss', self.valid_loss, on_step=False, on_epoch=True, batch_size=len(y))
            self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True, batch_size=len(y))

    def test_step(self, test_batch, batch_idx):
        with torch.no_grad():
            x, y = test_batch
            
            # I think the hook still happens in validation,
            # but I'm not clearing it out causing CUDA OOM(?)
            if self.log_activations:
                for i in range(len(self.activations)):
                    self.activations[i] = []
                    self.activations_after_nonlinearity[i] = []
            
            logits = self.forward(x)

            loss = self.cross_entropy_loss(logits, y)

            self.test_acc(logits, y)
            self.test_loss(loss)

            self.log('test_loss', self.test_loss, on_step=False, on_epoch=True, batch_size=len(y))
            self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=len(y))

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
        cov_matrix = helper.get_activation_covariance(activations)
        C = cov_matrix.shape[0]
        off_diag = cov_matrix[~torch.eye(C, dtype=bool)]
        mean_cov = off_diag.abs().mean().item()
        return cov_matrix, mean_cov

    def get_activation_cosine_distance(self, activations):
        cos_dist_matrix = helper.get_activation_cosine_distance(activations)
        C = cos_dist_matrix.shape[0]
        off_diag = cos_dist_matrix[~torch.eye(C, dtype=bool)]
        mean_cosine_distance=off_diag.mean().item()
        return mean_cosine_distance
    
    def get_activation_correlation(self, cov_matrix):
        C = cov_matrix.shape[0]
        v = torch.diag(cov_matrix)
        stddev = torch.sqrt(v+1e-8)
        corr_matrix=cov_matrix/stddev[:, None] / stddev[None, :]
        off_diag_corr = corr_matrix[~torch.eye(C, dtype=bool)].abs().mean().item()
        return off_diag_corr

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
