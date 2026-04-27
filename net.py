import gc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper_hpc as helper
import time
import torchmetrics

# DEFINE a CONV NN

class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None, diversity=None, lr=.001, bn=True, data_dims=(3,32,32), log_activations=False, use_scheduler=False):
        super().__init__()
        self.save_hyperparameters()
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.BatchNorm2 = nn.BatchNorm2d(128)
        self.BatchNorm3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        # x4 because I have 2 pools in forward
        self.fc1 = nn.Linear(data_dims[1]*data_dims[2]*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.1)
        self.conv_layers = nn.ModuleList([nn.Conv2d(3, 32, 3, padding=1), 
                            nn.Conv2d(32, 64, 3, padding=1), 
                            nn.Conv2d(64, 128, 3, padding=1), 
                            nn.Conv2d(128, 128, 3, padding=1), 
                            nn.Conv2d(128, 256, 3, padding=1), 
                            nn.Conv2d(256, 256, 3, padding=1)])
        
        self.log_activations = log_activations
        self.use_scheduler=use_scheduler
        
        self.activations = {}
        self.activations_after_nonlinearity = {}
        self.layer_metrics = torch.nn.ModuleDict()
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
            self.activations_after_nonlinearity[i] = []
            self.layer_metrics[f"activation_{i}"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_correlation_{i}"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_covariance_{i}"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_cosine_distance_{i}"] = torchmetrics.MeanMetric() 
            self.layer_metrics[f"activation_{i}_afterRELU"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_correlation_{i}_afterRELU"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_covariance_{i}_afterRELU"] = torchmetrics.MeanMetric()
            self.layer_metrics[f"activation_cosine_distance_{i}_afterRELU"] = torchmetrics.MeanMetric() 


        self.classnames = classnames
        self.diversity = diversity
        self.lr = lr
        self.bn = bn
        
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.valid_per_class_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="none")
        self.novelty_score = torchmetrics.MeanMetric()
        self.avg_novelty = 0
        self.global_step_count=0
        self.global_val_step_count=0


    def forward(self, x, get_activations=False, get_activations_after_nonlinearity=False):
        conv_count = 0
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        if self.bn:
            x = self.BatchNorm1(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        conv_count += 1
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        conv_count += 1
        x = self.pool(x)
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        if self.bn:
            x = self.BatchNorm2(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        conv_count += 1
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        conv_count += 1
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        if self.bn:
            x = self.BatchNorm3(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        conv_count += 1
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        x = F.relu(x)
        if get_activations_after_nonlinearity:
            self.activations_after_nonlinearity[conv_count].append(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.log_activations:
            for i in range(len(self.conv_layers)):
                self.activations_after_nonlinearity[i] = []
                self.activations[i] = []
                
        logits = self.forward(x, get_activations=self.log_activations, get_activations_after_nonlinearity=self.log_activations)
        # get loss
        loss = self.cross_entropy_loss(logits, y)
        # get acc
        self.train_acc(logits, y)
        self.train_loss(loss)
        
        # labels_hat = torch.argmax(logits, 1)
        # acc = torch.sum(y==labels_hat)/(len(y)*1.0)
        # log loss and acc
        
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, batch_size=len(y))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, batch_size=len(y))
        self.log('step_count', self.global_step_count)
        self.global_step_count+=1
        if self.log_activations:
            for i in range(len(self.conv_layers)):
                # --------------- BEFORE RELU -----------------
                # calculate and log activation map scalar
                mean_activation = torch.stack(self.activations[i]).flatten().mean()
                self.layer_metrics[f"activation_{i}"](mean_activation)
                self.log(f'activation_{i+1}', self.layer_metrics[f"activation_{i}"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_{i+1}', mean_activation, on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map covariance
                cov_matrix, mean_cov = self.get_activation_covariance(torch.cat(self.activations[i]))
                self.layer_metrics[f"activation_covariance_{i}"](mean_cov)
                self.log(f'activation_map_covariance{i+1}', self.layer_metrics[f"activation_covariance_{i}"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_covariance{i+1}', mean_cov, on_step=True, on_epoch=True, batch_size=len(y))
                
                #calculate and log activation map pearson correlation
                off_diag_corr = self.get_activation_correlation(cov_matrix)
                self.layer_metrics[f"activation_correlation_{i}"](off_diag_corr)
                self.log(f'activation_map_correlation{i+1}', self.layer_metrics[f"activation_correlation_{i}"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_correlation{i+1}', off_diag_corr, on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map cosine distance
                mean_cosine_distance = self.get_activation_cosine_distance(torch.cat(self.activations[i]))
                self.layer_metrics[f"activation_cosine_distance_{i}"](mean_cosine_distance)
                self.log(f'activation_map_cosine_distance{i+1}', self.layer_metrics[f"activation_cosine_distance_{i}"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_cosine_distance{i+1}', mean_cosine_distance, on_step=True, on_epoch=True, batch_size=len(y))


                # ----------- AFTER RELU -------------
                # calculate and log activation map scalar
                mean_activation_after_nonlinearity = torch.stack(self.activations_after_nonlinearity[i]).flatten().mean()
                self.layer_metrics[f"activation_{i}_afterRELU"](mean_activation_after_nonlinearity)
                self.log(f'activation_{i+1}_afterRELU', self.layer_metrics[f"activation_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_{i+1}_afterRELU', mean_activation_after_nonlinearity, on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map covariance
                cov_matrix, mean_cov = self.get_activation_covariance(torch.cat(self.activations_after_nonlinearity[i]))
                self.layer_metrics[f"activation_covariance_{i}_afterRELU"](mean_cov)
                self.log(f'activation_map_covariance{i+1}_afterRELU', self.layer_metrics[f"activation_covariance_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_covariance{i+1}_afterRELU', mean_cov, on_step=True, on_epoch=True, batch_size=len(y))
                
                #calculate and log activation map pearson correlation
                off_diag_corr = self.get_activation_correlation(cov_matrix)
                self.layer_metrics[f"activation_correlation_{i}_afterRELU"](off_diag_corr)
                self.log(f'activation_map_correlation{i+1}_afterRELU', self.layer_metrics[f"activation_correlation_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_correlation{i+1}_afterRELU', off_diag_corr, on_step=True, on_epoch=True, batch_size=len(y))

                # calculate and log activation map cosine distance
                mean_cosine_distance = self.get_activation_cosine_distance(torch.cat(self.activations_after_nonlinearity[i]))
                self.layer_metrics[f"activation_cosine_distance_{i}_afterRELU"](mean_cosine_distance)
                self.log(f'activation_map_cosine_distance{i+1}_afterRELU', self.layer_metrics[f"activation_cosine_distance_{i}_afterRELU"], on_step=True, on_epoch=True, batch_size=len(y))
                # self.log(f'activation_map_cosine_distance{i+1}_afterRELU', mean_cosine_distance, on_step=True, on_epoch=True, batch_size=len(y))
                
                return loss
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
                self.activations_after_nonlinearity[i]=[]
            x, y = val_batch
            logits = self.forward(x, get_activations=True, get_activations_after_nonlinearity=True)
            # get loss
            val_loss = self.cross_entropy_loss(logits, y)
            
            self.valid_acc(logits, y)
            self.valid_loss(val_loss)
            
            # labels_hat = torch.argmax(logits, 1)
            # val_acc = torch.sum(y==labels_hat)/(len(y)*1.0)
            
            # self.valid_per_class_acc(logits, y)
            # current_per_class = self.valid_per_class_acc.compute()
            # for i, acc in enumerate(current_per_class):
            #     self.log(f"step_acc_class_{i}", acc, on_step=True, on_epoch=True, batch_size=len(y))
            
            # get novelty score
            novelty_score = self.compute_feature_novelty()

            self.novelty_score(novelty_score)

            # log loss, acc, class acc, and novelty score
            # clear out activations
            
            self.log('val_loss', self.valid_loss, on_step=False, on_epoch=True, batch_size=len(y))
            self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True, batch_size=len(y))
            self.log('val_novelty', self.novelty_score, on_step=False, on_epoch=True, batch_size=len(y))
            # self.log('val_step_count', self.global_val_step_count, on_step=True, on_epoch=False)
            
            self.global_val_step_count+=1
            # return val_loss

    def get_fitness(self, batch):
        with torch.no_grad():
            x, y = batch
            logits = self.forward(x, get_activations=True)
            novelty_score = self.compute_feature_novelty()
            # clear out activations
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
        return novelty_score

    def test_step(self, test_batch, batch_idx):
        with torch.no_grad():
            if self.log_activations:
                for i in range(len(self.conv_layers)):
                    self.activations[i] = []
                    self.activations_after_nonlinearity[i] = []
            x, y = test_batch
            logits = self.forward(x)
            # get loss
            test_loss = self.cross_entropy_loss(logits, y)
            
            self.test_acc(logits, y)
            self.test_loss(test_loss)
            
            # labels_hat = torch.argmax(logits, 1)
            # test_acc = torch.sum(y==labels_hat)/(len(y)*1.0)
            
            # log loss, acc
            self.log('test_loss', self.test_loss, on_step=False, on_epoch=True, batch_size=len(y))
            self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=len(y))
            
            # return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},}
        else:
            return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)
        
    def set_filters(self, filters):
        for i in range(len(filters)):
            self.conv_layers[i].weight.data = filters[i]
    
    def get_filters(self, numpy=False):
        if numpy:
            return [m.weight.data.detach().cpu().numpy() for m in self.conv_layers]
        return [m.weight.data.detach().cpu() for m in self.conv_layers]

    def get_features(self, numpy=False):
        if numpy:
            return [self.activations[a][0] for a in range(len(self.activations))]
        return [self.activations[a][0] for a in range(len(self.activations))]
    
    def compute_activation_dist(self):
        activations = self.get_features(numpy=True)
        return helper.get_dist(activations)
    
    def compute_weight_dist(self):
        weights = self.get_filters(True)
        return helper.get_dist(weights)
    
    def clear_activations(self):
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
    
    def get_activations(self):
        return self.activations

    def get_activation_covariance(self, activations):
        cov_matrix = helper.get_activation_covariance(activations)
        C = cov_matrix.shape[0]
        off_diag = cov_matrix[~torch.eye(C, dtype=bool)]
        mean_cov = off_diag.abs().mean().item()
        return cov_matrix, mean_cov
    
    def get_activation_correlation(self, cov_matrix):
        C = cov_matrix.shape[0]
        v = torch.diag(cov_matrix)
        stddev = torch.sqrt(v+1e-8)
        corr_matrix=cov_matrix/stddev[:, None] / stddev[None, :]
        off_diag_corr = corr_matrix[~torch.eye(C, dtype=bool)].abs().mean().item()
        return off_diag_corr
        

    def get_activation_cosine_distance(self, activations):
        cos_dist_matrix = helper.get_activation_cosine_distance(activations)
        C = cos_dist_matrix.shape[0]
        off_diag = cos_dist_matrix[~torch.eye(C, dtype=bool)]
        mean_cosine_distance=off_diag.mean().item()
        return mean_cosine_distance

    def compute_feature_novelty(self):
        
        # start = time.time()
        # layer_totals = {}
        # with torch.no_grad():
        #     # for each conv layer 4d (batch, channel, h, w)
        #     for layer in range(len(self.activations)):
        #         B = len(self.activations[layer][0])
        #         C = len(self.activations[layer][0][0])
        #         a = self.activations[layer][0]
        #         layer_totals[layer] = torch.abs(a.unsqueeze(2) - a.unsqueeze(1)).sum().item()
        # end = time.time()
        # print('gpu answer: {}'.format(sum(layer_totals.values())))
        # print('gpu time: {}'.format(end-start))
        # return(sum(layer_totals.values()))

            # layer_totals[layer] = np.abs(np.expand_dims(a, axis=2) - np.expand_dims(a, axis=1)).sum().item()

        l = []
        for i in self.activations:
            if type(self.activations[i][0]) != type(np.zeros((1))):
                self.activations[i][0] = self.activations[i][0].detach().cpu().numpy()
            if self.diversity['type']=='relative':
                l.append(helper.diversity_relative(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='original':
                l.append(helper.diversity_orig(self.activations[i], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='absolute':
                l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='cosine':
                l.append(helper.diversity_cosine_distance(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type'] == 'constant':
                l.append(helper.diversity_constant(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            else:
                l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))

        if self.diversity['ldop'] == 'sum':
            return(sum(l))
        elif self.diversity['ldop'] == 'mean':
            return(np.mean(l))
        elif self.diversity['ldop'] == 'w_mean':
            total_channels = 0
            for i in range(len(self.conv_layers)):
                total_channels+=self.conv_layers[i].out_channels
            return(np.sum([l[i]*(self.conv_layers[i].out_channels)/total_channels for i in range(len(l))]))
