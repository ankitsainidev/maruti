# from torch_lr_finder import LRFinder
from tqdm.auto import tqdm
from functools import partial
import torch
import time
import numpy as np
from collections import Counter
from torchvision import transforms as torch_transforms
from . import callback as mcallback
tqdm_nl = partial(tqdm, leave=False)

__all__ = ['unfreeze', 'freeze', 'unfreeze_layers', 'freeze_layers', 'Learner']

def_norm = torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])


def children_names(model):
    return set([child[0] for child in model.named_children()])


def apply_method(model, method):
    for param in model.parameters():
        param.requires_grad = True if method == 'unfreeze' else False


def unfreeze(model):
    apply_method(model, 'unfreeze')


def freeze(model):
    apply_method(model, 'freeze')


def apply_recursively(model, layer_dict, method):
    if layer_dict is None:
        apply_method(model, method)
    else:
        memo = set()
        for name, child in model.named_children():
            if name in layer_dict:
                memo.add(name)
                apply_recursively(child, layer_dict[name], method)
        for name, parameter in model.named_parameters():
            if name in layer_dict and name not in memo:
                parameter.requires_grad = True if method == 'unfreeze' else False


def _dict_from_layers(layers):
    if layers is None:
        return {None}

    splitted = [layer.split('.') for layer in layers]
    childs = [split[0] for split in splitted]
    child_count = Counter(childs)
    layer_dict = {child: {} for child in child_count}
    none_layers = set()
    for split in splitted:
        if len(split) == 1:
            none_layers.add(split[0])
        else:
            layer_dict[split[0]] = {**layer_dict[split[0]],
                                    **_dict_from_layers(split[1:]), }
    for none_layer in none_layers:
        layer_dict[none_layer] = None
    return layer_dict


def freeze_layers(model: 'torch.nn Module', layers: 'generator of layer names'):
    apply_recursively(model, _dict_from_layers(layers), 'freeze')


def unfreeze_layers(model: 'torch.nn Module', layers: 'generator of layer names'):
    apply_recursively(model, _dict_from_layers(layers), 'unfreeze')


def _limit_string(string, length):
    string = str(string)
    if length > len(string):
        return string
    else:
        return string[:length - 2] + '..'


def _time_rep(seconds):
    if seconds >= 3600:
        return time.strftime('%H:%M:%S', time.gmtime(seconds))
    else:
        return time.strftime('%M:%S', time.gmtime(seconds))


class Learner:
    def __init__(self, model):
        self.model = model
        self.call_count = 0
        self.record = mcallback.Recorder()

    def compile(self, optimizer, loss, lr_scheduler=None,
                device='cpu', metrics=None, callback=mcallback.Callback(), max_metric_prints=3):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics_plimit = max_metric_prints
        self.device = device
        self.cb = mcallback.Compose([callback, self.record])
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = []

    def state_dict(self):
        if not hasattr(self, 'optimizer'):
            print('You first need to compile the learner')
            return

        state = {
            'record': self.record.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if hasattr(self, 'lr_scheduler'):
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        return state

    def load_state_dict(self, state):
        """Return True if everything wents write. Else raises error or returns False."""
        if not hasattr(self, 'optimizer'):
            print('Compile with earlier settings.')
            return False
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])
        self.record.load_state_dict(state['record'])
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        else:
            if 'lr_scheduler' in state:
                print(
                    'lr_scheduler is missing. Recommended to compile with same settings.')
                return False
        return True

    @property
    def header_str(self):
        header_string = ''
        # loss
        headings = ['Train Loss', 'Val Loss']

        # metrics
        for i in range(len(self.metrics)):
            headings.append(self.metrics[i].__name__)
            if i == self.metrics_plimit:
                break

        # time
        headings.append('Time')

        # getting together
        for heading in headings:
            header_string += _limit_string(heading, 12).center(12) + '|'

        return header_string

    @property
    def epoch_str(self):
        info = self.record.last_summary
        info_string = ''
        info_vals = [info['train_loss'], info['val_loss']
                     if 'val_loss' in info else None]
        for i in range(len(self.metrics)):
            info_vals.append(info['val_metrics'][self.metrics[i].__name__])
            if i == self.metrics_plimit:
                break
        info_vals.append(_time_rep(info['time']))
        for info_val in info_vals:
            if isinstance(info_val, int):
                info_val = round(info_val, 5)
            info_string += _limit_string(info_val, 12).center(12) + '|'

        return info_string

    @property
    def summary_str(self):
        total_time = sum(
            map(lambda x: x['time'], self.record.summaries))
        best_score = self.record.best_score
        return f'Total Time: {_time_rep(total_time)}, Best Score: {best_score}'

    def execute_metrics(self, ypred, y):
        metric_vals = {}
        for metric in self.metrics:
            # TODO: make better handling of non_scalar metrics
            metric_vals[metric.__name__] = metric(ypred, y).item()
        return metric_vals

    def fit(self, epochs, train_loader, val_loader=None, accumulation_steps=1, save_on_epoch='learn.pth', min_validations=0):
        # TODO: test for model on same device
        # Save_on_epoch = None or False to stop save, else path to save
        min_validation_idx = set(np.linspace(
            0, len(train_loader), min_validations + 1, dtype=int)[1:])

        self.call_count += 1

        print(self.header_str)

        # train
        self.optimizer.zero_grad()
        if self.cb.on_train_start(epochs):
            return
        for epoch in tqdm_nl(range(epochs)):
            epoch_predictions = []
            epoch_targets = []
            if self.cb.on_epoch_start(epoch):
                return

            self.model.train()

            start_time = time.perf_counter()
            train_length = len(train_loader)

            for i, (inputs, targets) in tqdm_nl(enumerate(train_loader), total=train_length, desc='Training: '):
                if self.cb.on_batch_start(epoch, i):
                    return
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                pred = self.model(inputs)
                loss = self.loss(pred, targets)

                # logging
                epoch_predictions.append(pred.clone().detach())
                epoch_targets.append(targets.clone().detach())
                batch_metrics = self.execute_metrics(pred, targets)

                #

                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                batch_extras = {'optimizer': self.optimizer, }
                if hasattr(self, 'lr_scheduler'):
                    batch_extras['lr_scheduler'] = self.lr_scheduler

                if self.cb.on_batch_end(loss.item(), batch_metrics, batch_extras, epoch, i):
                    return
                if val_loader is not None:

                    if i in min_validation_idx:
                        del inputs
                        del targets
                        if self.cb.on_min_val_start(epoch, i):
                            return
                        min_val_loss, min_val_metrics = self._validate(
                            val_loader)
                        min_val_extras = {'model': self.model}
                        if self.cb.on_min_val_end(min_val_loss, min_val_metrics, min_val_extras, epoch, i):
                            return
                        self.model.train()

            epoch_predictions = torch.cat(epoch_predictions)
            epoch_targets = torch.cat(epoch_targets)
            train_loss = self.loss(
                epoch_predictions, epoch_targets).clone().detach().item()
            train_metrics = self.execute_metrics(
                epoch_predictions, epoch_targets)
            losses = {'train': train_loss}
            metrics = {'train': train_metrics}
            if val_loader is not None:
                if self.cb.on_validation_start(epoch):
                    return
                val_loss, val_metrics = self._validate(val_loader)
                losses['val'] = val_loss
                metrics['val'] = val_metrics
                if self.cb.on_validation_end(val_loss, val_metrics, epoch):
                    return

            if save_on_epoch:
                torch.save(self.state_dict(), save_on_epoch)

            epoch_extra_dict = {'time': time.perf_counter() - start_time,
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer,
                                }
            if hasattr(self, 'lr_scheduler'):
                epoch_extra_dict['lr_scheduler'] = self.lr_scheduler
            if self.cb.on_epoch_end(losses, metrics, epoch_extra_dict, epoch):
                return
            # this should after the epoch_end callback to be ready
            tqdm.write(self.epoch_str)
        print(self.summary_str)

    def predict(self, data_loader, with_targets=True):
        self.model.eval()
        prediction_ar = []
        target_ar = []

        with torch.no_grad():
            if with_targets:
                for inputs, targets in tqdm_nl(data_loader, desc='Predicting: '):
                    inputs, targets = inputs.to(
                        self.device), targets.to(self.device)
                    pred = self.model(inputs)
                    prediction_ar.append(pred)
                    target_ar.append(targets)
                return torch.cat(prediction_ar), torch.cat(target_ar)

            for inputs in tqdm_nl(data_loader, desc='Predicting: '):
                inputs = inputs.to(self.device)
                pred = self.model(inputs)
                prediction_ar.append(pred)
            return torch.cat(prediction_ar)

    def validate(self, val_loader):
        self.call_count += 1
        return self._validate(val_loader)

    def _validate(self, val_loader):
        pred, target = self.predict(val_loader)
        loss = self.loss(pred, target).clone().detach().item()
        metrics = self.execute_metrics(pred, target)
        return loss, metrics
