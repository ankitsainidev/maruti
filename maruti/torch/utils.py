# from torch_lr_finder import LRFinder
from tqdm.auto import tqdm
from functools import partial
import torch
import time
from collections import Counter
from torchvision import transforms as torch_transforms
from torch.utils import data
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
        self.record = {'best_model': model.state_dict,
                       'best_score': float('inf'),
                       'history': [],  # history: [[[l1,l2..],[vl1,vl2..]],[2nd epoch]...]
                       'epoch_summary': []}  # epoch_summary: [[[l.mean,vl.mean..],[2nd epoch]]]

    def compile(self, optimizer, loss, lr_scheduler=None, device='cpu', metrics=None, tot_metrics_prints=3):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics_plimit = tot_metrics_prints
        self.device = device
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = []
        for metric in self.metrics:
            metric.reset()

    def state_dict(self):
        if not hasattr(self, 'optimizer'):
            print('You need first compile the learner')
            return

        state = {
            'record': self.record,
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
        self.record = state['record']
        if hasattr(self, lr_scheduler):
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
            headings.append(self.metrics[i].name)
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
        info = self.record['epoch_summary'][-1]
        info_string = ''
        info_vals = [info['train_loss'], info['val_loss']]
        for i in range(len(self.metrics)):
            info_vals.append(info['metrics'][self.metrics[i].name])
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
            map(lambda x: x['time'], self.record['epoch_summary']))
        best_score = self.record['best_score']
        return f'Total Time: {_time_rep(total_time)}, Best Score: {best_score}'

    def update_record(self):
        last_summary = self.record['epoch_summary'][-1]
        representative_loss = 'val_loss' if 'val_loss' in last_summary else 'train_loss'
        if last_summary[representative_loss] < self.record['best_score']:
            self.record['best_score'] = last_summary[representative_loss]
            self.record['best_model'] = self.model.state_dict()

    def create_epoch_summary(self, time):
        # assert len(self.record['history']) == len(
        #     self.record['epoch_summary']) + 1
        epoch_history = self.record['history'][-1]
        epoch_sum = {'time': time}
        epoch_sum['train_loss'] = torch.stack(
            epoch_history['train_loss']).mean().item()
        if 'val_loss' in epoch_history:
            epoch_sum['val_loss'] = torch.stack(
                epoch_history['val_loss']).mean().item()
        if self.metrics:
            epoch_sum['metrics'] = {}
            for metric in self.metrics:
                epoch_sum['metrics'][metric.name] = metric.value
        self.record['epoch_summary'].append(epoch_sum)

    def fit(self, epochs, train_loader, val_loader=None, accumulation_steps=1, save_on_epoch='.', save_with_name='learner'):
        # TODO: test for model on same device
        # Save_on_epoch = None or False to stop save, else path to save
        for metric in self.metrics:
            metric.reset()
        print(self.header_str)

        # train
        self.optimizer.zero_grad()

        for epoch in tqdm_nl(range(epochs)):
            self.record['history'].append({'train_loss': []})
            self.model.train()

            start_time = time.perf_counter()
            train_length = len(train_loader)

            for i, (inputs, targets) in tqdm_nl(enumerate(train_loader), total=train_length, desc='Training: '):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                pred = self.model(inputs)
                loss = self.loss(pred, targets)
                self.record['history'][-1]['train_loss'].append(loss)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            if val_loader is not None:
                self.validate(val_loader)

            self.create_epoch_summary(time.perf_counter() - start_time)
            self.update_record()
            tqdm.write(self.epoch_str)
            if save_on_epoch is not None:
                if os.path.exists(os.path.join(save_on_epoch, name + '_' + str(i - 1) + '.pth')):
                    os.remove(os.path.join(save_on_epoch,
                                           name + '_' + str(i - 1) + '.pth'))
                torch.save(self.state_dict(), os.path.join(
                    save_on_epoch, name + '_' + str(i) + '.pth'))

        print(self.summary_str)

    def validate(self, val_loader):
        if len(self.record['history']) == 0:
            self.record['history'].append({})
        self.record['history'][-1]['val_loss'] = []
        self.record['history'][-1]['metrics'] = {}

        for metric in self.metrics:
            metric.reset()
            self.record['history'][-1]['metrics'][metric.name] = []

        self.model.eval()
        val_loss = torch.zeros(1)

        with torch.set_grad_enabled(False):
            for inputs, targets in tqdm_nl(val_loader, desc='Validating: '):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                pred = self.model(inputs)
                loss = self.loss(pred, targets)
                self.record['history'][-1]['val_loss'].append(loss)
                for metric in self.metrics:

                    self.record['history'][-1]['metrics'][metric.name].append(
                        metric(pred, targets))


# if __name__ = '__main__':
#     model = torch.nn.seq
