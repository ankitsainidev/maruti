# from torch_lr_finder import LRFinder
from tqdm.auto import tqdm
from functools import partial
import torch
import time

tqdm_nl = partial(tqdm, leave=False)


class Callback:
    pass


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

    def compile(self, optimizer, loss, lr_scheduler=None, device='cpu', metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = []

    def fit(self, epochs, train_loader, val_loader=None, accumulation_steps=1):
        # TODO: test for model on same device
        best_loss = float('inf')
        each_train_info = []
        each_val_info = []
        complete_info = {}
        header_string = ''
        headings = ['Train Loss', 'Val Loss']
        for i in range(len(self.metrics)):
            headings.append(self.metrics[i].__name__)
            if i == 2:
                break

        for heading in headings:
            header_string += _limit_string(heading, 12).center(12) + '|'
        header_string += 'Time'.center(12) + '|'
        print(header_string)

        # train
        self.optimizer.zero_grad()

        for epoch in tqdm_nl(range(epochs)):

            self.model.train()
            train_info = {}
            val_info = {}
            train_info['losses'] = []

            start_time = time.perf_counter()
            train_length = len(train_loader)

            for i, (inputs, targets) in tqdm_nl(enumerate(train_loader), total=train_length, desc='Training: '):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                pred = self.model(inputs)
                loss = self.loss(pred, targets)
                train_info['losses'].append(loss)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
            train_info['time'] = time.perf_counter() - start_time

            if val_loader is not None:
                val_info = self.validate(val_loader)
            info_string = ''

            def format_infos(x, length):
                return _limit_string(round(torch.stack(x).mean().item(), 2), 12).center(12)
            info_values = [format_infos(train_info['losses'], 12)]

            if 'losses' in val_info:
                info_values.append(format_infos(val_info['losses'], 12))
                if torch.stack(val_info['losses']).mean().item() < best_loss:
                    complete_info['best_state_dict'] = self.model.state_dict()
            else:
                if torch.stack(train_info['losses']).mean().item() < best_loss:
                    complete_info['best_state_dict'] = self.model.state_dict()
                info_values.append(str(None).center(12))

            for i, metric in enumerate(self.metrics):
                info_values.append(format_infos(
                    val_info['metrics'][metric.__name__], 12))
                if i == 2:
                    break
            total_time = train_info['time']
            if 'time' in val_info:
                total_time += val_info['time']
            info_values.append(_time_rep(total_time).center(12))

            tqdm.write('|'.join(info_values) + '|')

            each_train_info.append(train_info)
            each_val_info.append(val_info)
        complete_info = {**complete_info,
                         'train': each_train_info, 'val': each_val_info}
        return complete_info

    def validate(self, val_loader):
        information = {}
        information['losses'] = []
        information['metrics'] = {}
        for metric in self.metrics:
            information['metrics'][metric.__name__] = []

        self.model.eval()
        val_loss = torch.zeros(1)
        start_time = time.perf_counter()
        with torch.set_grad_enabled(False):
            for inputs, targets in tqdm_nl(val_loader, desc='Validating: '):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                pred = self.model(inputs)
                loss = self.loss(pred, targets)
                information['losses'].append(loss)
                for metric in self.metrics:
                    information['metrics'][metric.__name__].append(
                        metric(pred, targets))

        total_time = time.perf_counter() - start_time
        information['time'] = total_time
        return information
