import os
from datetime import datetime, timezone, timedelta
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy


class Callback:
    def on_epoch_end(self, losses, metrics, extras, epoch):
        """
        extras-> dict ['time']['model']
        """
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_batch_start(self, epoch, batch):
        pass

    def on_batch_end(self, loss, metrics, extras, epoch, batch):
        pass

    def on_validation_start(self, epoch):
        pass

    def on_validation_end(self, loss, metrics, epoch):
        pass

    def on_min_val_start(self, epoch, batch):
        pass

    def on_min_val_end(self, loss, metrics, extras, epoch, batch):
        """extras['model']"""
        pass

    def on_train_start(self, epoch):
        pass


def Compose(callbacks):
    class NewCallback(Callback):
        def on_epoch_end(self, losses, metrics, extras, epoch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_epoch_end(
                    losses, metrics, extras, epoch)
            return isEnd

        def on_epoch_start(self, epoch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_epoch_start(epoch)
            return isEnd

        def on_batch_start(self, epoch, batch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_batch_start(epoch, batch)
            return isEnd

        def on_batch_end(self, loss, metrics, extras, epoch, batch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_batch_end(loss, metrics, extras,
                                                       epoch, batch)
            return isEnd

        def on_validation_start(self, epoch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_validation_start(epoch)
            return isEnd

        def on_validation_end(self, loss, metrics, epoch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_validation_end(
                    loss, metrics, epoch)
            return isEnd

        def on_min_val_start(self, epoch, batch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_min_val_start(
                    epoch, batch)
            return isEnd

        def on_min_val_end(self, loss, metrics, extras, epoch, batch):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_min_val_end(
                    loss, metrics, extras, epoch, batch)
            return isEnd

        def on_train_start(self, epochs):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_train_start(epochs)
            return isEnd
    return NewCallback()


class Recorder(Callback):

    def __init__(self):
        self.best_model = None
        self.best_score = float('inf')
        self.summaries = []
        self.others = []
        self.prevs = []
        # to monitor if the learner was stopped in between of an epoch
        self.epoch_started = False

    def on_train_start(self, epochs):
        self.new_state()

    def new_state(self):
        sd = self.state_dict()
        del sd['prevs']
        self.prevs.append(self.state_dict())
        self.summaries = []
        self.others = []

    def on_epoch_start(self, epoch):
        if self.epoch_started:
            self.new_state()
        self.summaries.append({})
        self.others.append({'train_losses': [], 'train_metrics': []})
        self.epoch_started = True

    def on_batch_end(self, train_loss, train_metrics, extras, epoch, batch):
        self.others[epoch]['train_losses'].append(train_loss)
        self.others[epoch]['train_metrics'].append(train_metrics)

    @property
    def last_summary(self):
        if self.summaries:
            return self.summaries[-1]
        raise Exception('no summaries exists')

    def on_min_val_end(self, loss, metrics, extras, epoch, batch):
        if loss < self.best_score:
            self.best_score = loss
            self.best_model = deepcopy(extras['model'].state_dict())

    def on_epoch_end(self, losses, metrics, extras, epoch):
        self.summaries[epoch]['train_loss'] = losses['train']
        self.summaries[epoch]['train_metrics'] = metrics['train']
        self.summaries[epoch]['time'] = extras['time']
        representative_loss = 'train'  # for best model udpate

        if 'val' in losses:
            representative_loss = 'val'
            self.summaries[epoch]['val_loss'] = losses['val']

        if 'val' in metrics:
            self.summaries[epoch]['val_metrics'] = metrics['val']

        if losses[representative_loss] < self.best_score:
            self.best_score = losses[representative_loss]
            self.best_model = deepcopy(extras['model'])
        self.epoch_started = False

    def state_dict(self):
        state = {}
        state['best_score'] = self.best_score
        state['best_model'] = self.best_model
        state['summaries'] = self.summaries
        state['others'] = self.others
        state['prevs'] = self.prevs
        return deepcopy(state)

    def load_state_dict(self, state):
        self.best_score = state['best_score']
        self.best_model = state['best_model']
        self.summaries = state['summaries']
        self.others = state['others']
        self.prevs = state['prevs']


class BoardLog(Callback):
    def __init__(self, comment='learn', path='runs'):
        self.path = path
        self.run = 0
        self.comment = comment
        self.batch_count = 0

    def on_train_start(self, epochs):
        india_timezone = timezone(timedelta(hours=5.5))
        time_str = datetime.now(tz=india_timezone).strftime('%d_%b_%H:%M:%S')
        path = os.path.join(self.path, self.comment, time_str)

        self.writer = SummaryWriter(log_dir=path, flush_secs=30)
        self.run += 1

    def on_batch_end(self, loss, metrics, extras, epoch, batch):
        lr_vals = {}
        for i, param in enumerate(extras['optimizer'].param_groups):
            lr_vals['lr_' + str(i)] = param['lr']
        self.writer.add_scalars(
            'batch', {'loss': loss, **metrics, **lr_vals}, global_step=self.batch_count)
        self.batch_count += 1

    def on_min_val_end(self, loss, metrics, extras, epoch, batch):
        self.writer.add_scalars(
            'min_val', {'loss': loss, **metrics}, global_step=self.batch_count)

    def on_epoch_end(self, losses, metrics, extras, epoch):
        self.writer.add_scalars('losses', losses, global_step=epoch)
        for metric in metrics['train']:
            self.writer.add_scalars(metric, {'val': metrics['val'][metric],
                                             'train': metrics['train'][metric]}, global_step=epoch)
