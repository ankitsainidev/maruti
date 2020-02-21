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
                isEnd = isEnd or callback.on_validation_end(epoch)
            return isEnd

        def on_train_start(self, epochs):
            isEnd = False
            for callback in callbacks:
                isEnd = isEnd or callback.on_train_start(epochs)
            return isEnd
    return NewCallback()
