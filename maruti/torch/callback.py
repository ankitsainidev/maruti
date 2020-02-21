class Callback:
    def on_epoch_end(self, losses, metrics, epoch):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_batch_start(self, epoch, batch):
        pass

    def on_batch_end(self, loss, metrics, epoch, batch):
        pass

    def on_validation_start(self, epoch):
        pass

    def on_validation_end(self, loss, metrics, epoch):
        pass


def Compose(callbacks):
    class NewCallback(Callback):
        def on_epoch_end(self, losses, metrics, epoch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_epoch_end(losses, metrics, epoch)
            return isEnd

        def on_epoch_start(self, epoch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_epoch_start(epoch)
            return isEnd

        def on_batch_start(self, epoch, batch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_batch_start(epoch, batch)
            return isEnd

        def on_batch_end(self, loss, metrics, epoch, batch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_batch_end(loss, metrics,
                                                        epoch, batch)
            return isEnd

        def on_validation_start(self, epoch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_validation_start(epoch)
            return isEnd

        def on_validation_end(self, loss, metrics, epoch):
            isEnd = True
            for callback in callbacks:
                isEnd = isEnd and callback.on_validation_end(epoch)
            return isEnd
    return NewCallback()
