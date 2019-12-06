class SingleCachedTensor:
    def __init__(self, creator):
        self._tensor = None
        self._key = None
        self._creator = creator

    def tensor(self, key):
        if self._tensor is None or self._key != key:
            self._key = key
            self._tensor = self._creator(key)
        #     print(f'Tensor creating new')
        # print(f'Tensor: {self._tensor.device}')
        return self._tensor

