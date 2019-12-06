

class BasePolicy(object):
    # individual agent policy

    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


