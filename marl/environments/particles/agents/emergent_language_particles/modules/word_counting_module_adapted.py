import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


class WordCountingModuleAdapted(nn.Module):

    def __init__(self, config):
        super(WordCountingModuleAdapted, self).__init__()
        self.oov_prob = config.oov_prob
        word_counts = Tensor(config.vocab_size)
        if config.use_cuda:
            word_counts.cuda()
        self.word_counts = Variable(word_counts)

    def forward(self, utterances):
        cost = -(utterances/(self.oov_prob + self.word_counts.sum() - 1)).sum()
        self.word_counts = self.word_counts + utterances
        return cost
