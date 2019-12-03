import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *


def norm(x):
    mu = x.mean(1, keepdims=True)
    std = x.std(1, keepdims=True)

    return (x - mu) / (std + 1e-8)


# Inputs are Memory, Attender -> Batch size, Features, Number of sources/Number of receivers
class AttentionLayer(nn.Module):
    def __init__(self, N1, N2, NK, NV):
        super(AttentionLayer, self).__init__()

        self.N1 = N1
        self.N2 = N2
        self.NV = NV
        self.NK = NK

    def forward(self, key, query, value, drop=False, beta=1):
        BS = key.size(0)
        NA = key.size(2)
        NB = query.size(2)
        NV = self.NV

        weights = torch.sum(key * query, 3) / sqrt(self.NK)
        if drop:
            mask = torch.le(torch.rand(BS, NA, NB).cuda(), 0.25).float()
            weights = weights - 40 * mask
        weights = F.softmax(weights * beta, dim=1)

        values = torch.sum(weights.unsqueeze(3) * value.unsqueeze(2), 1).view(BS * NB, NV)
        values = norm(values)
        values = values.view(BS, NB, NV).transpose(1, 2).contiguous()

        return values, weights


class BadgerAgent(nn.Module):
    def __init__(self, HIDDEN=32, KEY=8, ID=16, IN_HEADS=1, OUT_HEADS=1):
        super(BadgerAgent, self).__init__()
        self.device = 'cpu'

        # Initial hidden state value
        self.h0 = nn.Parameter(torch.randn(1, HIDDEN, 1))

        self.HIDDEN = HIDDEN
        self.KEY = KEY
        self.ID = ID
        self.IN_HEADS = IN_HEADS
        self.OUT_HEADS = OUT_HEADS

        # Key, query, value networks
        self.key1 = nn.ModuleList([nn.Linear(HIDDEN + ID, HIDDEN) for i in range(OUT_HEADS)])
        self.key2 = nn.ModuleList([nn.Linear(HIDDEN, KEY) for i in range(OUT_HEADS)])
        self.value = nn.ModuleList([nn.Linear(HIDDEN + ID, HIDDEN) for i in range(OUT_HEADS)])
        self.query = nn.ModuleList([nn.Linear(HIDDEN + ID, KEY) for i in range(IN_HEADS)])

        self.valnet1 = nn.Conv1d(HIDDEN + ID + HIDDEN * IN_HEADS, HIDDEN, 1)
        self.valnet2 = nn.Conv1d(HIDDEN, HIDDEN, 1)

        self.htoh = nn.Conv1d(HIDDEN + HIDDEN, HIDDEN, 1)
        self.htog = nn.Conv1d(HIDDEN + HIDDEN, HIDDEN, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def init_rollout(self, BS, N_EXPERTS):
        self.static_id = torch.randn(1, self.ID, N_EXPERTS).to(self.device).expand(BS, self.ID, N_EXPERTS)
        self.hid = self.h0.expand(BS, self.HIDDEN, N_EXPERTS).to(self.device)

    def generate_messages(self):
        BS = self.hid.size(0)
        N_EXPERTS = self.hid.size(2)
        NV = self.hid.size(1)

        z = torch.cat([self.hid, self.static_id], 1)
        z = z.transpose(1, 2).contiguous().view(BS * N_EXPERTS, self.HIDDEN + self.ID)

        key = torch.cat([self.key2[i](F.leaky_relu(self.key1[i](z))).view(BS, N_EXPERTS, 1, self.KEY) for i in
                         range(self.OUT_HEADS)], 1)
        query = torch.cat([self.query[i](z).view(BS, 1, N_EXPERTS, self.KEY) for i in range(self.IN_HEADS)], 2)
        value = torch.cat([self.value[i](z).view(BS, N_EXPERTS, self.HIDDEN) for i in range(self.OUT_HEADS)], 1)

        return key, query, value

    def receive_attention(self, result):
        # result will be BS, HIDDEN, HEADS * N_EXPERTS in sets of e1,e2,e3,e4 ... e1,e2,e3,e4 ...
        # reprocess to BS
        # Merge own hidden state with attention info
        N_EXPERTS = self.hid.size(2)

        results = []
        for i in range(self.IN_HEADS):
            results.append(result[:, :, N_EXPERTS * i:N_EXPERTS * (i + 1)])

        results = torch.cat(results, 1)
        z = torch.cat([results, self.hid, self.static_id], 1)
        z = norm(F.leaky_relu(self.valnet2(F.leaky_relu(self.valnet1(z)))))
        z = torch.cat([z, self.hid], 1)

        w = torch.sigmoid(self.htog(z))
        newhid = self.htoh(z)

        self.hid = newhid * w + self.hid * (1 - w)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0]  # hack to extract device
        return self


class MultitaskAgent(nn.Module):
    def __init__(self, HIDDEN=32, KEY=16, ID=16, INPUT_CHUNK=16, N_INPUTS=10):
        super().__init__()

        # test
        self.agent = BadgerAgent(HIDDEN, KEY, ID, IN_HEADS=4, OUT_HEADS=2)
        self.attn = AttentionLayer(HIDDEN + ID, HIDDEN + ID, KEY, HIDDEN)
        # test 2

        self.error_key = nn.Parameter(torch.randn(KEY))
        self.error_val = nn.Linear(1, HIDDEN)

        self.input_keys = nn.Linear(INPUT_CHUNK, KEY)
        self.input_val = nn.Linear(INPUT_CHUNK, HIDDEN)

        self.output_stub = nn.Parameter(torch.randn(1, KEY // 2))
        self.output_val = nn.Linear(HIDDEN, 1)

        self.N_INPUTS = N_INPUTS
        self.INPUT_CHUNK = INPUT_CHUNK
        self.KEY = KEY

        sigma = 1.65
        for p in self.agent.parameters():
            p.data *= sigma

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def init_rollout(self, BS, N_EXPERTS, N_OUTPUTS):
        self.N_OUTPUTS = N_OUTPUTS
        self.agent.init_rollout(BS, N_EXPERTS)

    def forward(self, x, err, output_addr=None, externalAgent=None, drop=False):
        inp_keys = [self.input_keys(x[:, :, i]).view(x.size(0), 1, 1, self.KEY) for i in range(self.N_INPUTS)]
        inp_vals = [self.input_val(x[:, :, i]).unsqueeze(1) for i in range(self.N_INPUTS)]

        err_keys = [self.error_key.view(1, 1, 1, self.KEY).expand(x.size(0), 1, 1, self.KEY)]
        err_vals = [self.error_val(err).unsqueeze(1)]

        agent_keys, agent_queries, agent_values = self.agent.generate_messages()
        agent_keys = [agent_keys]
        agent_queries = [agent_queries]
        agent_values = [agent_values]

        out_qs = self.output_stub.view(1, 1, 1, self.KEY // 2).expand(x.size(0), 1, self.N_OUTPUTS, self.KEY // 2)

        if output_addr is not None:
            output_queries = [torch.cat([out_qs, output_addr], 3)]
        else:
            output_queries = [
                torch.cat([out_qs, torch.randn(1, 1, self.N_OUTPUTS, self.KEY // 2).expand(*out_qs.size()).to(DEVICE)],
                          3)
            ]

        out_qs = output_queries[0][:, 0, :, :].permute(0, 2, 1).contiguous()

        # Lets add these as inputs
        for i in range(self.N_OUTPUTS):
            inp_keys.append(self.input_keys(out_qs[:, :, i]).view(x.size(0), 1, 1, self.KEY))
            inp_vals.append(self.input_val(out_qs[:, :, i]).unsqueeze(1))

        if externalAgent is not None:
            external_keys, external_queries, external_values = externalAgent.generate_messages()
            external_keys = [external_keys]
            external_queries = [external_queries]
            external_values = [external_values]
        else:
            external_keys = []
            external_values = []
            external_queries = []

        sender_keys = torch.cat(inp_keys + err_keys + agent_keys + external_keys, 1)
        sender_values = torch.cat(inp_vals + err_vals + agent_values + external_values, 1)
        receiver_queries = torch.cat(output_queries + agent_queries + external_queries, 2)

        result, w = self.attn(sender_keys, receiver_queries, sender_values, drop=drop, beta=2.5)

        outputs = result[:, :, :self.N_OUTPUTS]
        self.agent.receive_attention(
            result[:, :, self.N_OUTPUTS:self.N_OUTPUTS + self.agent.IN_HEADS * self.agent.hid.size(2)])

        if externalAgent is not None:
            externalAgent.receive_attention(
                result[:, :, self.N_OUTPUTS + self.agent.IN_HEADS * self.agent.hid.size(2):])

        outputs = torch.cat([self.output_val(outputs[:, :, i]) for i in range(self.N_OUTPUTS)], 1)

        return outputs, w, 0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.agent = self.agent.to(*args, **kwargs)
        return self
