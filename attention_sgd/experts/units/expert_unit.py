from typing import Optional

from torch import Tensor, nn as nn

from attention_sgd.experts.expert_group import ExpertHiddenStateParams
from attention_sgd.models.attention.units.external_module_unit import ExternalModuleWriteUnit, ExternalModuleReadUnit
from attention_sgd.models.with_module import WithModule
from attention_sgd.models.attention.multi_unit_attention import AttentionUnit


class ExpertUnit(WithModule, AttentionUnit[ExpertHiddenStateParams, Tensor]):
    @property
    def module(self) -> Optional[nn.Module]:
        return None


class ExternalModuleWriteHead(ExternalModuleWriteUnit[ExpertHiddenStateParams]):
    pass


class ExternalModuleReadHead(ExternalModuleReadUnit[ExpertHiddenStateParams]):
    pass