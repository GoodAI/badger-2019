from typing import Optional

from torch import Tensor, nn as nn

from attention.experts.expert_group import ExpertHiddenStateParams, ExpertInitRolloutParams
from attention.models.attention.units.external_module_unit import ExternalModuleWriteUnit, ExternalModuleReadUnit
from attention.models.with_module import WithModule
from attention.models.attention.multi_unit_attention import AttentionUnit


class ExpertUnit(WithModule, AttentionUnit[ExpertHiddenStateParams, Tensor, ExpertInitRolloutParams]):
    @property
    def module(self) -> Optional[nn.Module]:
        return None

    @property
    def n_queries(self) -> int:
        return 1


class ExternalModuleWriteHead(ExternalModuleWriteUnit[ExpertHiddenStateParams, ExpertInitRolloutParams]):
    pass


class ExternalModuleReadHead(ExternalModuleReadUnit[ExpertHiddenStateParams, ExpertInitRolloutParams]):
    pass
