import dataclasses
from typing import List, Any, Dict

import pymongo
from attr import dataclass
from badger_utils.sacred.sacred_config import SacredConfig
from sacred.observers import MongoObserver
import pandas as pd


@dataclass
class SacredRun:
    id: int
    config: Dict[str, Any]


class SacredUtils:
    _observer: MongoObserver

    def __init__(self, config: SacredConfig):
        self._observer = config.create_mongo_observer()

    def load_metrics(self, run_ids: List) -> pd.DataFrame:
        runs = self._observer.metrics.find({'run_id': {'$in': run_ids}})
        df = pd.DataFrame()
        for run in runs:
            steps, values = run['steps'], run['values']
            df = df.join(pd.DataFrame(values, index=steps, columns=[run['run_id']]), how='outer')
        return df

    def get_last_run(self) -> SacredRun:
        runs = self._observer.runs.find({}, {'_id': 1, 'config': 1}).sort([('_id', pymongo.DESCENDING)]).limit(1)
        last_run = next(runs)
        return SacredRun(last_run['_id'], last_run['config'])
