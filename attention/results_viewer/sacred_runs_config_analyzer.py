from typing import List, Dict, Any, Tuple

from bokeh.layouts import column, row
from bokeh.models import TextInput, Div
from sacred.observers import MongoObserver

from badger_utils.view.result_utils import find_common_keys, group_dicts, dict_omit_keys, tuple_to_dict


class SacredRunsConfigAnalyzer:
    def __init__(self, mongo_observer: MongoObserver):
        self._observer = mongo_observer

    def analyze_runs(self, run_ids: List[int]) -> Tuple[Dict[str, Any], Dict[List[Tuple[str, Any]], List[int]]]:
        runs = self._observer.runs.find({'_id': {'$in': run_ids}}, {'_id': 1, 'config': 1})
        items = list(runs)
        common_keys = find_common_keys(items, lambda x: x['config'])
        result = group_dicts(items, lambda x: dict_omit_keys(x['config'], set(common_keys) | {'seed'}),
                             lambda x: x['_id'])
        return common_keys, result

    def update_by_input(self):
        try:
            min_id = int(self.widget_min_id.value)
            max_id = int(self.widget_max_id.value)
            common_keys, diff = self.analyze_runs(list(range(min_id, max_id + 1)))

            def join_dict(delimiter: str, data: Dict):
                return delimiter.join([f'{k}: {v}' for k, v in data.items()])

            formatted_config = join_dict('<br/>', common_keys)
            formatted_diff = '<br/>'.join([f'{join_dict(", ", tuple_to_dict(k))}: {v}' for k, v in diff.items()])
            self.widget_config_common.text = f'<pre>{formatted_config} <hr/>{formatted_diff}</pre>'
        except Exception as e:
            print(f'Exception: {e}')

    def create_layout(self):
        self.widget_min_id = TextInput(title='Min Id', value='1957')
        self.widget_min_id.on_change('value', lambda a, o, n: self.update_by_input())
        self.widget_max_id = TextInput(title='Max Id', value='1970')
        self.widget_max_id.on_change('value', lambda a, o, n: self.update_by_input())
        self.widget_config_common = Div(text='')
        self.update_by_input()
        return column(
            row(self.widget_min_id, self.widget_max_id),
            self.widget_config_common
        )
