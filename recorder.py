import csv
import os
import json
from typing import Dict, List, Any

class Recorder:

    def __init__(self, outputpath: str, datapath: str | None = None):
        self.filepath = os.path.join(outputpath, "recorder.csv")
        self.datapath = datapath
        self.data: Dict[str, Dict[str, Any]] = {}
        self.headers: List[str] = []
        self._load()
        if self.datapath is not None:
            self._read_data()

    def _read_data(self):
        for file in os.listdir(self.datapath):
            if file.endswith('.json') and os.path.isfile(os.path.join(self.datapath, file)):
                with open(os.path.join(self.datapath, file), 'r') as f:
                    data = json.load(f)
                    task = file.split('.json')[0]
                    self.set(task, 'input1', data['train'][0]['input'])
                    self.set(task, 'output1', data['train'][0]['output'])
                    self.set(task, 'input2', data['train'][1]['input'])
                    self.set(task, 'output2', data['train'][1]['output'])
                    self.set(task, 'question', data['test'][0]['input'])
                    self.set(task, 'answer', data['test'][0]['output'])

    def _load(self):
        try:
            with open(self.filepath, newline='', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if not rows:
                    return
                self.headers = rows[0][1:]
                for row in rows[1:]:
                    row_name = row[0]
                    self.data[row_name] = {
                        self.headers[i]: row[i + 1] if i + 1 < len(row) else ''
                        for i in range(len(self.headers))
                    }
        except FileNotFoundError:
            self.headers = []
            self.data = {}

    def _write_to_csv(self, filepath: str):
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([''] + self.headers)
            for exp_id, values in self.data.items():
                row = [exp_id] + [values.get(h, '') for h in self.headers]
                writer.writerow(row)

    def save(self):
        """Save data to current file."""
        self._write_to_csv(self.filepath)

    def save_as(self, new_filepath: str):
        """Save as new file but still bind to current file."""
        self._write_to_csv(new_filepath)

    def get(self, row_name: str, col_name: str) -> Any:
        """Get a value."""
        return self.data.get(row_name, {}).get(col_name, None)

    def set(self, row_name: str, col_name: str, value: Any):
        """Set a value."""
        if col_name not in self.headers:
            self.headers.append(col_name)
            for e in self.data.values():
                e[col_name] = ''
        if row_name not in self.data:
            self.data[row_name] = {h: '' for h in self.headers}
        self.data[row_name][col_name] = value

    def __repr__(self):
        return f"<ExperimentTable {len(self.data)} experiments × {len(self.headers)} types>"