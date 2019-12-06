from datetime import datetime


class TimeProfiler:
    last: datetime

    def __init__(self):
        self.last = datetime.now()

    def update_and_print_ms(self, prefix: str = ''):
        now = datetime.now()
        diff = now - self.last
        self.last = now
        print(f'{prefix}: {diff.microseconds / 1000:.0f} ms')
