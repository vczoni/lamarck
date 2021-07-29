from __future__ import annotations

import time
import math


class ProgressBuilder:
    """
    Class for handling progress print formatting and time track.
    """
    _start_time: float
    step: int
    last_step: int
    max_step: int
    bar_size: int
    et: str
    tt: str

    def __init__(self, max_step: int, bar_size: int = 20):
        self._start_time = time.time()
        self.max_step = max_step
        self.bar_size = bar_size
        self.last_step = -1
        self.update(0)

    def update(self, step: int) -> None:
        """
        Update step and timers.
        """
        self.step = step
        et = time.time() - self._start_time
        self.et = self._format_time(et)
        if step != self.last_step:
            self.last_step = step
            if self.step > 0:
                tt = et/self.step * (self.max_step - self.step) + et
                self.tt = self._format_time(tt)
            else:
                self.tt = 'NA'

    def start_timer(self) -> None:
        """
        Start counting the seconds from now.
        """
        self._start_time = time.time()

    @staticmethod
    def _format_time(t: float) -> None:
        h = str(int(t/3600)).zfill(2)
        m = str(int(t/60) % 60).zfill(2)
        s = str(int(t/1) % 60).zfill(2)
        return f'{h}:{m}:{s}s'

    def get_progress(self) -> str:
        """
        Returns the bar progress with the step/max_step and the average step time
        and a time left estimation.

        Design
        ------
        |##-----------------------| 012 of 150 (00:16:38s / 03:11:17s)
        """
        phase = self.step + 1
        p = int(phase/self.max_step * self.bar_size)
        bar = ('#' * p) + ('-' * (self.bar_size - p))
        step_pow = int(math.log10(self.max_step)) + 1
        step_str = str(phase).zfill(step_pow)
        max_step_str = str(self.max_step)
        return f'|{bar}| {step_str} of {max_step_str} ({self.et} / {self.tt})'


class ProgressBuilderCollection:
    """
    Keeps track of multiple progress bars and plot them together.
    """

    _dict: dict

    def __init__(self):
        self._dict = {}

    def add_builder(self, name: str, builder: ProgressBuilder) -> None:
        self._dict.update({name: builder})
        self.__dict__.update(self._dict)

    def update(self, **step):
        """
        Updates each progress bar based on each :step: param.

        Example
        -------
        >>> b1 = ProgressBuilder(5, 8)
        >>> b2 = ProgressBuilder(50, 8)
        >>> bars = ProgressBuilderCollection()
        >>> bars.add_builder('first_bar', b1)
        >>> bars.add_builder('other_bar', b2)
        >>> bars.update(first_bar=3, other_bar=8)
        """
        [self._dict[pb].update(stepval) for pb, stepval in step.items()]

    def print(self):
        print(progress_printer(self._dict), end='\r')


def progress_printer(progress_builder: ProgressBuilder | dict) -> str:
    """
    Prints progress based on a ProgressBuilder object and sets the 'current_step'
    and the 'max_step'

    Can print multiple layers of progress if the :progress_builder: param is a
    `dict` of `ProgressBuilder`s as values and their Names as keys.
    """
    if isinstance(progress_builder, dict):
        progress = ' ::: '.join([f'{name}: {progress_printer(builder)}'
                                 for name, builder in progress_builder.items()])
    elif isinstance(progress_builder, ProgressBuilder):
        progress = progress_builder.get_progress()
    return progress
