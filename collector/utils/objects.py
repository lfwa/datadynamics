import numpy as np

from collector.utils.colors import ColorPicker

colorpicker = ColorPicker()


class Position:
    def __init__(
        self,
        pos,
        scaling,
        translation,
        static,
        color=colorpicker.get_color_by_name("black"),
    ):
        self._pos = pos
        self._scaled_pos = None
        self.scaling = scaling
        self.translation = translation
        self.static = static
        self._color = color

    def _compute_scaled_position(self, pos):
        return np.array(
            [
                pos[0] * self.scaling + self.translation,
                pos[1] * self.scaling + self.translation,
            ]
        )

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, pos):
        self._pos = pos

    @position.deleter
    def position(self):
        del self._pos

    @property
    def scaled_position(self):
        if self._scaled_pos is None or not self.static:
            self.scaled_position = self._compute_scaled_position(self.position)
        return self._scaled_pos

    @scaled_position.setter
    def scaled_position(self, scaled_pos):
        self._scaled_pos = scaled_pos

    @scaled_position.deleter
    def scaled_position(self):
        del self._scaled_pos

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @color.deleter
    def color(self):
        del self._color


class Point(Position):
    def __init__(self, pos, scaling, translation):
        super().__init__(
            pos,
            scaling,
            translation,
            static=True,
            color=colorpicker.get_color_by_name("lightgrey"),
        )
        self.collect_counter = 0

    def get_collect_counter(self):
        return self.collect_counter

    def is_collected(self):
        return self.collect_counter > 0

    def collect(self, collector):
        # FIXME: What happens if a point is collected by several collectors at the same time?
        factor = 1.2**self.collect_counter
        self.collect_counter += 1
        self.color = colorpicker.increase_intensity(
            collector.color, factor=factor
        )

    def __str__(self):
        return (
            f"Point: {{ pos: {self._pos}, collected: {self.collect_counter} }}"
        )


class Collector(Position):
    def __init__(self, pos, scaling, translation):
        super().__init__(
            pos,
            scaling,
            translation,
            False,
            color=colorpicker.get_color(),
        )
        self.points = []
        self.cheated = 0
        self.unique_points_collected = 0
        self.total_points_collected = 0

    def collect(self, point):
        self.points.append(point)
        self.total_points_collected += 1
        if point.is_collected():
            self.cheated += 1
        else:
            self.unique_points_collected += 1
        point.collect(self)
