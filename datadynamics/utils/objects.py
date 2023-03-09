import numpy as np

from datadynamics.utils.colors import ColorPicker

colorpicker = ColorPicker()


class Position:
    """Object that represents a position in the environment.

    The object also contains information about scaling, translation to fit on
    displays as well as a color, label, and whether the position is static or
    not.
    """

    def __init__(
        self,
        pos,
        scaling,
        translation,
        static,
        label,
        id,
        color=colorpicker.get_color_by_name("black"),
    ):
        """Initialize position object.

        Args:
            pos: Position in environment.
            scaling (tuple): Scaling factors for displaying.
            translation (tuple): Translation factors for displaying.
            static (bool): Whether the position is static or not.
            label: Label representing object.
            id (str): Unique identifier of object.
            color (tuple, optional): RGB tuple of color. Defaults to `black`.
        """
        self._pos = pos
        self._scaled_pos = None
        self.scaling = scaling
        self.translation = translation
        self.static = static
        self._label = label
        self._id = id
        self._color = color

    def _compute_scaled_position(self, pos):
        """Returns scaled and translated position of object.

        Args:
            pos (tuple): Position to scale and translate.

        Returns:
            np.ndarray: Scaled and translated position.
        """
        return np.array(
            [
                pos[0] * self.scaling[0] + self.translation[0],
                pos[1] * self.scaling[1] + self.translation[1],
            ]
        )

    @property
    def position(self):
        """Position of object.

        Returns:
            Object position.
        """
        return self._pos

    @position.setter
    def position(self, pos):
        self._pos = pos

    @position.deleter
    def position(self):
        del self._pos

    @property
    def scaled_position(self):
        """Scaled and translated position of object.

        Position is recomputed for every call if the object is not static.

        Returns:
            np.ndarray: Scaled and translated position.
        """
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
    def label(self):
        """Label of object.

        Returns:
            Object label.
        """
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @label.deleter
    def label(self):
        del self._label

    @property
    def id(self):
        """Unique identifier of object.

        Returns:
            Object identifier.
        """
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @id.deleter
    def id(self):
        del self._id

    @property
    def color(self):
        """Color of object.

        Returns:
            tuple: RGB tuple of object's color.
        """
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @color.deleter
    def color(self):
        del self._color


class Point(Position):
    """Object representing points in the environment.

    Attributes:
        collector_tracker (dict): Dictionary of collectors and how many times
            they have collected this point.
    """

    def __init__(self, pos, scaling, translation, label=None, id=None):
        """Initialize point.

        Args:
            pos: Position in the environment
            scaling (tuple): Scaling factors for displaying.
            translation (tuple): Translation factors for displaying.
            label (optional): Label of point. Defaults to None.
            id (str, optional): Unique identifier of point. Defaults to None.
        """
        super().__init__(
            pos=pos,
            scaling=scaling,
            translation=translation,
            static=True,
            label=label,
            id=id,
            color=colorpicker.get_color_by_name("lightgrey"),
        )
        self._collect_counter = 0
        self.collector_tracker = {}

    def get_collect_counter(self):
        """Returns number of times point has been collected.

        Returns:
            int: Collect counter.
        """
        return self._collect_counter

    def is_collected(self):
        """Returns whether or not the object has previously been collected.

        Returns:
            bool: True if object has been collected, False otherwise.
        """
        return self._collect_counter > 0

    def collect(self, collector):
        """Collects point, increments collect counter, and colors the point.

        Args:
            collector (Collector): Collector of the point.
        """
        self.collector_tracker[collector.id] = (
            self.collector_tracker.get(collector.id, 0) + 1
        )
        self._collect_counter += 1
        self.color = collector.color


class Collector(Position):
    """Object representing collectors in the environment.

    Attributes:
        label: Label of the current collector position.
        points (list): List of points collected by the collector.
        path_positions (list): List of positions that the collector has moved
            to.
        path_labels (list): List of labels that the collector has moved to.
        moves (int): Number of moves the collector has made.
        cheated (int): Number of points that the collector has collected that
            have already been collected by another collector.
        unique_points_collected (int): Number of unique points collected by
            the collector.
        total_points_collected (int): Total number of points collected by the
            collector.
    """

    def __init__(self, pos, scaling, translation, label=None, id=None):
        """Initialize collector.

        Args:
            pos: Position in the environment.
            scaling (tuple): Scaling factors for displaying.
            translation (tuple): Translation factors for displaying.
            label (optional): Optional label of the current collector
                position. Defaults to None.
            id (str, optional): Unique identifier of collector. Defaults to
                None.
        """
        super().__init__(
            pos=pos,
            scaling=scaling,
            translation=translation,
            static=False,
            label=label,
            id=id,
            color=colorpicker.get_color(),
        )
        self.points = []
        self.path_positions = [pos]
        self.path_labels = [label]
        self.moves = 0
        self.cheated = 0
        self.unique_points_collected = 0
        self.total_points_collected = 0

    def move(self, position, label=None):
        """Moves collector to a new position with an optional label.

        Args:
            position: New position of the collector.
            label (optional): New label. Defaults to None.
        """
        if label is not None:
            self.label = label
            self.path_labels.append(label)
        self.position = position
        self.path_positions.append(position)
        self.moves += 1

    def collect(self, point, timestamp):
        """Collects a point.

        Args:
            point (Point): Point to collect.
        """
        self.points.append((point, timestamp))
        self.total_points_collected += 1
        if point.is_collected():
            self.cheated += 1
        else:
            self.unique_points_collected += 1
        point.collect(self)
