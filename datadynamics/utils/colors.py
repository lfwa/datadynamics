from matplotlib import colors as mcolors


class ColorPicker:
    """Pick a color from a cycle."""

    def __init__(self):
        """Initialize color picker."""
        self._colors = list(mcolors.TABLEAU_COLORS)
        self._curr_color = 0

    def get_color(self):
        """Retrieves the next color in the cycle as RGB tuple.

        Returns:
            tuple: RGB tuple.
        """
        color = mcolors.to_rgb(self._colors[self._curr_color])
        rgb_255 = tuple([int(c * 255) for c in color])
        self._curr_color = (self._curr_color + 1) % len(self._colors)
        return rgb_255

    def get_color_by_name(self, name):
        """Returns a color as RGB tuple from its matplotlib name, e.g. 'black'.

        Args:
            name (str): Name of color

        Returns:
            tuple: RGB tuple.
        """
        return tuple([int(c * 255) for c in mcolors.to_rgb(name)])

    def increase_intensity(self, rgb, factor=1.2):
        """Increases intensity of a color by a given factor.

        Args:
            rgb (tuple): RGB tuple for color.
            factor (float, optional): Factor to increase intensity by.
                Defaults to 1.2.

        Returns:
            tuple: RGB tuple with intensity increased.
        """
        return tuple([min(255, int(c * factor)) for c in rgb])
