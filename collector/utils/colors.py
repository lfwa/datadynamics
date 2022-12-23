from matplotlib import colors as mcolors


class ColorPicker:
    def __init__(self):
        self._colors = list(mcolors.TABLEAU_COLORS)
        self._curr_color = 0

    def get_color(self):
        color = mcolors.to_rgb(self._colors[self._curr_color])
        rgb_255 = tuple([int(c * 255) for c in color])
        self._curr_color = (self._curr_color + 1) % len(self._colors)
        return rgb_255

    def get_color_by_name(self, name):
        return tuple([int(c * 255) for c in mcolors.to_rgb(name)])

    def increase_intensity(self, rgb, factor=1.2):
        return tuple([min(255, int(c * factor)) for c in rgb])
