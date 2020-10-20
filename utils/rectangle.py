import matplotlib.pyplot as plt
import numpy as np


class Point:
    def __init__(self, x, y):
        self.__arr = [x, y]

    @property
    def x(self):
        return self.__arr[0]

    @property
    def y(self):
        return self.__arr[1]

    def add_y(self, y):
        return Point(self.x, self.y + y)

    def add_x(self, x):
        return Point(self.x + x, self.y)

    def __getitem__(self, item):
        return self.__arr[item]

    def __repr__(self):
        return type(self).__name__ + str((self.x, self.y))

    def __array__(self):
        return np.array(self.__arr)


class Rectangle:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @property
    def x1(self):
        return self.p1.x

    @property
    def x2(self):
        return self.p2.x

    @property
    def y1(self):
        return self.p1.y

    @property
    def y2(self):
        return self.p2.y

    @staticmethod
    def from_coords(x1, y1, x2, y2):
        return Rectangle(Point(x1, y1), Point(x2, y2))

    @staticmethod
    def from_anchor(point, width, height):
        return Rectangle(point, point.add_x(width).add_y(height))

    @property
    def area(self):
        return self.height * self.width

    @property
    def width(self):
        return abs(self.p1.x - self.p2.x)

    @property
    def height(self):
        return abs(self.p1.y - self.p2.y)

    @property
    def points(self):
        return np.hstack([np.array(self.p1), np.array(self.p2)])

    def __repr__(self):
        return type(self).__name__ + str((self.p1, self.p2))

    def __array__(self):
        return self.points

    def plot(self, subplot):
        subplot.plot(self[:, 0], self[:, 1])
        return subplot

    def intersection(self, other) -> float:
        # return (self[:, 0].min() - Ls[:, 0].max()) * (self[:, 1].min() - Ls[:, 1].max())
        i_box = self._intersection(other)
        if i_box:
            return i_box.area
        return 0

    def _intersection(self, other) -> 'Rectangle':
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return Rectangle.from_coords(x1, y1, x2, y2)

    def iou(self, curr_box):
        # compute iou
        Si = self.intersection(curr_box)
        return Si / (self.area + curr_box.area - Si)


def plot(*shapes):
    fig = plt.figure()
    for x in shapes:
        ax = fig.add_subplot()
        x.plot(ax)
    plt.xticks(range(-2, 17))
    plt.yticks(range(-2, 17))
    plt.grid()
    plt.show()


def test():
    root = Rectangle.from_coords(0, 0, 10, 10)
    o1 = Rectangle.from_coords(5, 5, 15, 15)
    o2 = Rectangle.from_coords(5, 3, 15, 7)
    o3 = Rectangle.from_coords(3, 3, 7, 7)
    o4 = Rectangle.from_anchor(Point(11, 0), 5, 5)
    o5 = Rectangle.from_anchor(Point(-1, -1), root.width + 2, root.height + 2)

    plot(root, o1, o2, o3, o4, o5)

    assert root.iou(o1) == 25 / 175
    print('iou #1 works')
    assert root.iou(o2) == 20 / 120
    print('iou #2 works')
    assert root.iou(o3) == o3.area / root.area
    print('iou #3 works')
    assert root.iou(o4) == 0
    print('iou #4 works')
    assert root.iou(o5) == root.area / o5.area
    print('iou #5 works')
