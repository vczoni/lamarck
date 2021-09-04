import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return ', '.join([f'x: {self.x}', f'y: {self.y}'])

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return np.hypot(dx, dy)


class TravelSalesman:
    def __init__(self, n_cities, seed=42):
        self._n = n_cities
        self._seed = seed
        self.coords = []
        self.coord_df = None
        self.build()

    def build(self):
        np.random.seed(self._seed)
        self.coords = [Location(*np.random.randint(0, 100, size=2))
                       for _ in range(self._n)]
        coord_data = [(loc.x, loc.y) for loc in self.coords]
        self.coord_df = pd.DataFrame(coord_data, columns=['x', 'y'])

    def get_distance(self, id1, id2):
        loc1 = self.coords[id1]
        loc2 = self.coords[id2]
        return loc1.distance(loc2)

    def get_route_distance(self, seq):
        if seq is None:
            seq = list(range(self._n))
        dist = 0
        for city1, city2 in zip(seq[0:-1], seq[1:]):
            dist += self.get_distance(city1, city2)
        return dist

    def plot(self, hl=None, **kw):
        ax = self.coord_df.plot.scatter('x', 'y', xlim=(0, 100), ylim=(0, 100), **kw)
        if hl is not None:
            if isinstance(hl, int):
                self.coord_df[hl:hl+1].plot.scatter('x', 'y', ax=ax, color='m')
            elif isinstance(hl, (list, tuple)):
                [self.coord_df[h:h+1].plot.scatter('x', 'y', ax=ax, color='m')
                 for h in hl]
        plt.gca().set_aspect('equal', adjustable='box')
        return ax

    def plot_route(self, seq=None, hl=None, **kw):
        if seq is None:
            seq = list(range(self._n))
        ax = self.coord_df.loc[np.array(seq)]\
            .plot('x', 'y', color='grey', legend=False)
        ax = self.plot(hl=hl, ax=ax, **kw)
        plt.gca().set_aspect('equal', adjustable='box')
        start = seq[0]
        end = seq[-1]
        ax = self.coord_df[start:start+1]\
            .plot.scatter('x', 'y', color='lime', ax=ax)
        ax = self.coord_df[end:end+1]\
            .plot.scatter('x', 'y', color='r', ax=ax)
        return ax
