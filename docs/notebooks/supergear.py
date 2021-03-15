from functools import reduce


def calculate_R(z1o, z3o, z1i, z3i):
    if z3o/z3i != z1o/z1i:
        return (z1o + z3o)/(z1o - z1i*z3o/z3i)
    else:
        return 0


def calculate_w2i_ratio(z1, z3):
    return 2*z1/(z3-z1)


def calculate_w2o_ratio(z1o, z3o, z1i, z3i):
    return (z1o + z1i*z3o/z3i) / (z3o - z1o)


def factors(n):
    f = set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    return list(f)


class SuperGear:    
    def __init__(self, m):
        self.m = m
        self.reset_geometry()
    
    @property
    def wo(self):
        return self.wi / self.R
        
    def reset_geometry(self):
        # n teeth
        self.z1i = None
        self.z2i = None
        self.z3i = None
        self.z1o = None
        self.z2o = None
        self.z3o = None
        # diameters
        self.d1i = None
        self.d2i = None
        self.d3i = None
        self.d1o = None
        self.d2o = None
        self.d3o = None
        # rotations
        self.w1 = None
        self.w2i = None
        self.w2o = None
        self.w3 = None
        # i/o rotation
        self.wi = None
        self.R = None
    
    def set_gears(self, z1i, z3i, z1o, z3o):
        self.z1i = z1i
        self.z3i = z3i
        self.z1o = z1o
        self.z3o = z3o
        self._set_everything(10000)
    
    def set_input_rotation(self, wi):
        self._set_everything(wi)
    
    def _set_everything(self, wi):
        # n teeth
        self.z2i = (self.z3i-self.z1i) / 2
        self.z2o = (self.z3o-self.z1o) / 2
        #diameters
        self.d1i = self.m * self.z1i
        self.d2i = self.m * self.z2i
        self.d3i = self.m * self.z3i
        self.d1o = self.m * self.z1o
        self.d2o = self.m * self.z2o
        self.d3o = self.m * self.z3o
        #i/o rotation
        self.wi = wi
        self.R = calculate_R(self.z1o, self.z3o, self.z1i, self.z3i)
        #rotations
        self.w1 = wi
        self.w2i = wi * 2 * self.z1i / (self.z3i - self.z1i)
        self.w2o = wi * (self.z1o + self.z1i * self.z3o/self.z3i) / (self.z3o - self.z1o)
        self.w3 = wi * self.z1i / self.z3i
    
    def possible_satellites(self):
        totalzi = self.z1i + self.z3i
        totalzo = self.z1o + self.z3o
        fi = factors(totalzi)
        fo = factors(totalzo)
        return {
            'gi': [f for f in fi if f < self.z1i],
            'go': [f for f in fo if f < self.z1o],
        }
    
    def n_possible_satellites(self):
        return {k: len(v) for k, v in self.possible_satellites().items()}
    
    def show_specs(self):
        for key, val in self.__dict__.items():
            print(key + ':\t', int(round(val)))
        print()
        print('wo:\t', self.wo)
        