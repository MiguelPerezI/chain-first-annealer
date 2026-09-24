"""Runtime validation shared by the CLI and Python API (also active with -O)."""
import math
from numbers import Integral, Real


def validate_parameters(**values):
    integer_limits = {'k': (1, None), 'cmax': (2, 255), 'S': (1, None),
                      'iters': (1, None), 'epochs': (1, None),
                      'tournament_every': (1, None), 'threads': (1, 1024),
                      'seed': (0, None), 'h1_guide': (0, None)}
    probabilities = {'pcopy', 'ptrack', 'tournament_frac', 'crossover'}
    for name, value in values.items():
        if name in integer_limits:
            low, high = integer_limits[name]
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < low or (high is not None and value > high)):
                raise ValueError(f'{name} must be an integer in [{low}, {high or "unbounded"}]')
        elif name in probabilities or name in {'T0', 'T1', 'beta', 'gamma', 'mu', 'guide', 'guide_max', 'time_limit'}:
            if name == 'time_limit' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if name in probabilities and not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
            if name in {'T0', 'T1', 'time_limit'} and value <= 0:
                raise ValueError(f'{name} must be positive')
            if name in {'beta', 'gamma', 'mu', 'guide'} and value < 0:
                raise ValueError(f'{name} must be nonnegative')
            if name == 'guide_max' and value < 1:
                raise ValueError('guide_max must be at least 1')
        else:
            raise ValueError(f'Unknown parameter: {name}')
