"""Budgeted Active Experimentation reproduction (arXiv 2602.22021).

Clean-room re-implementation of the paper's Algorithms 1 & 2 and the
theoretical objects (Lemmas 4.4, Theorems 4.5/4.8/4.9) under *adaptive*
covariate selection -- the condition the prior toy reproduction omitted.
"""
from . import dgp, estimators, acquire, algorithm1, minimax  # noqa: F401

__version__ = "0.1.0"
