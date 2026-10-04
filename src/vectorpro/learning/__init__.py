"""Learning loop: plans in, verified vector programs out, kept in a searchable registry."""

from vectorpro.learning.examples import ExampleLesson, ExampleSet, ExampleStream, TargetSource
from vectorpro.learning.learner import Learner, LearnerConfig, LearningOutcome, structure_candidates
from vectorpro.learning.plan import LearningPlan, OutputWidth
from vectorpro.learning.registry import Capability, Registry
from vectorpro.learning.search import search_composition

__all__ = [
    "Capability",
    "ExampleSet",
    "ExampleLesson",
    "ExampleStream",
    "Learner",
    "LearnerConfig",
    "LearningOutcome",
    "LearningPlan",
    "OutputWidth",
    "Registry",
    "TargetSource",
    "search_composition",
    "structure_candidates",
]
