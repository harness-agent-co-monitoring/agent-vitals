"""Section contracts: what a harness declares about its prompt sections, and the structural fallbacks.

- ``tb``: the declaration contracts (``prompt_sections``, ``feedback_sections``;
  AST loaders, validators) and the terminus-2 structural parser.
- ``tc``: the classification structural segmenter (``segment``,
  ``char_to_span_requests``) used when a design declines to declare.
"""

from vitals.addons.metaharness.contracts.tb import (  # noqa: F401
    authored_region,
    load_declared_feedback_sections,
    load_declared_function,
    load_declared_prompt_sections,
    sections_for,
    structural_sections,
    validate_feedback_sections,
    validate_prompt_sections,
)
