# ARC Competition Solver

A compact ARC-style solver built around:
- deterministic grid parsing
- candidate rule generation
- rule scoring against training examples
- output prediction on test inputs
- fallback heuristics for unsolved tasks

## Project layout

arc_solver/
  __init__.py
  grid_utils.py
  rule_library.py
  task_loader.py
  solver.py
  benchmark_runner.py

## Quick start

1. Install dependencies:
   python -m pip install numpy

2. Put task JSON files in a folder, e.g. tasks/

3. Run the benchmark:
   python -m arc_solver.benchmark_runner tasks/

## Expected task format

Each JSON file may look like:
{
  "train": [
    {"input": [[...]], "output": [[...]]},
    {"input": [[...]], "output": [[...]]}
  ],
  "test": [
    {"input": [[...]]},
    {"input": [[...]]}
  ]
}

## Notes

This project is intentionally simple but strong:
- it focuses on symbolic rules that ARC tasks often use
- it ranks rules by training consistency
- it includes a fallback strategy when no rule is clearly dominant

That makes it a good baseline for competition work.
