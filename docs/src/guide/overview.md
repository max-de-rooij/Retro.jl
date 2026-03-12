# Algorithm Tutorial — Overview

This tutorial builds the Retro optimizer **one concept at a time**, starting
from the simplest possible optimizer and ending with a production-grade
trust-region reflective method.

Each page answers three questions:

1. **What problem are we trying to solve?**
2. **What idea fixes it?**
3. **How does Retro implement it?**

You do **not** need to read these pages to *use* Retro — the
[Quick Start](../getting-started/quickstart.md) is enough for that.  This
tutorial is for anyone who wants to **understand** what happens under the hood.

---

## The chapters

| # | Page | One-sentence summary |
|:-:|:-----|:---------------------|
| 1 | [The Simplest Optimizer](basic-optimizer.md)       | Gradient descent: easy, but unreliable. |
| 2 | [Adding Trust Regions](trust-region.md)             | A safety radius that adapts to local curvature. |
| 3 | [Handling Bounds](reflective-bounds.md)             | Reflecting steps at box boundaries à la Coleman–Li. |
| 4 | [Working in Subspaces](subspaces.md)                | Solving the TR problem cheaply in 2-D or via CG. |
| 5 | [Hessian Approximations](hessian-approximations.md) | BFGS, SR1, and exact Hessians — trade-offs. |
| 6 | [Robustness & Fallbacks](robustness.md)             | What happens when things go wrong. |

Start with Chapter 1, or jump to any topic you are curious about.
