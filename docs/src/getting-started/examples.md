# Examples

A collection of complete, runnable problems.

## Rosenbrock (unconstrained)

The classic banana-shaped valley.  The minimum is at ``(1, 1)`` with ``f = 0``.

```@example examples
using Retro, ForwardDiff

rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob   = RetroProblem(rosenbrock, [-1.2, 1.0], AutoForwardDiff())
result = optimize(prob; display = Iteration())

@assert is_successful(result)
@assert result.fx < 1e-10
```

## Rosenbrock (bounded)

Force the solution into a box.

```@example examples
prob = RetroProblem(rosenbrock, [-1.2, 1.0], AutoForwardDiff();
                    lb = [-2.0, -2.0], ub = [2.0, 2.0])
result = optimize(prob)
```

## Quadratic with narrow bounds

A simple quadratic where the solution is near a boundary.

```@example examples
f(x) = sum(abs2, x .- [1.0, 2.0, 3.0])

prob = RetroProblem(f, zeros(3), AutoForwardDiff();
                    lb = [0.5, 0.5, 0.5], ub = [1.5, 1.5, 1.5])
result = optimize(prob)

# Solution should be at [1.0, 1.5, 1.5] — clamped by upper bounds
```

## Exact Hessian for small problems

```@example examples
prob = RetroProblem(rosenbrock, [-1.2, 1.0], AutoForwardDiff())
result = optimize(prob;
    hessian_approximation = ExactHessian(),
    subspace = TwoDimSubspace(),
    display  = Iteration())
```

## Tuning convergence tolerances

```@example examples
opts = RetroOptions(
    gtol_a = 1e-10,          # tight gradient tolerance
    ftol_a = 1e-14,          # tight function tolerance
    initial_tr_radius = 0.1, # conservative initial radius
    max_tr_radius = 100.0,
)

result = optimize(prob; options = opts, maxiter = 5000)
```

## Trigonometric objective

```@example examples
trig(x) = sin(x[1])^2 + cos(x[2])^2 + x[1]*x[2]

prob = RetroProblem(trig, [1.0, -1.0], AutoForwardDiff();
                    lb = [-π, -π], ub = [π, π])
result = optimize(prob; display = Final())
```
