# API — Objective Functions

Retro supports three levels of user control over derivatives.

## Abstract type

```@docs
AbstractObjectiveFunction
```

## AD-only objective

```@docs
ADObjectiveFunction
```

## User gradient + AD Hessian

```@docs
GradientObjectiveFunction
```

## Fully analytic

```@docs
AnalyticObjectiveFunction
```

## Evaluation interface

These functions are used internally by the optimizer.  You normally do not
need to call them yourself.

```@docs
objfunc!
gradient!
hessian!
value_and_gradient!
value_gradient_and_hessian!
```
