# Boilerplate tests for Fides.jl
@testset "Error calls" begin

    # real valued objective with Gauss-Newton update
    f(x) = sum(x.^2)
    x0 = [1.0, 1.0]
    prob = RetroProblem(f, x0, AutoForwardDiff())
    @test_throws ArgumentError solve(prob, GaussNewtonUpdate(), TwoDimSubspace(); maxiter=10000, verbose=true)

    # vector valued objective with non-Gauss-Newton update
    residuals(x) = [10*(x[2] - x[1]^2); 1 - x[1]]
    prob_gn = RetroProblem(residuals, x0, AutoForwardDiff())
    @test_throws ArgumentError solve(prob_gn, BFGSUpdate(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test_throws ArgumentError solve(prob_gn, SR1Update(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test_throws ArgumentError solve(prob_gn, ExactHessian(), TwoDimSubspace(); maxiter=10000, verbose=false)

    # Mooncake AD with ExactHessian
    import Mooncake
    prob_mooncake = RetroProblem(f, x0, AutoMooncakeForward())
    @test_throws ArgumentError solve(prob_mooncake, ExactHessian(), TwoDimSubspace(); maxiter=10000, verbose=false)

end
