# Rosenbrock function with bounds
rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = RetroProblem(rosenbrock, [-1.2, 1.5], AutoForwardDiff(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])


@testset "BFGS Update" begin
    result_2dim = solve(prob, BFGSUpdate(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = solve(prob, BFGSUpdate(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = solve(prob, BFGSUpdate(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "SR1 Update" begin

    result_2dim = solve(prob, SR1Update(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = solve(prob, SR1Update(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = solve(prob, SR1Update(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "Exact Hessian Update" begin

    result_2dim = solve(prob, ExactHessian(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = solve(prob, ExactHessian(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = solve(prob, ExactHessian(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-3)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "Gauss Newton" begin

    residuals(x) = [10*(x[2] - x[1]^2); 1 - x[1]]
    objective(x) = 0.5 * sum(abs2, residuals(x))

    result_2dim = solve(prob, GaussNewtonUpdate(residuals), TwoDimSubspace(); maxiter=10000, verbose=true)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = solve(prob, GaussNewtonUpdate(residuals), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = solve(prob, GaussNewtonUpdate(residuals), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-3)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end
