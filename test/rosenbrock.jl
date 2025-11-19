# Rosenbrock function with bounds
rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = FidesProblem(rosenbrock, [-1.2, 1.5], AutoForwardDiff(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])


@testset "BFGS Update" begin
    options = Fides.TrustRegionOptions(verbose=false, maxiter=10000)

    result_2dim = Fides.solve(prob, BFGSUpdate(), TwoDimSubspace(); options=options)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = Fides.solve(prob, BFGSUpdate(), CGSubspace(); options=options)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = Fides.solve(prob, BFGSUpdate(), FullSpace(); options=options)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "SR1 Update" begin
    options = Fides.TrustRegionOptions(verbose=false, maxiter=10000)

    result_2dim = Fides.solve(prob, SR1Update(), TwoDimSubspace(); options=options)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = Fides.solve(prob, SR1Update(), CGSubspace(); options=options)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = Fides.solve(prob, SR1Update(), FullSpace(); options=options)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "Exact Hessian Update" begin
    options = Fides.TrustRegionOptions(verbose=false, maxiter=10000)

    result_2dim = Fides.solve(prob, ExactHessian(), TwoDimSubspace(); options=options)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = Fides.solve(prob, ExactHessian(), CGSubspace(); options=options)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = Fides.solve(prob, ExactHessian(), FullSpace(); options=options)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-3)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

