# Rosenbrock function with bounds
rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = RetroProblem(rosenbrock, [-1.2, 1.5], AutoForwardDiff(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])


@testset "BFGS Update" begin
    result_2dim = optimize(prob, BFGSUpdate(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = optimize(prob, BFGSUpdate(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = optimize(prob, BFGSUpdate(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "SR1 Update" begin

    result_2dim = optimize(prob, SR1Update(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = optimize(prob, SR1Update(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = optimize(prob, SR1Update(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "Exact Hessian Update" begin

    result_2dim = optimize(prob, ExactHessian(), TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = optimize(prob, ExactHessian(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = optimize(prob, ExactHessian(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-3)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

@testset "Gauss Newton" begin
    # Define residuals - this will be prob.f for Gauss-Newton
    residuals(x) = [10*(x[2] - x[1]^2); 1 - x[1]]
    
    # Create problem with RESIDUAL function (not scalar objective)
    prob_gn = RetroProblem(residuals, [-1.2, 1.5], AutoForwardDiff(); 
                          lb=[-2.0, -2.0], ub=[2.0, 2.0])

    result_2dim = optimize(prob_gn, GaussNewtonUpdate(), TwoDimSubspace(); maxiter=10000, verbose=true)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)

    result_cg = optimize(prob_gn, GaussNewtonUpdate(), CGSubspace(); maxiter=10000, verbose=false)
    @test result_cg.converged
    @test isapprox(result_cg.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_cg.fx, 0.0; atol=1e-6)

    result_full = optimize(prob_gn, GaussNewtonUpdate(), FullSpace(); maxiter=10000, verbose=false)
    @test result_full.converged
    @test isapprox(result_full.x, [1.0, 1.0]; atol=1e-3)
    @test isapprox(result_full.fx, 0.0; atol=1e-6)
end

