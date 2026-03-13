using Retro
using Test

@testset "Error Handling" begin
    @testset "Dimension Mismatches" begin
        f(x) = sum(abs2, x)
        
        # Mismatched bounds
        x0 = [0.0, 0.0]
        lb_wrong = [-1.0]  # Wrong length
        ub_wrong = [1.0]   # Wrong length
        
        @test_throws ArgumentError RetroProblem(f, x0, AutoForwardDiff(); lb=lb_wrong, ub=[1.0, 1.0])
        @test_throws ArgumentError RetroProblem(f, x0, AutoForwardDiff(); lb=[-1.0, -1.0], ub=ub_wrong)
    end
    
    @testset "Invalid Bounds" begin
        f(x) = sum(abs2, x)
        x0 = [0.0, 0.0]
        
        # Lower bound >= upper bound
        lb = [1.0, -1.0]
        ub = [0.0, 1.0]  # lb[1] > ub[1]
        
        @test_throws ArgumentError RetroProblem(f, x0, AutoForwardDiff(); lb=lb, ub=ub)
    end
    
    @testset "Empty or Invalid Inputs" begin
        f(x) = sum(abs2, x)
        
        # Empty vector
        @test_throws ArgumentError RetroProblem(f, Float64[], AutoForwardDiff())
        
        # Invalid function (should not crash constructor but might fail during optimization)
        invalid_f(x) = x[100]  # Will fail for small x
        prob = RetroProblem(invalid_f, [1.0], AutoForwardDiff())
        @test prob isa RetroProblem
        
        # Should fail during optimization
        @test_throws BoundsError optimize(prob; maxiter=1, display=Silent())
    end
    
    @testset "NaN/Inf Function Values" begin
        # Function that returns NaN
        nan_f(x) = x[1] > 0 ? NaN : sum(abs2, x)
        prob_nan = RetroProblem(nan_f, [1.0], AutoForwardDiff())
        
        # Should handle gracefully (terminate with error or fallback)
        result = optimize(prob_nan; maxiter=5, display=Silent())
        @test result isa RetroResult
        
        # Function that returns Inf
        inf_f(x) = x[1] > 0 ? Inf : sum(abs2, x)
        prob_inf = RetroProblem(inf_f, [1.0], AutoForwardDiff())
        
        result_inf = optimize(prob_inf; maxiter=5, display=Silent())
        @test result_inf isa RetroResult
    end
end

@testset "Robustness Tests" begin
    @testset "Ill-conditioned Problems" begin
        # Very flat objective
        flat_f(x) = 1e-16 * sum(abs2, x)
        prob_flat = RetroProblem(flat_f, [1.0, 1.0], AutoForwardDiff())
        
        result = optimize(prob_flat; maxiter=10, display=Silent())
        @test result isa RetroResult
        
        # Very steep objective
        steep_f(x) = 1e16 * sum(abs2, x .- [1.0, 2.0])
        prob_steep = RetroProblem(steep_f, [0.0, 0.0], AutoForwardDiff())
        
        result_steep = optimize(prob_steep; maxiter=10, display=Silent())
        @test result_steep isa RetroResult
    end
    
    @testset "Zero Gradient at Start" begin
        # Function with zero gradient at starting point
        zero_grad_f(x) = sum(x.^4)  # Gradient is zero at x=0
        prob_zero = RetroProblem(zero_grad_f, [0.0, 0.0], AutoForwardDiff())
        
        result = optimize(prob_zero; maxiter=5, display=Silent())
        @test result isa RetroResult
        # Should terminate quickly due to gradient tolerance
        @test result.termination_reason ∈ [:gtol, :maxiter, :stagnation]
    end
    
    @testset "Large Scale (within reason)" begin
        # Test with moderately large problem
        n = 100
        large_f(x) = sum(abs2, x .- ones(n))
        x0_large = zeros(n)
        
        prob_large = RetroProblem(large_f, x0_large, AutoForwardDiff())
        
        # Use CG subspace for large problem
        result = optimize(prob_large; 
                         maxiter=20, 
                         display=Silent(), 
                         subspace=CGSubspace(max_cg_iter=min(n, 10)))
        @test result isa RetroResult
    end
end

@testset "Input Validation" begin
    @testset "Options Validation" begin
        # Negative tolerances (should be allowed)
        opts_neg = RetroOptions(gtol_a=-1e-6)
        @test opts_neg.gtol_a == -1e-6
        
        # Zero trust region radius
        opts_zero_tr = RetroOptions(initial_tr_radius=0.0)
        @test opts_zero_tr.initial_tr_radius == 0.0
        
        # Very large trust region
        opts_large_tr = RetroOptions(initial_tr_radius=1e10, max_tr_radius=1e15)
        @test opts_large_tr.initial_tr_radius == 1e10
    end
end