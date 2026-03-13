using Retro
using Test
using LinearAlgebra

@testset "Package Structure" begin
    @test isa(Retro, Module)
    
    # Test that main types are exported
    @test isdefined(Retro, :RetroProblem)
    @test isdefined(Retro, :RetroResult)
    @test isdefined(Retro, :RetroOptions)
    @test isdefined(Retro, :RetroCache)
    @test isdefined(Retro, :optimize)
    
    # Test abstract types
    @test isdefined(Retro, :AbstractObjectiveFunction)
    @test isdefined(Retro, :AbstractSubspace)
    @test isdefined(Retro, :AbstractTRSolver)
    @test isdefined(Retro, :AbstractHessianApproximation)
end

@testset "Basic Construction" begin
    # Simple quadratic function
    f(x) = sum(abs2, x .- [1.0, 2.0])
    x0 = [0.0, 0.0]
    
    @testset "RetroProblem Construction" begin
        # With AD
        prob_ad = RetroProblem(f, x0, AutoForwardDiff())
        @test prob_ad isa RetroProblem
        @test length(prob_ad.x0) == 2
        
        # With bounds
        lb = [-5.0, -5.0]
        ub = [5.0, 5.0]
        prob_bounded = RetroProblem(f, x0, AutoForwardDiff(); lb=lb, ub=ub)
        @test prob_bounded isa RetroProblem
        @test prob_bounded.lb == lb
        @test prob_bounded.ub == ub
    end
    
    @testset "RetroOptions Construction" begin
        opts = RetroOptions()
        @test opts isa RetroOptions
        @test opts.gtol_a == 1e-6
        
        opts_custom = RetroOptions(gtol_a=1e-4)
        @test opts_custom.gtol_a == 1e-4
    end
    
    @testset "Cache Construction" begin
        cache = RetroCache(5)
        @test cache isa RetroCache
        @test length(cache.g) == 5
        @test length(cache.p) == 5
        @test cache.f_calls == 0
        @test cache.g_calls == 0
    end
end

@testset "Subspace Methods" begin
    @testset "TwoDimSubspace" begin
        subspace = TwoDimSubspace()
        @test subspace isa TwoDimSubspace
        @test subspace.normalize == true
        
        subspace_no_norm = TwoDimSubspace(normalize=false)
        @test subspace_no_norm.normalize == false
    end
    
    @testset "CGSubspace" begin
        cg = CGSubspace()
        @test cg isa CGSubspace
        @test cg.max_cg_iter == 50
        @test cg.cg_tol == 1e-6
    end
    
    @testset "FullSpace" begin
        fs = FullSpace()
        @test fs isa FullSpace
    end
end

@testset "TR Solvers" begin
    @testset "EigenTRSolver" begin
        solver = EigenTRSolver()
        @test solver isa EigenTRSolver
        @test solver.regularization == 1e-8
    end
    
    @testset "CauchyTRSolver" begin
        solver = CauchyTRSolver()
        @test solver isa CauchyTRSolver
    end  
end

@testset "Hessian Approximations" begin
    @testset "BFGS" begin
        bfgs = BFGS()
        @test bfgs isa BFGS
        @test bfgs.B0_scale == 1.0
        @test bfgs.skip_update == true
    end
    
    @testset "SR1" begin
        sr1 = SR1()
        @test sr1 isa SR1
        @test sr1.B0_scale == 1.0
        @test sr1.skip_threshold == 1e-8
    end
    
    @testset "ExactHessian" begin
        exact = ExactHessian()
        @test exact isa ExactHessian
        @test exact.regularization == 1e-8
    end
end

@testset "Display Modes" begin
    @test Silent() isa Silent
    @test Iteration() isa Iteration
    @test Final() isa Final
    @test Verbose() isa Verbose
end