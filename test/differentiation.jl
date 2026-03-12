using Retro
using Test
using ForwardDiff
using LinearAlgebra

@testset "Automatic Differentiation" begin
    @testset "ADObjectiveFunction Tests" begin
        # Simple quadratic
        f(x) = sum(abs2, x .- [1.0, 2.0])
        x0 = [0.0, 0.0]
        
        obj_ad = ADObjectiveFunction(f, AutoForwardDiff(), x0)
        @test obj_ad isa ADObjectiveFunction
        
        cache = RetroCache(2)
        
        # Test function evaluation
        f_val = objfunc!(cache, obj_ad, x0)
        @test f_val ≈ 5.0  # (0-1)^2 + (0-2)^2 = 5
        @test cache.f_calls == 1
        
        # Test gradient evaluation
        g = similar(x0)
        gradient!(g, cache, obj_ad, x0)
        @test g ≈ [-2.0, -4.0]  # 2*(x-[1,2]) at x=[0,0]
        @test cache.g_calls == 1
    end
    
    @testset "GradientObjectiveFunction Tests" begin
        f(x) = sum(abs2, x .- [1.0, 2.0])
        function grad!(g, x)
            g[1] = 2*(x[1] - 1.0)
            g[2] = 2*(x[2] - 2.0)
        end
        x0 = [0.0, 0.0]
        
        obj_grad = GradientObjectiveFunction(f, grad!, AutoForwardDiff(), x0)
        @test obj_grad isa GradientObjectiveFunction
        
        cache = RetroCache(2)
        
        # Test function evaluation
        f_val = objfunc!(cache, obj_grad, x0)
        @test f_val ≈ 5.0
        
        # Test gradient evaluation
        g = similar(x0)
        gradient!(g, cache, obj_grad, x0)
        @test g ≈ [-2.0, -4.0]
    end
    
    @testset "AnalyticObjectiveFunction Tests" begin
        f(x) = sum(abs2, x .- [1.0, 2.0])
        
        function grad!(g, x)
            g[1] = 2*(x[1] - 1.0)
            g[2] = 2*(x[2] - 2.0)
        end
        
        function hess!(H, x)
            fill!(H, 0.0)
            H[1, 1] = 2.0
            H[2, 2] = 2.0
        end
        
        x0 = [0.0, 0.0]
        
        obj_analytic = AnalyticObjectiveFunction(f, grad!, hess!)
        @test obj_analytic isa AnalyticObjectiveFunction
        
        cache = RetroCache(2)
        
        # Test function evaluation
        f_val = objfunc!(cache, obj_analytic, x0)
        @test f_val ≈ 5.0
        
        # Test gradient evaluation
        g = similar(x0)
        gradient!(g, cache, obj_analytic, x0)
        @test g ≈ [-2.0, -4.0]
        
        # Test Hessian evaluation
        H = zeros(2, 2)
        hessian!(H, cache, obj_analytic, x0)
        @test H ≈ [2.0 0.0; 0.0 2.0]
    end
end

@testset "Different AD backends" begin   
    @testset "ForwardDiff Backend" begin
        f(x) = x[1]^4 + x[2]^4 + x[1]*x[2]
        x0 = [1.0, -1.0]
        
        prob = RetroProblem(f, x0, AutoForwardDiff())
        @test prob isa RetroProblem
        
        # Quick optimization test
        result = optimize(prob; maxiter=5, display=Silent())
        @test result isa RetroResult
        @test result.function_evaluations > 0
        @test result.gradient_evaluations > 0
    end
end

@testset "Gradient Accuracy Tests" begin
    @testset "Complex Functions" begin
        # Rosenbrock function
        rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
        x_test = [0.5, 0.6]
        
        prob = RetroProblem(rosenbrock, [0.0, 0.0], AutoForwardDiff())
        cache = RetroCache(2)
        
        # Test gradient at specific point
        g_ad = similar(x_test)
        gradient!(g_ad, cache, prob.objective, x_test)
        
        # Compare with analytical gradient
        g_true = [-400*x_test[1]*(x_test[2] - x_test[1]^2) - 2*(1 - x_test[1]),
                  200*(x_test[2] - x_test[1]^2)]
        
        @test g_ad ≈ g_true atol=1e-10
    end
    
    @testset "Trigonometric Functions" begin
        trig_f(x) = sin(x[1])^2 + cos(x[2])^2 + x[1]*x[2]
        x_test = [π/4, π/3]
        
        prob = RetroProblem(trig_f, [0.0, 0.0], AutoForwardDiff())
        cache = RetroCache(2)
        
        g_ad = similar(x_test)
        gradient!(g_ad, cache, prob.objective, x_test)
        
        # Analytical gradient
        g_true = [2*sin(x_test[1])*cos(x_test[1]) + x_test[2],
                  -2*cos(x_test[2])*sin(x_test[2]) + x_test[1]]
        
        @test g_ad ≈ g_true atol=1e-10
    end
end

@testset "Hessian Evaluation" begin
    @testset "AD Hessian" begin
        f(x) = sum(abs2, x)
        x0 = [1.0, 2.0]
        obj = ADObjectiveFunction(f, AutoForwardDiff(), x0)
        cache = RetroCache(2)

        H = zeros(2, 2)
        hessian!(H, cache, obj, x0)
        @test H ≈ [2.0 0.0; 0.0 2.0] atol=1e-6
    end
end