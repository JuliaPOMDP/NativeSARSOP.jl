@testset "prune" begin
    # NativeSARSOP.strictly_dominates
    a1 = [1.0, 2.0, 3.0]
    a2 = [1.0, 2.1, 2.9]
    a3 = [0.9, 1.9, 2.9]
    @test !NativeSARSOP.strictly_dominates(a1, a2, 1e-10)
    @test NativeSARSOP.strictly_dominates(a1, a1, 1e-10)
    @test NativeSARSOP.strictly_dominates(a1, a3, 1e-10)

    # NativeSARSOP.intersection_distance
    b = SparseVector([1.0, 0.0])
    a1 = [1.0, 0.0]
    a2 = [0.0, 1.0]
    @test isapprox(NativeSARSOP.intersection_distance(a1, a2, b),
        sqrt(0.5^2 + 0.5^2), atol=1e-10)

    b = SparseVector([0.5, 0.5])
    @test isapprox(NativeSARSOP.intersection_distance(a1, a2, b), 0.0, atol=1e-10)
end
