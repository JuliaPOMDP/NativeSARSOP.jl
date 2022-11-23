@testset "sparse" begin
    vv1 = SparseVector{Float64, Int}[]
    for i ∈ 0:3 # 0, 1, 2, or 3 filled values
        for nz_idxs ∈ combinations(1:3, i)
            v = zeros(3)
            v[nz_idxs] .= rand(length(nz_idxs))
            push!(vv1, sparse(v))
        end
    end

    vv2 = SparseVector{Float64, Int}[]
    for i ∈ 0:3 # 0, 1, 2, or 3 filled values
        for nz_idxs ∈ combinations(1:3, i)
            v = zeros(3)
            v[nz_idxs] .= rand(length(nz_idxs))
            push!(vv2, sparse(v))
        end
    end

    # make sure we're not dispatching on the same thing twice
    cl1 = code_lowered(JSOP.min_ratio, (Vector{Float64},SparseVector{Float64, Int})) |> first
    cl2 = code_lowered(JSOP.min_ratio, (SparseVector{Float64, Int},SparseVector{Float64, Int})) |> first
    @test cl1 ≠ cl2

    for vi ∈ vv1
        dense_vi = Vector(vi)
        for vj ∈ vv2
            @test JSOP.min_ratio(dense_vi, vj) == JSOP.min_ratio(vi, vj)
        end
    end
end
