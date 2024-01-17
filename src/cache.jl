struct TreeCache
    pred::SparseVector{Float64,Int}
    alpha::Vector{Float64}
    Γ::Vector{Float64}
end

function TreeCache(pomdp::ModifiedSparseTabular)
    Ns = n_states(pomdp)
    Na = n_actions(pomdp)
    No = n_observations(pomdp)

    pred = Vector{Float64}(undef, Ns)
    alpha = Vector{Float64}(undef, Ns)
    Γ = Vector{Float64}(undef, Ns*No)
    return TreeCache(pred, alpha, Γ)
end
