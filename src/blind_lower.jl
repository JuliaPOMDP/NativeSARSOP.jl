Base.@kwdef struct BlindLowerBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function update!(pomdp::ModifiedSparseTabular, M::BlindLowerBound, Γ, S, A, _)
    residuals = M.residuals
    (;T,R,O) = pomdp
    γ = discount(pomdp)

    for a ∈ A
        α_a = M.α_tmp
        T_a = T[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)
        for s ∈ S
            Vb′ = 0.0
            for idx ∈ nzrange(T_a, s)
                sp = rv[idx]
                p = nz[idx]
                Vb′ += p*Γ[a][sp]
            end
            α_a[s] = R[s,a] + γ*Vb′
        end
        res = bel_res(Γ[a], α_a)
        residuals[a] = res
        copyto!(Γ[a], α_a)
    end
    return Γ
end

function worst_state_alphas(pomdp::ModifiedSparseTabular, S, A)
    (;R,T) = pomdp
    S = states(pomdp)
    A = actions(pomdp)
    γ = discount(pomdp)

    Γ = [zeros(length(S)) for _ in eachindex(A)]
    for a ∈ A
        nz = nonzeros(T[a])
        rv = rowvals(T[a])
        for s ∈ S
            rsa = R[s, a]
            r_min = Inf
            for idx ∈ nzrange(T[a], s)
                sp = rv[idx]
                p = nz[idx]
                r′ = p*R[sp, a]
                r_min = min(r′, r_min)
            end
            r_min = r_min===Inf ? -Inf : r_min
            Γ[a][s] = rsa + γ / (1 - γ) * r_min
        end
    end
    return Γ
end

function POMDPs.solve(sol::BlindLowerBound, pomdp::ModifiedSparseTabular)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    Γ = worst_state_alphas(pomdp, S, A)
    resize!(sol.α_tmp, length(S))
    residuals = resize!(sol.residuals, length(A))

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ, S, A, O)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return AlphaVectorPolicy(pomdp, Γ, A)
end
