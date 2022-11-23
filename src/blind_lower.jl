"""
Blind Lower bound initialization

Note:
If using SparseTabularPOMDP, transition matrix is assumed to be ordered
as T[s′, s], as opposed to the default T[s,s′]
"""
Base.@kwdef struct BlindLowerBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function update!(pomdp::POMDP, M::BlindLowerBound, Γ, S, A, O)
    residuals = M.residuals
    γ = discount(pomdp)

    for (a_idx, a) in enumerate(A)
        α_a = M.α_tmp
        for (s_idx, s) in enumerate(S)
            Qas = reward(pomdp, s, a)
            for (s′, p) in weighted_iterator(transition(pomdp, s, a))
                Qas += γ*p*Γ[a_idx][stateindex(pomdp, s′)]
            end
            α_a[s_idx] = Qas
        end
        res = bel_res(Γ[a_idx], α_a)
        residuals[a_idx] = res
        copyto!(Γ[a_idx], α_a)
    end
    return Γ
end

"""
Finds distribution T(s′|s,a)
Return iterator of support s′∈ 𝒮 and probabilities T(s′|s,a)

---
assumes T[s′,s] as opposed to default T[s,s′] for `SparseTabularPOMDP`
"""
function transitions(pomdp::SparseTabularPOMDP, s, a)
    T = pomdp.T
    T_a = T[a]
    nz = nonzeros(T_a)
    rv = rowvals(T_a)
    nzr = nzrange(T_a,s)
    return zip(@view(rv[nzr]), @view(nz[nzr]))
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
            rsa = R[s,a]
            if rsa === -Inf
                α_a[s] = -Inf
            else
                Vb′ = 0.0
                for idx ∈ nzrange(T_a, s)
                    sp = rv[idx]
                    p = nz[idx]
                    Vb′ += p*Γ[a][sp]
                end
                α_a[s] = R[s,a] + γ*Vb′
            end
        end
        res = bel_res(Γ[a], α_a)
        residuals[a] = res
        copyto!(Γ[a], α_a)
    end
    return Γ
end

function worst_state_alphas(pomdp::POMDP, S, A)
    γ = discount(pomdp)
    Γ = [zeros(length(S)) for _ in eachindex(A)]
    for (a_idx, a) in enumerate(A)
        for (s_idx, s) in enumerate(S)
            r_min = Inf
            if isterminal(pomdp, s)
                r_min = 0.
            else
                for (s′, p) ∈ weighted_iterator(transition(pomdp,s,a))
                    r = reward(pomdp, s′, a)
                    if r < r_min
                        r_min = r
                    end
                end
            end
            Γ[a_idx][s_idx] = 1 / (1 - γ) * r_min
        end
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

function POMDPs.solve(sol::BlindLowerBound, pomdp::POMDP)
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
