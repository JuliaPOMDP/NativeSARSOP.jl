#=
NOTE: With default utility initialization not already an upper bound on value,
if too few iterations are run, the calculated upper value may not actually be
an upper bound on the value.

Same applies to using QMDP as an upper bound, but provided that the upper bound is
used (relative to lower bound) to guide search as well as determining convergence
the detriment to the final policy may be minor.
=#
Base.@kwdef struct FastInformedBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    init_value::Float64         = 0.
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function bel_res(α1, α2)
    max_res = 0.
    @inbounds for i ∈ eachindex(α1, α2)
        res = abs(α1[i] - α2[i])
        res > max_res && (max_res = res)
    end
    return max_res
end

function update!(𝒫::POMDP, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    γ = discount(𝒫)
    residuals = M.residuals

    for (a_idx, a) ∈ enumerate(𝒜)
        α_a = M.α_tmp
        for (s_idx, s) ∈ enumerate(𝒮)
            T = transition(𝒫, s, a)
            tmp = 0.0
            for o ∈ 𝒪
                Vmax = -Inf
                for α′ ∈ Γ
                    Vb′ = 0.0
                    for (sp_idx, sp) ∈ enumerate(𝒮)
                        Oprob = pdf(observation(𝒫, s, a, sp), o)
                        Tprob = pdf(T, sp)
                        @inbounds Vb′ += Oprob*Tprob*α′[sp_idx]
                    end
                    Vb′ > Vmax && (Vmax = Vb′)
                end
                tmp += Vmax
            end
            α_a[s_idx] = reward(𝒫, s, a) + γ*tmp
        end
        res = bel_res(Γ[a_idx], α_a)
        residuals[a_idx] = res
        copyto!(Γ[a_idx], α_a)
    end
    return Γ
end

function update!(𝒫::ModifiedSparseTabular, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    (;R,T,O) = 𝒫
    γ = discount(𝒫)
    residuals = M.residuals

    for a ∈ 𝒜
        α_a = M.α_tmp
        T_a = T[a]
        O_a = O[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)

        for s ∈ 𝒮
            rsa = R[s,a]

            if isinf(rsa)
                α_a[s] = -Inf
            elseif isterminal(𝒫,s)
                α_a[s] = 0.
            else
                tmp = 0.0
                for o ∈ 𝒪
                    O_ao = @view O_a[:,o] # FIXME: slow sparse indexing for inner O_ao[sp]
                    Vmax = -Inf
                    for α′ ∈ Γ
                        Vb′ = 0.0
                        for idx ∈ nzrange(T_a, s)
                            sp = rv[idx]
                            Tprob = nz[idx]
                            Vb′ += O_ao[sp]*Tprob*α′[sp]
                        end
                        Vb′ > Vmax && (Vmax = Vb′)
                    end
                    tmp += Vmax
                end
                α_a[s] = rsa + γ*tmp
            end
        end
        res = bel_res(Γ[a], α_a)
        residuals[a] = res
        copyto!(Γ[a], α_a)
    end
end

#=
function _update!(𝒫::ModifiedSparseTabular, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    (;R,T,O) = 𝒫
    γ = discount(𝒫)
    residuals = M.residuals

    for a ∈ 𝒜
        α_a = M.α_tmp
        T_a = T[a]
        Z_a = O[a]
        Tnz = nonzeros(T_a)
        Trv = rowvals(T_a)

        for s ∈ 𝒮
            rsa = R[s,a]

            if isinf(rsa)
                α_a[s] = -Inf
            elseif isterminal(𝒫,s)
                α_a[s] = 0.
            else
                tmp = 0.0
                for o ∈ 𝒪
                    Vmax = -Inf
                    for α′ ∈ Γ
                        Vb′ = sparse_col_mul_reduce(Z_a, o, T_a, s, α′)
                        Vb′ > Vmax && (Vmax = Vb′)
                    end
                    tmp += Vmax
                end
                α_a[s] = rsa + γ*tmp
            end
        end
        res = bel_res(Γ[a], α_a)
        residuals[a] = res
        copyto!(Γ[a], α_a)
    end
end

function sparse_col_mul_reduce(A::SparseMatrixCSC, a_col, B::SparseMatrixCSC, b_col, coeff::Vector)
    Anzr = nzrange(A, a_col)
    Anzval = @view nonzeros(A)[Anzr]
    Anzind = @view rowvals(A)[Anzr]
    mx = length(Anzind)

    Bnzr = nzrange(B, b_col)
    Bnzval = @view nonzeros(B)[Bnzr]
    Bnzind = @view rowvals(B)[Bnzr]
    my = length(Bnzind)

    return _binary_mul_reduce(mx,my, Anzind, Anzval, Bnzind, Bnzval, coeff)
end

function _binary_mul_reduce(mx::Int, my::Int, xnzind, xnzval, ynzind, ynzval, coeff)
    # f(nz, nz) -> nz, f(z, nz) -> z, f(nz, z) ->  z
    # require_one_based_indexing(xnzind, ynzind, xnzval, ynzval, rind, rval)
    ir = 0; ix = 1; iy = 1; v = 0.
    while ix ≤ mx && iy ≤ my
        jx = xnzind[ix]
        jy = ynzind[iy]
        if jx === jy
            v += xnzval[ix]*ynzval[iy]*coeff[jx]
            ir += 1; ix += 1; iy += 1
        elseif jx < jy
            ix += 1
        else
            iy += 1
        end
    end
    return v
end
=#

function POMDPs.solve(sol::FastInformedBound, pomdp::POMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)

    init_value = sol.init_value
    Γ = if isfinite(sol.init_value)
        [fill(sol.init_value, length(S)) for a ∈ A]
    else
        r_max = maximum(reward(pomdp, s, a) for a ∈ A, s ∈ S)
        V̄ = r_max/(1-γ)
        [fill(V̄, length(S)) for a ∈ A]
    end
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
