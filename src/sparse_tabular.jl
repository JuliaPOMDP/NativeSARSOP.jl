struct ModifiedSparseTabular <: POMDP{Int,Int,Int}
    T::Vector{SparseMatrixCSC{Float64, Int64}} # T[a][sp, s]
    R::Array{Float64, 2} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    isterminal::SparseVector{Bool, Int}
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

function ModifiedSparseTabular(pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    terminal = _vectorized_terminal(pomdp, S)
    T = _tabular_transitions(pomdp, S, A, terminal)
    R = _tabular_rewards(pomdp, S, A, terminal)
    O = _tabular_observations(pomdp, S, A, O)
    b0 = _vectorized_initialstate(pomdp, S)
    return ModifiedSparseTabular(T,R,O,terminal,b0,discount(pomdp))
end

function _tabular_transitions(pomdp, S, A, terminal)
    T = [Matrix{Float64}(undef, length(S), length(S)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(T)
        _fill_transitions!(pomdp, T[i], S, A[i], terminal)
    end
    T
end

function _fill_transitions!(pomdp, T, S, a, terminal)
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            T[:, s_idx] .= 0.0
            T[s_idx, s_idx] = 1.0
            continue
        end
        Tsa = transition(pomdp, s, a)
        for (sp_idx, sp) ∈ enumerate(S)
            T[sp_idx, s_idx] = pdf(Tsa, sp)
        end
    end
    T
end

function _tabular_rewards(pomdp, S, A, terminal)
    R = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            R[s_idx, :] .= 0.0
            continue
        end
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx] = reward(pomdp, s, a)
        end
    end
    R
end

function _tabular_observations(pomdp, S, A, O)
    _O = [Matrix{Float64}(undef, length(S), length(O)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(_O)
        _fill_observations!(pomdp, _O[i], S, A[i], O)
    end
    _O
end

function _fill_observations!(pomdp, Oa, S, a, O)
    for (sp_idx, sp) ∈ enumerate(S)
        obs_dist = observation(pomdp, a, sp)
        for (o_idx, o) ∈ enumerate(O)
            Oa[sp_idx, o_idx] = pdf(obs_dist, o)
        end
    end
    Oa
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(term,S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S, b0_vec)
        b0_vec[i] = pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

POMDPTools.ordered_states(pomdp::ModifiedSparseTabular) = axes(pomdp.R, 1)
POMDPs.states(pomdp::ModifiedSparseTabular) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::ModifiedSparseTabular) = eachindex(pomdp.T)
POMDPs.actions(pomdp::ModifiedSparseTabular) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::ModifiedSparseTabular) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::ModifiedSparseTabular) = ordered_observations(pomdp)

POMDPs.discount(pomdp::ModifiedSparseTabular) = pomdp.discount
POMDPs.initialstate(pomdp::ModifiedSparseTabular) = pomdp.initialstate
POMDPs.isterminal(pomdp::ModifiedSparseTabular, s::Int) = pomdp.isterminal[s]

n_states(pomdp::ModifiedSparseTabular) = length(states(pomdp))
n_actions(pomdp::ModifiedSparseTabular) = length(actions(pomdp))
n_observations(pomdp::ModifiedSparseTabular) = length(observations(pomdp))
