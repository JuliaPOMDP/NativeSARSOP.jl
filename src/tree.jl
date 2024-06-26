mutable struct PruneData
    last_Γ_size::Int
    last_B_size::Int
    prune_threshold::Float64
end

struct BinManager
    lowest_ub::Float64
    num_levels::Int
    num_of_bins_per_level::Vector{Int}
    bin_levels_intervals::Vector{Dict{Symbol, Float64}}
    bin_levels_nodes::Vector{Dict{Int, Dict{Symbol, Union{Tuple{Int, Int}, Float64}}}}
    bin_levels::Vector{Dict{Symbol, Dict{Tuple{Int, Int}, Union{Float64, Int}}}}
    previous_lowerbound::Dict{Int, Float64}
end

function BinManager(Vs_upper::Vector{Float64}, num_bins_per_level=[5, 10])
    num_levels = length(num_bins_per_level)
    lowest_ub = minimum(Vs_upper)
    highest_ub = maximum(Vs_upper)
    
    # [level][:ub|:entropy] => value
    bin_levels_intervals = [Dict{Symbol, Float64}() for _ in 1:num_levels]
    
    # [level][b_idx][:key|:prev_error] => (ub_interval_idx, entropy_interval_idx)|previous_error
    bin_levels_nodes = Vector{Dict{Int, Dict{Symbol, Union{Tuple{Int, Int}, Float64}}}}(undef, num_levels) 
    
    # [level][:bin_value|:bin_count|:bin_error][(ub_interval_idx, entropy_interval_idx)] => Float64|Int|Float64
    bin_levels = Vector{Dict{Symbol, Dict{Tuple{Int, Int}, Union{Float64, Int}}}}(undef, num_levels) 
    
    num_states = length(Vs_upper)
    max_e = max_entropy(num_states)
    for level_i in 1:num_levels
        num_bins = num_bins_per_level[level_i]
        
        bin_levels_intervals[level_i][:ub] = (highest_ub - lowest_ub) / num_bins
        bin_levels_intervals[level_i][:entropy] = max_e / num_bins
        
        bin_levels_nodes[level_i] = Dict{Int, Dict{Symbol, Union{Tuple{Int, Int}, Float64}}}()
        
        level = Dict{Symbol, Dict{Tuple{Int, Int}, Union{Float64, Int}}}()
        level[:bin_value] = Dict{Tuple{Int, Int}, Float64}()
        level[:bin_count] = Dict{Tuple{Int, Int}, Int}()
        level[:bin_error] = Dict{Tuple{Int, Int}, Float64}()
        bin_levels[level_i] = level
    end
    
    previous_lowerbound = Dict{Int, Float64}() # b_idx => lowerbound
    
    return BinManager(
        lowest_ub,
        num_levels,
        num_bins_per_level,
        bin_levels_intervals,
        bin_levels_nodes,
        bin_levels,
        previous_lowerbound
    )
end

struct SARSOPTree
    pomdp::ModifiedSparseTabular

    b::Vector{SparseVector{Float64,Int}} # b_idx => belief vector
    b_children::Vector{UnitRange{Int}} # [b_idx][a_idx] => ba_idx
    Vs_upper::Vector{Float64}
    V_upper::Vector{Float64}
    V_lower::Vector{Float64}
    Qa_upper::Vector{Vector{Float64}}
    Qa_lower::Vector{Vector{Float64}}

    ba_children::Vector{UnitRange{Int}} # [ba_idx][o_idx] => bp_idx
    poba::Vector{Vector{Float64}} # [ba_idx][o_idx] => p(o|ba)

    _discount::Float64
    is_terminal::BitVector
    is_terminal_s::SparseVector{Bool, Int}

    #do we need both b_pruned and ba_pruned? b_pruned might be enough
    sampled::Vector{Int} # b_idx
    b_pruned::BitVector
    ba_pruned::BitVector
    real::Vector{Int} # b_idx
    is_real::BitVector
    cache::TreeCache
    prune_data::PruneData

    Γ::Vector{AlphaVec{Int}}
    
    bm::BinManager
end


function SARSOPTree(solver, pomdp::POMDP; num_bins_per_level=[5, 10])
    sparse_pomdp = ModifiedSparseTabular(pomdp)
    cache = TreeCache(sparse_pomdp)

    upper_policy = solve(solver.init_upper, sparse_pomdp)
    corner_values = map(maximum, zip(upper_policy.alphas...))

    bin_manager = BinManager(corner_values, num_bins_per_level)
    
    tree = SARSOPTree(
        sparse_pomdp,

        Vector{Float64}[],
        Vector{Int}[],
        corner_values, #upper_policy.util,
        Float64[],
        Float64[],
        Vector{Float64}[],
        Vector{Float64}[],
        Vector{Int}[],
        Vector{Float64}[],
        discount(pomdp),
        BitVector(),
        sparse_pomdp.isterminal,
        Int[],
        BitVector(),
        BitVector(),
        Vector{Int}(),
        BitVector(),
        cache,
        PruneData(0,0,solver.prunethresh),
        AlphaVec{Int}[],
        bin_manager
    )
    return insert_root!(solver, tree, _initialize_belief(pomdp, initialstate(pomdp)))
end

const NO_CHILDREN = 1:0

POMDPs.states(tree::SARSOPTree) = ordered_states(tree)
POMDPTools.ordered_states(tree::SARSOPTree) = states(tree.pomdp)
POMDPs.actions(tree::SARSOPTree) = ordered_actions(tree)
POMDPTools.ordered_actions(tree::SARSOPTree) = actions(tree.pomdp)
POMDPs.observations(tree::SARSOPTree) = ordered_observations(tree)
POMDPTools.ordered_observations(tree::SARSOPTree) = observations(tree.pomdp)
POMDPs.discount(tree::SARSOPTree) = discount(tree.pomdp)

function _initialize_belief(pomdp::POMDP, dist::Any=initialstate(pomdp))
    ns = length(states(pomdp))
    b = zeros(ns)
    for s in support(dist)
        sidx = stateindex(pomdp, s)
        b[sidx] = pdf(dist, s)
    end
    return b
end

function insert_root!(solver, tree::SARSOPTree, b)
    pomdp = tree.pomdp

    Γ_lower = solve(solver.init_lower, pomdp)
    for (α,a) ∈ alphapairs(Γ_lower)
        new_val = dot(α, b)
        push!(tree.Γ, AlphaVec(α, a))
    end
    tree.prune_data.last_Γ_size = length(tree.Γ)

    push!(tree.b, b)
    push!(tree.b_children, NO_CHILDREN)
    push!(tree.V_upper, init_root_value(tree, b))
    push!(tree.real, 1)
    push!(tree.is_real, true)
    push!(tree.V_lower, lower_value(tree, b))
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])
    push!(tree.b_pruned, false)
    push!(tree.is_terminal, is_terminal_belief(b, tree.is_terminal_s))
    fill_belief!(tree, 1)
    return tree
end

function update(tree::SARSOPTree, b_idx::Int, a, o)
    b = tree.b[b_idx]
    ba_idx = tree.b_children[b_idx][a]
    bp_idx = tree.ba_children[ba_idx][o]
    V̲, V̄ = if tree.is_terminal[bp_idx]
        0.,0.
    else
        lower_value(tree, tree.b[bp_idx]), upper_value(tree, tree.b[bp_idx])
    end
    tree.V_lower[bp_idx] = V̲
    tree.V_upper[bp_idx] = V̄
    return bp_idx, V̲, V̄
end

function add_belief!(tree::SARSOPTree, b, ba_idx::Int, o)
    push!(tree.b, b)
    b_idx = length(tree.b)
    push!(tree.b_children, NO_CHILDREN)
    push!(tree.is_real, false)
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])

    terminal = iszero(tree.poba[ba_idx][o]) || is_terminal_belief(b, tree.is_terminal_s)
    push!(tree.is_terminal, terminal)

    V̲, V̄ = if terminal
        0., 0.
    else
        lower_value(tree, b), upper_value(tree, b)
    end

    push!(tree.V_upper, V̄)
    push!(tree.V_lower, V̲)
    push!(tree.b_pruned, true)
    return b_idx, V̲, V̄
end

function add_action!(tree::SARSOPTree, b_idx::Int, a::Int)
    ba_idx = length(tree.ba_children) + 1
    push!(tree.ba_children, NO_CHILDREN)
    push!(tree.ba_pruned, true)
    return ba_idx
end

function fill_belief!(tree::SARSOPTree, b_idx::Int)
    if isempty(tree.b_children[b_idx])
        fill_unpopulated!(tree, b_idx)
    else
        fill_populated!(tree, b_idx)
    end
    update_bin_node!(tree, b_idx)
end

"""
Fill p(o|b,a), V̲(τ(bao)), V̄(τ(bao)) ∀ o,a if not already filled
"""
function fill_populated!(tree::SARSOPTree, b_idx::Int)
    γ = discount(tree)
    ACT = actions(tree)
    OBS = observations(tree)
    b = tree.b[b_idx]
    Qa_upper = tree.Qa_upper[b_idx]
    Qa_lower = tree.Qa_lower[b_idx]
    for a in ACT
        ba_idx = tree.b_children[b_idx][a]
        tree.ba_pruned[ba_idx] && continue
        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        Q̲ = Rba

        for o in OBS
            bp_idx, V̲, V̄ = update(tree, b_idx, a, o)
            b′ = tree.b[bp_idx]
            po = tree.poba[ba_idx][o]
            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
        end

        Qa_upper[a] = Q̄
        Qa_lower[a] = Q̲
    end

    tree.V_lower[b_idx] = lower_value(tree, tree.b[b_idx])
    tree.V_upper[b_idx] = maximum(tree.Qa_upper[b_idx])
end

function fill_unpopulated!(tree::SARSOPTree, b_idx::Int)
    pomdp = tree.pomdp
    γ = discount(tree)
    A = actions(tree)
    O = observations(tree)
    N_OBS = length(O)
    N_ACT = length(A)
    b = tree.b[b_idx]
    n_b = length(tree.b)
    n_ba = length(tree.ba_children)

    Qa_upper = Vector{Float64}(undef, N_ACT)
    Qa_lower = Vector{Float64}(undef, N_ACT)
    b_children = (n_ba+1):(n_ba+N_ACT)

    for a in A
        ba_idx = add_action!(tree, b_idx, a)
        ba_children = (n_b+1):(n_b+N_OBS)
        tree.ba_children[ba_idx] = ba_children

        n_b += N_OBS
        pred = dropzeros!(mul!(tree.cache.pred, pomdp.T[a],b))
        poba = zeros(Float64, N_OBS)
        Rba = belief_reward(tree, b, a)

        Q̄ = Rba
        Q̲ = Rba
        push!(tree.poba, poba)
        for o ∈ O
            # belief update
            bp = corrector(pomdp, pred, a, o)
            po = sum(bp)
            if po > 0.
                bp.nzval ./= po
                poba[o] = po
            end

            bp_idx, V̲, V̄ = add_belief!(tree, bp, ba_idx, o)

            Q̄ += γ*po*V̄
            Q̲ += γ*po*V̲
        end
        Qa_upper[a] = Q̄
        Qa_lower[a] = Q̲
    end
    tree.b_children[b_idx] = b_children
    tree.Qa_upper[b_idx] = Qa_upper
    tree.Qa_lower[b_idx] = Qa_lower
    tree.V_lower[b_idx] = lower_value(tree, tree.b[b_idx])
    tree.V_upper[b_idx] = maximum(tree.Qa_upper[b_idx])
end

function initialize_bin_node!(tree::SARSOPTree, b_idx::Int)
    lb_val = tree.V_lower[b_idx]
    ub_val = tree.V_upper[b_idx]
    node_entropy = entropy(tree.b[b_idx])
    
    for level_i in 1:tree.bm.num_levels
        ub_interval_idx = get_interval_idx(
            ub_val, tree.bm.lowest_ub, tree.bm.bin_levels_intervals[level_i][:ub], 
            tree.bm.num_of_bins_per_level[level_i]
        )
        
        entropy_interval_idx = get_interval_idx(
            node_entropy, 0.0, tree.bm.bin_levels_intervals[level_i][:entropy], 
            tree.bm.num_of_bins_per_level[level_i]
        )
        
        key = (ub_interval_idx, entropy_interval_idx)
        
        if !haskey(tree.bm.bin_levels_nodes[level_i], b_idx)
            tree.bm.bin_levels_nodes[level_i] = Dict(b_idx => Dict(:key => key))
        end
        
        bin_count = get(tree.bm.bin_levels[level_i][:bin_count], key, 0)
        if bin_count > 0
            err = tree.bm.bin_levels[level_i][:bin_value][key] - lb_val
            tree.bm.bin_levels_nodes[level_i][b_idx][:prev_error] = err * err
            tree.bm.bin_levels[level_i][:bin_value][key] += err * err
            
            value = (tree.bm.bin_levels[level_i][:bin_value][key] * bin_count + lb_val) / (bin_count + 1)
            tree.bm.bin_levels[level_i][:bin_count][key] += 1
        else
            err = ub_val - lb_val
            tree.bm.bin_levels_nodes[level_i][b_idx][:prev_error] = err * err
            tree.bm.bin_levels[level_i][:bin_error][key] = err * err
            
            value = lb_val
            tree.bm.bin_levels[level_i][:bin_count] = Dict(key => bin_count + 1)
        end
        tree.bm.bin_levels[level_i][:bin_value][key] = value
    end
    tree.bm.previous_lowerbound[b_idx] = lb_val
end

function update_bin_node!(tree::SARSOPTree, b_idx::Int)
    lb_val = tree.V_lower[b_idx]
    up_val = tree.V_upper[b_idx]
   
    if !haskey(tree.bm.bin_levels_nodes[1], b_idx)
        return initialize_bin_node!(tree, b_idx)
    end
    
    key = tree.bm.bin_levels_nodes[1][b_idx][:key]
    bin_count = get(tree.bm.bin_levels[1][:bin_count], key, 0)
    if bin_count == 0
        error("We shouldn't ever reach here, right?")
        # return initialize_bin_node!(tree, b_idx)
    end
    
    for level_i in 1:tree.bm.num_levels
        key = tree.bm.bin_levels_nodes[level_i][b_idx][:key]
        
        bin_count = get(tree.bm.bin_levels[level_i][:bin_count], key, 0)
        if bin_count == 1
            err = up_val - lb_val
            tree.bm.bin_levels_nodes[level_i][b_idx][:prev_error] = err * err
            tree.bm.bin_levels[level_i][:bin_error][key] = err * err
        else
            err = tree.bm.bin_levels[level_i][:bin_value][key] - lb_val
            tree.bm.bin_levels[level_i][:bin_error][key] -= tree.bm.bin_levels_nodes[level_i][b_idx][:prev_error]
            tree.bm.bin_levels_nodes[level_i][b_idx][:prev_error] = err * err
            tree.bm.bin_levels[level_i][:bin_error][key] += err * err
        end

        tree.bm.bin_levels[level_i][:bin_value][key] = (tree.bm.bin_levels[level_i][:bin_value][key] * bin_count + lb_val - tree.bm.previous_lowerbound[b_idx]) / bin_count
    end
    tree.bm.previous_lowerbound[b_idx] = lb_val
end

function get_bin_value(tree::SARSOPTree, b_idx::Int)
    
    lb_val = tree.V_lower[b_idx]
    ub_val = tree.V_upper[b_idx]
    
    key = tree.bm.bin_levels_nodes[1][b_idx][:key]
    if tree.bm.bin_levels[1][:bin_count][key] == 1
        return ub_val
    else
        smallest_error = Inf
        best_level = 0
        best_key = key
        for level_i in 1:tree.bm.num_levels
            key = tree.bm.bin_levels_nodes[level_i][b_idx][:key]
            if tree.bm.bin_levels[level_i][:bin_error][key] + 1e-10 < smallest_error
                best_level = level_i
                smallest_error = tree.bm.bin_levels[level_i][:bin_error][key]
                best_key = key
            end
        end
        
        if tree.bm.bin_levels[best_level][:bin_value][best_key] > ub_val + 1e-10
            return ub_val
        elseif tree.bm.bin_levels[best_level][:bin_value][best_key] + 1e-10 < lb_val
            return lb_val
        else    
            return tree.bm.bin_levels[best_level][:bin_value][best_key]
        end
    end
end

function max_entropy(n::Int)
    return -1 * ((1.0 / n) * log(1.0 / n)) * n
end

function entropy(b::AbstractVector)
    ent = 0.0
    for b_i in b
        b_i > 0 && (ent -= b_i * log(b_i))
    end
    return ent
end

function get_interval_idx(value::Float64, lower::Float64, interval::Float64, num_intervals::Int)
    if interval == 0.0
        return 1
    end
    idx = Int(floor((value - lower) / interval) + 1)
    return clamp(idx, 1, num_intervals) 
end
