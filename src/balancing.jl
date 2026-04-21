# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Balancing helpers — build output tables, assign detector IDs

using Random: randperm

# ============================================================================
# Build balanced output table from PreparedDataset
# ============================================================================

"""
    build_bal_table(prep, idxs, assigned_det) → Table

Build a flat Table from PreparedDataset for jlbal/jlbalml output.
Same columns as :wpe group plus `assigned_ged_detector`.
`idxs` selects rows (1-based into prep arrays).
"""
function build_bal_table(prep::PreparedDataset, idxs::Vector{Int},
                         assigned_det::Vector{UInt32})
    n = length(idxs)
    Table(
        sipm_pe_sums               = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums[i, :]) for i in idxs]),
        sipm_pe_sums_prompt        = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums_prompt[i, :]) for i in idxs]),
        sipm_pe_sums_delayed       = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums_delayed[i, :]) for i in idxs]),
        event_sum_pe               = prep.event_sum_pe[idxs],
        event_multiplicity         = Int32.(prep.event_multiplicity[idxs]),
        event_sum_pe_prompt        = prep.event_sum_pe_prompt[idxs],
        event_multiplicity_prompt  = Int32.(prep.event_multiplicity_prompt[idxs]),
        event_sum_pe_delayed       = prep.event_sum_pe_delayed[idxs],
        event_multiplicity_delayed = Int32.(prep.event_multiplicity_delayed[idxs]),
        assigned_ged_detector      = assigned_det,
        ged_energy_keV             = prep.ged_energy_keV[idxs],
        ged_t0_us                  = prep.ged_t0_us[idxs],
        trigger_det_ids            = VectorOfVectors(prep.trigger_det_ids[idxs]),
        trigger_times_us           = VectorOfVectors(prep.trigger_times_us[idxs]),
        trigger_pe_vals            = VectorOfVectors(prep.trigger_pe_vals[idxs]),
    )
end
export build_bal_table

# ============================================================================
# Detector ID assignment — sample from sub500keV distribution
# ============================================================================

"""
    sample_detector_ids(sub500_prep, n) → Vector{UInt32}

Sample `n` detector IDs from the sub500keV HPGe distribution (non-zero IDs).
"""
function sample_detector_ids(sub500_prep::PreparedDataset, n::Int)
    pool = filter(!=(UInt32(0)), sub500_prep.ged_detector_id)
    isempty(pool) && return zeros(UInt32, n)
    UInt32[pool[rand(1:length(pool))] for _ in 1:n]
end
export sample_detector_ids

# ============================================================================
# Training exclusion rules — parsed from config
# ============================================================================

function parse_exclusion_rules(rules)
    parsed = NamedTuple{(:key, :op, :val), Tuple{Symbol, Symbol, Float64}}[]
    for rule in rules
        key = Symbol(rule["key"])
        r = string(rule["range"])
        m = match(r"^(>=|<=|>|<)\s*(\d+\.?\d*)$", r)
        if m !== nothing
            op = Symbol(m.captures[1])
            val = parse(Float64, m.captures[2])
        else
            val = parse(Float64, r)
            op = :(==)
        end
        push!(parsed, (key=key, op=op, val=val))
    end
    parsed
end
export parse_exclusion_rules

function apply_exclusion(prep::PreparedDataset, rules)
    keep = trues(prep.n_events)
    for rule in rules
        col = getproperty(prep, rule.key)
        op = rule.op; val = rule.val
        @inbounds for i in 1:prep.n_events
            if op == :(<=)
                col[i] <= val && (keep[i] = false)
            elseif op == :(>=)
                col[i] >= val && (keep[i] = false)
            elseif op == :(<)
                col[i] < val && (keep[i] = false)
            elseif op == :(>)
                col[i] > val && (keep[i] = false)
            elseif op == :(==)
                col[i] == val && (keep[i] = false)
            end
        end
    end
    findall(keep)
end
export apply_exclusion
