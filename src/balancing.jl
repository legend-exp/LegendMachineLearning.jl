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

# ============================================================================
# Detector exclusions — drop by HPGe (events) or SiPM (columns)
# ============================================================================

_name_to_rawid(n::AbstractString) = UInt32(DetectorId(String(n)))
_name_to_rawid(n::Symbol) = _name_to_rawid(String(n))

"""
    resolve_excluded_ids(names) → Vector{UInt32}

Resolve a list of detector names (String/Symbol) to rawids via `DetectorId`.
"""
resolve_excluded_ids(names) = UInt32[_name_to_rawid(n) for n in names]
export resolve_excluded_ids

"""
    apply_ged_exclusion!(prep::PreparedDataset, excl_ids::Vector{UInt32}) → n_dropped

Drop all events from `prep` whose leading HPGe rawid is in `excl_ids`.
Rebuilds all per-event fields in place (n_events, matrices, trigger VoV, etc.).
Returns the number of dropped events.
"""
function apply_ged_exclusion!(prep::PreparedDataset, excl_ids::Vector{UInt32})
    isempty(excl_ids) && return 0
    excl_set = Set(excl_ids)
    keep_idxs = findall(!in(excl_set), prep.ged_detector_id)
    n_dropped = prep.n_events - length(keep_idxs)
    n_dropped == 0 && return 0

    prep.sipm_pe_sums          = prep.sipm_pe_sums[keep_idxs, :]
    prep.sipm_pe_sums_prompt   = prep.sipm_pe_sums_prompt[keep_idxs, :]
    prep.sipm_pe_sums_delayed  = prep.sipm_pe_sums_delayed[keep_idxs, :]
    prep.event_sum_pe          = prep.event_sum_pe[keep_idxs]
    prep.event_multiplicity    = prep.event_multiplicity[keep_idxs]
    prep.event_sum_pe_prompt   = prep.event_sum_pe_prompt[keep_idxs]
    prep.event_multiplicity_prompt  = prep.event_multiplicity_prompt[keep_idxs]
    prep.event_sum_pe_delayed  = prep.event_sum_pe_delayed[keep_idxs]
    prep.event_multiplicity_delayed = prep.event_multiplicity_delayed[keep_idxs]
    prep.ged_detector_id       = prep.ged_detector_id[keep_idxs]
    prep.ged_energy_keV        = prep.ged_energy_keV[keep_idxs]
    prep.ged_t0_us             = prep.ged_t0_us[keep_idxs]
    prep.trigger_det_ids       = prep.trigger_det_ids[keep_idxs]
    prep.trigger_times_us      = prep.trigger_times_us[keep_idxs]
    prep.trigger_pe_vals       = prep.trigger_pe_vals[keep_idxs]
    prep.valid_indices         = collect(1:length(keep_idxs))
    prep.n_events              = length(keep_idxs)
    n_dropped
end
export apply_ged_exclusion!

"""
    drop_sipm_columns!(prep::PreparedDataset, excl_ids::Vector{UInt32}) → n_dropped

Drop excluded SiPMs from `prep`: trims `sipm_detector_ids`, the three PE matrices,
`per_det_raw_trig_pe`, and filters `trigger_det_ids`/`trigger_times_us`/`trigger_pe_vals`
per-event to drop trigger entries pointing at excluded SiPMs.
Returns the number of dropped SiPM columns.
"""
function drop_sipm_columns!(prep::PreparedDataset, excl_ids::Vector{UInt32})
    isempty(excl_ids) && return 0
    excl_set = Set(excl_ids)
    keep_cols = findall(!in(excl_set), prep.sipm_detector_ids)
    n_dropped = length(prep.sipm_detector_ids) - length(keep_cols)
    n_dropped == 0 && return 0

    prep.sipm_detector_ids     = prep.sipm_detector_ids[keep_cols]
    prep.sipm_pe_sums          = prep.sipm_pe_sums[:, keep_cols]
    prep.sipm_pe_sums_prompt   = prep.sipm_pe_sums_prompt[:, keep_cols]
    prep.sipm_pe_sums_delayed  = prep.sipm_pe_sums_delayed[:, keep_cols]
    prep.n_sipms               = length(keep_cols)
    for id in excl_ids; haskey(prep.per_det_raw_trig_pe, id) && delete!(prep.per_det_raw_trig_pe, id); end

    # Filter raw trigger VoV per-event
    for i in 1:prep.n_events
        dids = prep.trigger_det_ids[i]
        any(in(excl_set), dids) || continue
        mask = [!(d in excl_set) for d in dids]
        prep.trigger_det_ids[i]  = dids[mask]
        prep.trigger_times_us[i] = prep.trigger_times_us[i][mask]
        prep.trigger_pe_vals[i]  = prep.trigger_pe_vals[i][mask]
    end
    n_dropped
end
export drop_sipm_columns!

