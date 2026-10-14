# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
#
# Extraction — single-step: read jlevt → filter → compute windowed PE → write jlext
#
# I/O helpers live in src/io.jl.  Report generation in src/report.jl.

using LegendDataManagement: ljl_propfunc
using PropertyFunctions: PropertyFunctions
using TypedTables: Table
using LegendHDF5IO: lh5open
using Unitful

# ============================================================================
# Unit Helpers
# ============================================================================

prep_strip(x::Real)              = Float64(x)
prep_strip(x::Unitful.Quantity)  = Float64(ustrip(x))
prep_to_ns(x::Real)              = Float64(x)
prep_to_ns(x::Unitful.Quantity)  = Float64(ustrip(u"ns", x))
prep_to_us(x::Real)              = Float64(x)
prep_to_us(x::Unitful.Quantity)  = Float64(ustrip(u"μs", x))

export prep_strip, prep_to_ns, prep_to_us

# Δt = t_max_pe(SiPM mode) − t0_hpge — absolute SiPM-trigger window
const _DT_T_LO_US     = 40.0
const _DT_T_HI_US     = 60.0
const _DT_N_TIME_BINS = 100

# ============================================================================
# Event Filter Parsing
# ============================================================================

function parse_event_filter(filter_string::String)
    cleaned = replace(strip(replace(filter_string, r"\s+" => " ")), r"\$" => "")
    ljl_propfunc(cleaned)
end
export parse_event_filter

# Top-level property names referenced by a PropertyFunction. The published
# PropertyFunctions.jl (v0.2.x) does not export an accessor — the input
# columns are encoded as the first type parameter of `PropertyFunction{...}`.
_propfunc_input_columns(pf) = typeof(pf).parameters[1]::Tuple{Vararg{Symbol}}

# ============================================================================
# Key Selection — preserves jlevt Table/VoV structure
# ============================================================================

function select_keys_from_table(evt, keys_config::Dict)
    groups = Dict{Symbol, Any}()
    for (group_name, key_list) in keys_config
        group_sym = Symbol(group_name)
        hasproperty(evt, group_sym) || continue
        group_data = getproperty(evt, group_sym)
        syms = [Symbol(k) for k in key_list if hasproperty(group_data, Symbol(k))]
        isempty(syms) && continue
        groups[group_sym] = Table(NamedTuple{Tuple(syms)}(Tuple(getproperty(group_data, s) for s in syms)))
    end
    group_syms = filter(s -> haskey(groups, s), Symbol.(collect(keys(keys_config))))
    Table(NamedTuple{Tuple(group_syms)}(Tuple(groups[s] for s in group_syms)))
end
export select_keys_from_table

# ============================================================================
# FileKey Collection
# ============================================================================

function collect_run_filekeys(l200::LegendData, group_def)
    run_filekeys = Tuple{DataPeriod, DataRun, Vector{FileKey}}[]
    for (period_str, runs_def) in group_def
        period, runs = parse_period_run(String(period_str), runs_def)
        for run in runs
            fks = filter(!in(bad_filekeys(l200)), search_disk(FileKey, l200.tier[:jlevt, :phy, period, run]))
            isempty(fks) || push!(run_filekeys, (period, run, fks))
        end
    end
    run_filekeys
end
export collect_run_filekeys

"""
    _k40_first_filekey(l200, group_def) → FileKey or nothing

First valid :phy filekey for the group (used to derive channelinfo ordering).
Falls back to :cal via `start_filekey` if no :phy files exist.
"""
function _k40_first_filekey(l200::LegendData, group_def)
    for (period_str, runs_def) in group_def
        period, runs = parse_period_run(String(period_str), runs_def)
        for run in runs
            fks = try
                filter(!in(bad_filekeys(l200)), search_disk(FileKey, l200.tier[:jlevt, :phy, period, run]))
            catch; FileKey[] end
            !isempty(fks) && return first(fks)
            try
                return start_filekey(l200, (period, run, :phy))
            catch; end
        end
    end
    nothing
end


# ============================================================================
# Windowed PE — per-event inner loop
# ============================================================================

function _windowed_pe(trig_pos, trig_dc, trig_pe,
                      lo_ns::Float64, hi_ns::Float64,
                      trig_thresh::Float64, mult_thresh::Float64)
    n = length(trig_pos)
    pe_sums = Vector{Float64}(undef, n)
    @inbounds for d in 1:n
        s = 0.0
        for i in eachindex(trig_pos[d])
            Bool(trig_dc[d][i]) && continue
            pe = prep_strip(trig_pe[d][i])
            pe < trig_thresh && continue
            t = prep_to_ns(trig_pos[d][i])
            lo_ns <= t <= hi_ns && (s += pe)
        end
        pe_sums[d] = s
    end
    pe_sums, sum(pe_sums), count(>=(mult_thresh), pe_sums)
end

function _event_has_bad_values(evt, idx::Int, check_ged::Bool;
                               pe_key::Symbol, pos_key::Symbol)
    try
        trig_pe  = getproperty(evt.spms, pe_key)[idx]
        trig_pos = getproperty(evt.spms, pos_key)[idx]
        for d in eachindex(trig_pe), t in eachindex(trig_pe[d])
            (!isfinite(prep_strip(trig_pe[d][t])) || !isfinite(prep_to_ns(trig_pos[d][t]))) && return true
        end
    catch; return true; end
    if check_ged
        try
            !isfinite(prep_strip(evt.geds.max_e_cusp_ctc_cal[idx])) && return true
            !isfinite(prep_to_us(evt.geds.t0_start[idx])) && return true
        catch; return true; end
    end
    false
end

# ============================================================================
# compute_windowed_pe — main reusable function (works on any jlevt Table)
# ============================================================================

function compute_windowed_pe(evt::Table, ds_cfg::Dict, ds_name::String)
    # ── Parse SiPM key mapping ───────────────────────────────────────────
    sk = ds_cfg["spms_keys"]
    pe_key  = Symbol(sk["pe_cal"])
    pos_key = Symbol(sk["trig_pos"])
    dc_key  = Symbol(sk["is_dc"])
    trig_thresh = Float64(ds_cfg["trigger_threshold_pe"])
    mult_thresh = Float64(ds_cfg["multiplicity_threshold_pe"])

    # ── Parse summed trigger time windows ────────────────────────────────
    stw = ds_cfg["summed_trigger_time_window"]
    fw = stw["full"]; pw = stw["prompt"]; dw = stw["delayed"]
    reference = String(fw["reference"])
    has_ged_t0 = reference == "ged_t0"
    center_us = has_ged_t0 ? NaN : Float64(fw["center_us"])
    fw_before = Float64(fw["before_us"]); fw_after = Float64(fw["after_us"])
    pw_before = Float64(pw["before_us"]); pw_after = Float64(pw["after_us"])
    dw_before = Float64(dw["before_us"]); dw_after = Float64(dw["after_us"])

    # ── Parse unique trigger time window ─────────────────────────────────
    utw = ds_cfg["unique_trigger_time_window"]
    utw_before = Float64(utw["before_us"]); utw_after = Float64(utw["after_us"])

    # ── Build SiPM ID → column mapping ───────────────────────────────────
    n_total = length(evt)
    global_sipm_set = Set{UInt32}()
    for idx in 1:n_total
        for d in evt.spms.detector[idx]; push!(global_sipm_set, UInt32(d)); end
    end
    sipm_ids = sort!(collect(global_sipm_set))
    n_sipms  = length(sipm_ids)
    sipm_id_to_col = Dict{UInt32, Int}(id => i for (i, id) in enumerate(sipm_ids))

    # ── Accumulators ─────────────────────────────────────────────────────
    pe_rows = Vector{Vector{Float64}}(); pe_rows_p = Vector{Vector{Float64}}(); pe_rows_d = Vector{Vector{Float64}}()
    det_ids = Vector{Vector{UInt32}}()
    sums = Float64[]; mults = Int[]; sums_p = Float64[]; mults_p = Int[]; sums_d = Float64[]; mults_d = Int[]
    ged_dets = UInt32[]; ged_e = Float64[]; ged_t0v = Float64[]
    # Δt = t_max_pe(SiPM mode in [40,60] µs) − t0_hpge
    delta_t_max_pe_us = Float64[]; sizehint!(delta_t_max_pe_us, n_total)
    dt_hist_buf = zeros(Int, _DT_N_TIME_BINS)
    dt_bin_w_us = (_DT_T_HI_US - _DT_T_LO_US) / _DT_N_TIME_BINS
    # Raw triggers in unique window
    trig_det_ids_all = Vector{Vector{UInt32}}()
    trig_times_all   = Vector{Vector{Float64}}()
    trig_pes_all     = Vector{Vector{Float64}}()
    per_det_raw_pe = Dict{UInt32, Vector{Float64}}()
    valid_idxs = Int[]; n_bad = 0

    for idx in 1:n_total
        _event_has_bad_values(evt, idx, has_ged_t0; pe_key, pos_key) && (n_bad += 1; continue)
        push!(valid_idxs, idx)

        t0_us = has_ged_t0 ? prep_to_us(evt.geds.t0_start[idx]) : center_us

        # Summed windows (ns)
        win_lo  = (t0_us - fw_before) * 1000.0; win_hi  = (t0_us + fw_after) * 1000.0
        pwin_lo = (t0_us - pw_before) * 1000.0; pwin_hi = (t0_us + pw_after) * 1000.0
        dwin_lo = (t0_us - dw_before) * 1000.0; dwin_hi = (t0_us + dw_after) * 1000.0
        # Unique trigger window (ns)
        utw_lo = (t0_us - utw_before) * 1000.0; utw_hi = (t0_us + utw_after) * 1000.0

        trig_pos = getproperty(evt.spms, pos_key)[idx]
        trig_dc  = getproperty(evt.spms, dc_key)[idx]
        trig_pe  = getproperty(evt.spms, pe_key)[idx]

        # Summed PE in 3 windows
        pe, s, m   = _windowed_pe(trig_pos, trig_dc, trig_pe, win_lo, win_hi, trig_thresh, mult_thresh)
        pp, sp, mp = _windowed_pe(trig_pos, trig_dc, trig_pe, pwin_lo, pwin_hi, trig_thresh, mult_thresh)
        pd, sd, md = _windowed_pe(trig_pos, trig_dc, trig_pe, dwin_lo, dwin_hi, trig_thresh, mult_thresh)

        push!(pe_rows, pe); push!(pe_rows_p, pp); push!(pe_rows_d, pd)
        push!(det_ids, UInt32[UInt32(d) for d in evt.spms.detector[idx]])
        push!(sums, s); push!(mults, m); push!(sums_p, sp); push!(mults_p, mp); push!(sums_d, sd); push!(mults_d, md)

        # GED info
        if has_ged_t0
            det_idx = max(1, Int(evt.geds.max_e_det_idxs[idx]))
            ged_arr = evt.geds.detector[idx]
            push!(ged_dets, (1 <= det_idx <= length(ged_arr)) ? UInt32(ged_arr[det_idx]) : UInt32(0))
            push!(ged_e, prep_strip(evt.geds.max_e_cusp_ctc_cal[idx]))
            push!(ged_t0v, prep_to_us(evt.geds.t0_start[idx]))
        else
            push!(ged_dets, UInt32(0)); push!(ged_e, NaN); push!(ged_t0v, NaN)
        end

        # Raw triggers in unique window (PE threshold applied)
        evt_trig_det = UInt32[]; evt_trig_t = Float64[]; evt_trig_pe = Float64[]
        for d in eachindex(trig_pe)
            did = UInt32(evt.spms.detector[idx][d])
            for t in eachindex(trig_pe[d])
                Bool(trig_dc[d][t]) && continue
                pe_val = prep_strip(trig_pe[d][t])
                (!isfinite(pe_val) || pe_val < trig_thresh) && continue
                pos_ns = prep_to_ns(trig_pos[d][t])
                utw_lo <= pos_ns <= utw_hi || continue
                push!(evt_trig_det, did)
                push!(evt_trig_t, (pos_ns / 1000.0) - t0_us)  # relative to t0 in μs
                push!(evt_trig_pe, pe_val)
                # Also collect for per-det raw PE distribution
                buf = get!(per_det_raw_pe, did, Float64[])
                push!(buf, pe_val)
            end
        end
        push!(trig_det_ids_all, evt_trig_det)
        push!(trig_times_all, evt_trig_t)
        push!(trig_pes_all, evt_trig_pe)

        # Δt = t_max_pe(SiPM mode in absolute [40,60] µs) − t0_hpge.
        # Per-trigger cleaning: DC excluded + PE ≥ trig_thresh (same as raw-trigger loop).
        fill!(dt_hist_buf, 0)
        for d in eachindex(trig_pe), t in eachindex(trig_pe[d])
            Bool(trig_dc[d][t]) && continue
            pe_val = prep_strip(trig_pe[d][t])
            (!isfinite(pe_val) || pe_val < trig_thresh) && continue
            tp_us = prep_to_ns(trig_pos[d][t]) / 1000.0
            (tp_us < _DT_T_LO_US || tp_us >= _DT_T_HI_US) && continue
            bi = clamp(floor(Int, (tp_us - _DT_T_LO_US) / dt_bin_w_us) + 1, 1, _DT_N_TIME_BINS)
            dt_hist_buf[bi] += 1
        end
        push!(delta_t_max_pe_us,
            if has_ged_t0 && maximum(dt_hist_buf) > 0
                bmax = argmax(dt_hist_buf)
                (_DT_T_LO_US + (bmax - 0.5) * dt_bin_w_us) - t0_us
            else
                NaN
            end)
    end

    # ── Build matrices (events × n_sipms) ────────────────────────────────
    n_events = length(sums)
    mat   = zeros(Float64, n_events, n_sipms)
    mat_p = zeros(Float64, n_events, n_sipms)
    mat_d = zeros(Float64, n_events, n_sipms)
    for (i, (row, rp, rd, dids)) in enumerate(zip(pe_rows, pe_rows_p, pe_rows_d, det_ids))
        for (j, did) in enumerate(dids)
            col = get(sipm_id_to_col, did, 0)
            col > 0 && j <= length(row) && (mat[i,col] = row[j]; mat_p[i,col] = rp[j]; mat_d[i,col] = rd[j])
        end
    end

    n_trigs = sum(length(v) for v in trig_pes_all; init=0)
    stats = Dict{String,Any}(
        "events_read_total" => n_total, "events_nan_inf_removed" => n_bad,
        "events_final" => n_events, "events_zero_pe" => count(==(0.0), sums),
        "n_sipms" => n_sipms, "n_raw_triggers" => n_trigs,
    )
    @info "  WPE complete" dataset=ds_name events=n_events nan_inf=n_bad sipms=n_sipms triggers=n_trigs

    PreparedDataset(ds_name, n_events, n_sipms, sipm_ids,
        mat, mat_p, mat_d, sums, mults, sums_p, mults_p, sums_d, mults_d,
        ged_dets, ged_e, ged_t0v, delta_t_max_pe_us,
        trig_det_ids_all, trig_times_all, trig_pes_all,
        per_det_raw_pe, valid_idxs, stats)
end
export compute_windowed_pe

# ============================================================================
# Merge multiple PreparedDatasets (from batch processing)
# ============================================================================

function _merge_prepared_datasets(preps::Vector{PreparedDataset})
    length(preps) == 1 && return preps[1]
    all_ids = sort!(collect(union(Set.(p.sipm_detector_ids for p in preps)...)))
    n_sipms = length(all_ids)
    id_to_col = Dict{UInt32, Int}(id => i for (i, id) in enumerate(all_ids))
    n_total = sum(p.n_events for p in preps)

    mat = zeros(Float64, n_total, n_sipms); mat_p = zeros(Float64, n_total, n_sipms); mat_d = zeros(Float64, n_total, n_sipms)
    offset = 0
    for p in preps
        for (j, sid) in enumerate(p.sipm_detector_ids)
            col = id_to_col[sid]
            for i in 1:p.n_events
                mat[offset+i, col] = p.sipm_pe_sums[i, j]
                mat_p[offset+i, col] = p.sipm_pe_sums_prompt[i, j]
                mat_d[offset+i, col] = p.sipm_pe_sums_delayed[i, j]
            end
        end
        offset += p.n_events
    end

    merged_raw_pe = Dict{UInt32, Vector{Float64}}()
    for p in preps, (k, v) in p.per_det_raw_trig_pe
        append!(get!(merged_raw_pe, k, Float64[]), v)
    end

    # Merge raw triggers
    merged_trig_det = Vector{Vector{UInt32}}()
    merged_trig_t   = Vector{Vector{Float64}}()
    merged_trig_pe  = Vector{Vector{Float64}}()
    for p in preps
        append!(merged_trig_det, p.trigger_det_ids)
        append!(merged_trig_t, p.trigger_times_us)
        append!(merged_trig_pe, p.trigger_pe_vals)
    end

    PreparedDataset(preps[1].name, n_total, n_sipms, all_ids,
        mat, mat_p, mat_d,
        reduce(vcat, p.event_sum_pe for p in preps),
        reduce(vcat, p.event_multiplicity for p in preps),
        reduce(vcat, p.event_sum_pe_prompt for p in preps),
        reduce(vcat, p.event_multiplicity_prompt for p in preps),
        reduce(vcat, p.event_sum_pe_delayed for p in preps),
        reduce(vcat, p.event_multiplicity_delayed for p in preps),
        reduce(vcat, p.ged_detector_id for p in preps),
        reduce(vcat, p.ged_energy_keV for p in preps),
        reduce(vcat, p.ged_t0_us for p in preps),
        reduce(vcat, p.delta_t_max_pe_us for p in preps),
        merged_trig_det, merged_trig_t, merged_trig_pe,
        merged_raw_pe, collect(1:n_total),
        Dict{String,Any}("events_final" => n_total, "n_sipms" => n_sipms))
end

# ============================================================================
# Write WPE group into LH5 dataset
# ============================================================================

function _write_wpe_group(ds, prep::PreparedDataset)
    wpe = Table(
        sipm_pe_sums               = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums[i, :]) for i in 1:prep.n_events]),
        sipm_pe_sums_prompt        = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums_prompt[i, :]) for i in 1:prep.n_events]),
        sipm_pe_sums_delayed       = VectorOfVectors([Vector{Float64}(prep.sipm_pe_sums_delayed[i, :]) for i in 1:prep.n_events]),
        event_sum_pe               = prep.event_sum_pe,
        event_multiplicity         = Int32.(prep.event_multiplicity),
        event_sum_pe_prompt        = prep.event_sum_pe_prompt,
        event_multiplicity_prompt  = Int32.(prep.event_multiplicity_prompt),
        event_sum_pe_delayed       = prep.event_sum_pe_delayed,
        event_multiplicity_delayed = Int32.(prep.event_multiplicity_delayed),
        ged_detector_id            = prep.ged_detector_id,
        ged_energy_keV             = prep.ged_energy_keV,
        ged_t0_us                  = prep.ged_t0_us,
        delta_t_max_pe_us          = prep.delta_t_max_pe_us,
    )
    ds[:wpe] = _fix_vov(wpe)
    ds["sipm_detector_ids"] = prep.sipm_detector_ids

    # Raw triggers in unique_trigger_time_window
    trigs = Table(
        det_id     = VectorOfVectors(prep.trigger_det_ids),
        time_rel_us = VectorOfVectors(prep.trigger_times_us),
        pe         = VectorOfVectors(prep.trigger_pe_vals),
    )
    ds[:triggers] = _fix_vov(trigs)
end

# ============================================================================
# Distributed Worker — reads jlevt, filters, computes WPE, writes chunk
# ============================================================================

function extract_run_worker(
    period::DataPeriod, run::DataRun, fks::Vector{FileKey},
    ext_specs::Vector, chunk_dir::String; fk_batch_size::Int=20,
)
    try
        l200 = LegendData(:l200)
        n_batches = cld(length(fks), fk_batch_size)

        # Per-spec accumulators
        acc_preps = Dict(ext.name => PreparedDataset[] for ext in ext_specs)
        acc_keys  = Dict(ext.name => Table[] for ext in ext_specs)
        acc_stats = Dict(ext.name => (n_read=0, n_filtered=0) for ext in ext_specs)

        # Per-spec read plan: parsed filter + union of top-level groups
        # referenced by filter and keys_config (drives PropSel of read_ldata)
        spec_plan = map(ext_specs) do ext
            filter_pf  = parse_event_filter(ext.filter_string)
            top_groups = Tuple(unique((
                _propfunc_input_columns(filter_pf)...,
                Symbol.(collect(keys(ext.keys_config)))...,
            )))
            (; ext, filter_pf, top_groups)
        end

        isempty(spec_plan) && return Dict{String, NamedTuple}()

        for b in 1:n_batches
            batch_fks = fks[((b-1)*fk_batch_size+1):min(b*fk_batch_size, length(fks))]

            # Cheap raw event count. The no-det `read_ldata` path on jlevt
            # works against actual top-level subgroups of the tier (`:geds`,
            # `:aux`, ...), not the flat-namespaced leaf names. The previous
            # `(:timestamp,)` failed because there is no top-level `timestamp`.
            # `:geds` is guaranteed to exist (every spec filter references it).
            # Fall back to `-1` if it ever fails so the worker keeps going.
            n_raw = try
                length(read_ldata((:geds,), l200, (DataTier(:jlevt), batch_fks)))
            catch e
                @warn "raw-count read failed, falling back to n_filt for stats" exception=e
                -1
            end

            for sp in spec_plan
                ext = sp.ext

                # Push filter into read_ldata: only filtered events come back.
                # NOTE: HPGe blacklist exclusion is no longer applied here — it
                # lives exclusively in process_balancing (uses the
                # `excluded_ged_detectors:` list from metadata/balancing/<group>.yaml).
                evt_sel = read_ldata(sp.top_groups, l200, (DataTier(:jlevt), batch_fks);
                                     filterby = sp.filter_pf)

                n_filt = length(evt_sel)
                prev = acc_stats[ext.name]
                acc_stats[ext.name] = (n_read=prev.n_read + n_raw, n_filtered=prev.n_filtered + n_filt)
                n_filt == 0 && continue

                push!(acc_keys[ext.name], _fix_vov(select_keys_from_table(evt_sel, ext.keys_config)))
                push!(acc_preps[ext.name], compute_windowed_pe(evt_sel, ext.ds_cfg, ext.name))

                evt_sel = nothing
            end

            GC.gc()
        end

        # Write one chunk file per dataset for this run
        result = Dict{String, NamedTuple}()
        for ext in ext_specs
            preps = acc_preps[ext.name]; tables = acc_keys[ext.name]; stats = acc_stats[ext.name]
            if isempty(preps)
                result[ext.name] = (n_read=stats.n_read, n_filtered=stats.n_filtered,
                                     chunk_path="", error_msg="")
                continue
            end
            combined_tbl  = _fix_vov(reduce(vcat, tables))
            combined_prep = _merge_prepared_datasets(preps)

            chunk_path = joinpath(chunk_dir, "$(ext.name)_$(period)_$(run).lh5")
            lh5open(chunk_path, "w") do ds
                ds[:jlext] = combined_tbl
                _write_wpe_group(ds, combined_prep)
            end
            stats.n_filtered > 0 && @info "  $period-$run [$(ext.name)]: $(stats.n_filtered)/$(stats.n_read)"
            result[ext.name] = (n_read=stats.n_read, n_filtered=stats.n_filtered,
                                 chunk_path=chunk_path, error_msg="")
        end
        return result
    catch e
        @error "Error processing run $period-$run" exception=(e, catch_backtrace())
        err_str = sprint(showerror, e)
        return Dict(ext.name => (n_read=0, n_filtered=0, chunk_path="",
                                  error_msg="$period-$run: $err_str") for ext in ext_specs)
    end
end
export extract_run_worker

# ============================================================================
# Distributed Worker Management
# ============================================================================

function ensure_extraction_workers(processing_config)
    nworkers() > 1 && return

    n_workers_cfg = get(processing_config.processors.process_extraction, :n_workers, 18)
    n_workers = if n_workers_cfg == "all"
        n_tpw = get(processing_config.processors.process_extraction, :threads_per_worker, 3)
        max(1, Sys.CPU_THREADS ÷ (n_tpw + 1))
    else
        Int(n_workers_cfg)
    end
    n_threads = get(processing_config.processors.process_extraction, :threads_per_worker, 3)

    project = Base.active_project()
    total_mem_mb = div(Sys.total_memory(), 1024^2)
    heap_hint_mb = div(total_mem_mb, n_workers + 1)

    @info "Starting $n_workers distributed workers" threads_per_worker=n_threads heap_hint="$(heap_hint_mb)M"
    addprocs(n_workers;
        exeflags=["--project=$project", "--threads=$n_threads", "--heap-size-hint=$(heap_hint_mb)M"],
        env=["OMP_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1"])

    src_dir = joinpath(dirname(@__DIR__), "src")
    load_expr = quote
        using LegendHDF5IO, LegendDataManagement, LegendDataManagement.LDMUtils
        using LegendEventAnalysis
        using PropertyFunctions, TypedTables, PropDicts
        using Unitful, Dates, Printf
        using ArraysOfArrays: VectorOfVectors
        using StructArrays: StructArrays, StructArray, StructVector
        using Tables: Tables
        using YAML
        include(joinpath($src_dir, "io.jl"))
        include(joinpath($src_dir, "extraction.jl"))
    end
    @sync for w in workers()
        @async Distributed.remotecall_eval(Main, w, load_expr)
    end
    @info "Distributed workers ready: $(nworkers()) workers"
end
export ensure_extraction_workers
