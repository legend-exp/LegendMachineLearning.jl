# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Shared I/O helpers — path builders, VoV fix, metadata writing

using LegendHDF5IO: lh5open
using ArraysOfArrays: VectorOfVectors, flatview, element_ptr
using TypedTables: Table, columnnames
using Tables: Tables

# ============================================================================
# _fix_vov — ensure VoV structures round-trip through LH5 correctly
# ============================================================================

_fix_vov(x) = x
_fix_vov(x::AbstractVector{<:AbstractVector}) = VectorOfVectors(x)
_fix_vov(x::VectorOfVectors{<:AbstractVector}) = VectorOfVectors(VectorOfVectors(flatview(x)), x.elem_ptr)
_fix_vov(t::Table) = Table(NamedTuple{Tuple(propertynames(t))}(
    [if c isa Table
         Table(StructArray(map(_fix_vov, Tables.columns(c))))
     else
         _fix_vov(c)
     end for c in Tables.columns(t)]
))

export _fix_vov

# ============================================================================
# Path Builders
# ============================================================================

"""
    get_tier_path(processing_config, tier, group_name, dataset_name) → String
    get_tier_path(output_base::String, tier, group_name, dataset_name) → String

Standard output path for tier files.
Format: `{output_base}/{tier}/{group}/l200-{group}-{dataset}-tier_{tier}.lh5`
"""
function get_tier_path(processing_config::PropDict, tier::String, group_name::String, dataset_name::String)
    get_tier_path(String(processing_config.paths.output.tier), tier, group_name, dataset_name)
end

function get_tier_path(output_base::String, tier::String, group_name::String, dataset_name::String)
    output_dir = joinpath(output_base, tier, group_name)
    mkpath(output_dir)
    joinpath(output_dir, "l200-$(group_name)-$(dataset_name)-tier_$(tier).lh5")
end

"""
    get_plot_dir(processing_config, group_name, processor) → String

Directory for plot output: `{output.plots}/{group}/{processor}/`.
Creates the directory if it does not exist.
"""
function get_plot_dir(processing_config::PropDict, group_name::String, processor::Symbol)
    d = joinpath(processing_config.paths.output.plots, group_name, string(processor))
    mkpath(d)
    d
end

"""
    get_report_dir(processing_config, group_name) → String

Directory for reports: `{output.reports}/{group}/`.
Creates the directory if it does not exist.
"""
function get_report_dir(processing_config::PropDict, group_name::String)
    d = joinpath(processing_config.paths.output.reports, group_name)
    mkpath(d)
    d
end

export get_tier_path, get_plot_dir, get_report_dir

# ============================================================================
# Metadata Writing
# ============================================================================

"""
    write_extraction_metadata(ds, extraction_name, group_name, filter_str, n_read, n_filtered)

Write extraction metadata to an open LH5 dataset.
"""
function write_extraction_metadata(ds, extraction_name::String, group_name::String,
                                   filter_str::String, n_read::Int, n_filtered::Int)
    ds[:metadata] = (
        extraction_name = extraction_name,
        group_name = group_name,
        filter = filter_str,
        generated_at = string(Dates.now()),
        events_read = n_read,
        events_extracted = n_filtered,
        extraction_efficiency = n_filtered / max(n_read, 1)
    )
end
export write_extraction_metadata

# ============================================================================
# Dataset Iteration Helper
# ============================================================================

"""
    iterate_all_datasets(extraction_config) → Dict{String, Any}

Flatten `datasets.training` and `datasets.physics` into a single Dict.
"""
function iterate_all_datasets(extraction_config::Dict)
    ds = extraction_config["datasets"]
    all = Dict{String, Any}()
    for category in ("training", "physics")
        if haskey(ds, category)
            for (name, cfg) in ds[category]
                all[String(name)] = cfg
            end
        end
    end
    all
end
export iterate_all_datasets

# ============================================================================
# Period / Run Parsing
# ============================================================================

"""
    parse_period_run(period_str::String, runs_def) → (DataPeriod, Vector{DataRun})

Parse period string ("p16") and runs definition (vector or scalar) into typed objects.
"""
function parse_period_run(period_str::String, runs_def)
    period = DataPeriod(parse(Int, replace(period_str, "p" => "")))
    runs = if runs_def isa AbstractVector
        [DataRun(parse(Int, replace(string(r), "r" => ""))) for r in runs_def]
    else
        [DataRun(parse(Int, replace(string(runs_def), "r" => "")))]
    end
    period, runs
end
export parse_period_run

# ============================================================================
# ML Group Utilities
# ============================================================================

"""
    get_ml_grouping(groupings_path::String, group_name::String)

Load ML grouping definition from YAML file.  Returns Dict of period => runs.
"""
function get_ml_grouping(groupings_path::String, group_name::String)
    groupings = YAML.load_file(groupings_path)
    haskey(groupings, group_name) || error("Group '$group_name' not found in $(basename(groupings_path))")
    groupings[group_name]
end
export get_ml_grouping

# ============================================================================
# LH5 Table I/O — read + cast detector IDs
# ============================================================================

"""
    _cast_detector_ids(tbl)

Post-read cast: convert `detector` VoV{UInt32} columns back to VoV{DetectorId}.
LegendHDF5IO serializes DetectorId as UInt32 and does not restore the type on read.
"""
function _cast_detector_ids(tbl)
    groups = Dict{Symbol, Any}()
    for gname in columnnames(tbl)
        grp = getproperty(tbl, gname)
        if hasproperty(grp, :detector)
            det_col = grp.detector
            if eltype(flatview(det_col)) === UInt32
                flat_ids = DetectorId.(flatview(det_col))
                new_det = VectorOfVectors(flat_ids, element_ptr(det_col))
                nt = NamedTuple{columnnames(grp)}(
                    Tuple(k == :detector ? new_det : getproperty(grp, k) for k in columnnames(grp))
                )
                groups[gname] = Table(nt)
            else
                groups[gname] = grp
            end
        else
            groups[gname] = grp
        end
    end
    ksyms = Tuple(collect(keys(groups)))
    Table(NamedTuple{ksyms}(Tuple(groups[k] for k in ksyms)))
end

"""
    read_lh5_table(path::String; key::Symbol=:jlext) → Table

Read an LH5 table, fix VoV round-trip issues, and cast detector IDs.
General-purpose reader for any tier stored in LH5 format.
"""
function read_lh5_table(path::String; key::Symbol=:jlext)
    isfile(path) || error("LH5 file not found: $path")
    raw = lh5open(path, "r") do ds; ds[string(key)][:]; end
    _cast_detector_ids(_fix_vov(raw))
end
export read_lh5_table

"""
    read_lh5_metadata(path::String; group::String="metadata")

Read metadata group from an LH5 file.  Returns `nothing` if the group does not exist.
"""
function read_lh5_metadata(path::String; group::String="metadata")
    isfile(path) || error("LH5 file not found: $path")
    lh5open(path, "r") do ds
        haskey(ds.data_store, group) ? ds[group] : (@warn "No $group in $path"; nothing)
    end
end
export read_lh5_metadata

# ============================================================================
# Data Structures
# ============================================================================

mutable struct PreparedDataset
    name::String
    n_events::Int
    n_sipms::Int

    sipm_detector_ids::Vector{UInt32}

    # Per-detector PE sums: (n_events × n_sipms)
    sipm_pe_sums::Matrix{Float64}
    sipm_pe_sums_prompt::Matrix{Float64}
    sipm_pe_sums_delayed::Matrix{Float64}

    # Event-level aggregates (full / prompt / delayed window)
    event_sum_pe::Vector{Float64}
    event_multiplicity::Vector{Int}
    event_sum_pe_prompt::Vector{Float64}
    event_multiplicity_prompt::Vector{Int}
    event_sum_pe_delayed::Vector{Float64}
    event_multiplicity_delayed::Vector{Int}

    # GED info
    ged_detector_id::Vector{UInt32}
    ged_energy_keV::Vector{Float64}
    ged_t0_us::Vector{Float64}

    # Δt = t_max_pe(SiPM mode in absolute [40,60] µs) − ged_t0_us; NaN if undefined
    delta_t_max_pe_us::Vector{Float64}

    # Raw triggers in unique_trigger_time_window (flat per event)
    trigger_det_ids::Vector{Vector{UInt32}}
    trigger_times_us::Vector{Vector{Float64}}
    trigger_pe_vals::Vector{Vector{Float64}}

    per_det_raw_trig_pe::Dict{UInt32, Vector{Float64}}
    valid_indices::Vector{Int}
    stats::Dict{String, Any}
end
export PreparedDataset

# ============================================================================
# Windowed PE I/O — write/read :wpe group inside jlext LH5 files
# ============================================================================

"""
    _read_wpe_from_lh5(path) → PreparedDataset

Read windowed PE results from the `:wpe` group of an LH5 file (chunk or final).
"""
function _read_wpe_from_lh5(path::String)
    isfile(path) || error("LH5 file not found: $path")

    local wpe, sipm_ids, trigs
    has_triggers = false
    lh5open(path, "r") do ds
        haskey(ds.data_store, "wpe") || error("No :wpe group in $(basename(path))")
        wpe = ds["wpe"][:]
        sipm_ids = ds["sipm_detector_ids"][:]
        if haskey(ds.data_store, "triggers")
            trigs = ds["triggers"][:]
            has_triggers = true
        end
    end

    n_events = length(wpe.event_sum_pe)
    n_sipms  = length(sipm_ids)

    _vov_to_matrix(vov, ne, ns) = begin
        m = zeros(Float64, ne, ns)
        for i in 1:ne
            row = vov[i]
            for j in 1:min(length(row), ns); m[i, j] = Float64(row[j]); end
        end; m
    end

    mat   = _vov_to_matrix(wpe.sipm_pe_sums, n_events, n_sipms)
    mat_p = hasproperty(wpe, :sipm_pe_sums_prompt)  ? _vov_to_matrix(wpe.sipm_pe_sums_prompt, n_events, n_sipms)  : zeros(Float64, n_events, n_sipms)
    mat_d = hasproperty(wpe, :sipm_pe_sums_delayed) ? _vov_to_matrix(wpe.sipm_pe_sums_delayed, n_events, n_sipms) : zeros(Float64, n_events, n_sipms)

    has_p = hasproperty(wpe, :event_sum_pe_prompt)
    has_d = hasproperty(wpe, :event_sum_pe_delayed)

    if has_triggers
        trig_det = [Vector{UInt32}(t) for t in trigs.det_id]
        trig_t   = [Vector{Float64}(t) for t in trigs.time_rel_us]
        trig_pe  = [Vector{Float64}(t) for t in trigs.pe]
    else
        trig_det = [UInt32[] for _ in 1:n_events]
        trig_t   = [Float64[] for _ in 1:n_events]
        trig_pe  = [Float64[] for _ in 1:n_events]
    end

    delta_t = hasproperty(wpe, :delta_t_max_pe_us) ?
              Vector{Float64}(wpe.delta_t_max_pe_us) :
              fill(NaN, n_events)

    PreparedDataset(
        "", n_events, n_sipms, Vector{UInt32}(sipm_ids),
        mat, mat_p, mat_d,
        Vector{Float64}(wpe.event_sum_pe),
        Vector{Int}(wpe.event_multiplicity),
        has_p ? Vector{Float64}(wpe.event_sum_pe_prompt) : zeros(Float64, n_events),
        has_p ? Vector{Int}(wpe.event_multiplicity_prompt) : zeros(Int, n_events),
        has_d ? Vector{Float64}(wpe.event_sum_pe_delayed) : zeros(Float64, n_events),
        has_d ? Vector{Int}(wpe.event_multiplicity_delayed) : zeros(Int, n_events),
        Vector{UInt32}(wpe.ged_detector_id),
        Vector{Float64}(wpe.ged_energy_keV),
        Vector{Float64}(wpe.ged_t0_us),
        delta_t,
        trig_det, trig_t, trig_pe,
        Dict{UInt32, Vector{Float64}}(),
        collect(1:n_events),
        Dict{String, Any}(),
    )
end
export _read_wpe_from_lh5
