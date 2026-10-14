# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Extraction Processor — thin orchestrator (Juleana pattern)
#
# Single step: distributed workers read jlevt → filter → compute WPE → write chunk
# Then combine chunks into final jlext (with :jlext keys + :wpe group).

using ProgressMeter

# ============================================================================
# Main Extraction Orchestrator
# ============================================================================

function process_extraction(processing_config::PropDict, l200::LegendData, group_name::String)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Extraction for ML group: $group_name"

    # ── Load config ──────────────────────────────────────────────────────
    group_def = get_ml_grouping(processing_config.paths.groupings, group_name)

    extraction_config = load_metadata_config(processing_config, :extraction)
    if isnothing(extraction_config)
        @error "No extraction config found"; return false
    end

    all_datasets = iterate_all_datasets(extraction_config)
    @info "$(length(all_datasets)) datasets"

    # NOTE: HPGe / SiPM exclusion is now handled exclusively by the balancing
    # processor (see metadata/balancing/<group>.yaml `excluded_ged_detectors:` /
    # `excluded_sipm_detectors:`). Extraction writes all events; balancing drops
    # the excluded ones afterwards. Keeps extraction free of detector-list deps.

    output_base = processing_config.paths.output.tier

    # ── Build extraction specs ───────────────────────────────────────────
    # Per-dataset `enabled: true|false` (default true) lets users selectively
    # re-extract only certain datasets (extraction is by far the slowest step).
    # Disabled datasets are skipped entirely — their existing jlext file (if any)
    # from a previous run is left intact so balancing/normalization can still use
    # it. Only enabled datasets get their jlext recreated.
    ext_specs = NamedTuple[]
    skipped_disabled = String[]
    for (ds_name, ds_cfg) in all_datasets
        if !Bool(get(ds_cfg, "enabled", true))
            push!(skipped_disabled, ds_name)
            continue
        end

        output_path = get_tier_path(output_base, "jlext", group_name, ds_name)
        isfile(output_path) && rm(output_path)

        filter_string = get(ds_cfg, "event_filter", "")
        isempty(filter_string) && (@error "No event_filter: $ds_name"; continue)
        parse_event_filter(filter_string)  # validate

        push!(ext_specs, (
            name = ds_name,
            filter_string = filter_string,
            keys_config = get(ds_cfg, "keys", Dict()),
            ds_cfg = ds_cfg,
            output_path = output_path,
        ))
    end
    if !isempty(skipped_disabled)
        @info "Skipping disabled datasets: $(join(skipped_disabled, ", "))"
    end
    isempty(ext_specs) && (@warn "All datasets disabled — nothing to extract"; return true)

    wpe_results = Dict{String, PreparedDataset}()
    # Per-dataset status carried into the report so it always reflects what
    # happened, even when datasets fail.  `error_msgs` is the (possibly empty)
    # list of unique worker error strings encountered for the dataset.
    ds_status = Dict{String, NamedTuple}()
    for ext in ext_specs
        ds_status[ext.name] = (n_read=0, n_filtered=0, n_chunks=0,
                                status="not_run", error_msgs=String[])
    end

    if !isempty(ext_specs)
        timer = TimerOutput()
        run_filekeys = collect_run_filekeys(l200, group_def)
        n_total_fk = sum(length(fks) for (_, _, fks) in run_filekeys; init=0)

        if n_total_fk > 0
            @info "$n_total_fk filekeys in $(length(run_filekeys)) runs"
            ensure_extraction_workers(processing_config)

            chunk_dir = joinpath(output_base, "jlext", group_name, "chunks")
            mkpath(chunk_dir)

            worker_specs = [(name=e.name, filter_string=e.filter_string,
                            keys_config=e.keys_config, ds_cfg=e.ds_cfg) for e in ext_specs]
            fk_batch_size = Int(get(processing_config.processors.process_extraction, :fk_batch_size, 20))

            @timeit timer "Distributed extract+WPE" begin
                results = pmap(run_filekeys) do (period, run, fks)
                    extract_run_worker(period, run, fks, worker_specs, chunk_dir; fk_batch_size)
                end
            end

            # ── Surface worker errors in the master log (otherwise they are
            #    only visible in the worker stdout stream) ─────────────────
            worker_errors = Dict{String, Vector{String}}()
            for res in results, (ds_name, st) in res
                em = get(st, :error_msg, "")
                isempty(em) || push!(get!(worker_errors, ds_name, String[]), em)
            end
            for (ds_name, errs) in worker_errors
                @error "Extraction failed in $(length(errs))/$(length(results)) run(s)" dataset=ds_name first_error=first(errs)
            end

            @timeit timer "Combine chunks" begin
                for ext in ext_specs
                    chunk_files = String[]
                    n_read_total = 0; n_filtered_total = 0

                    for res in results
                        haskey(res, ext.name) || continue
                        stats = res[ext.name]
                        n_read_total += stats.n_read
                        n_filtered_total += stats.n_filtered
                        !isempty(stats.chunk_path) && isfile(stats.chunk_path) && push!(chunk_files, stats.chunk_path)
                    end

                    errs = unique(get(worker_errors, ext.name, String[]))

                    if isempty(chunk_files)
                        @warn "No chunks produced for $(ext.name)"
                        ds_status[ext.name] = (n_read=n_read_total, n_filtered=n_filtered_total,
                                                n_chunks=0,
                                                status=isempty(errs) ? "no_data" : "failed",
                                                error_msgs=errs)
                        continue
                    end

                    @info "Combining $(length(chunk_files)) chunks for $(ext.name)"

                    # Read keys + WPE from chunks, combine, write final file
                    all_keys = Table[]; all_preps = PreparedDataset[]
                    for cf in chunk_files
                        tbl = lh5open(cf, "r") do ds; ds["jlext"][:]; end
                        push!(all_keys, tbl)
                        push!(all_preps, _read_wpe_from_lh5(cf))
                    end

                    combined_keys = _fix_vov(reduce(vcat, all_keys))
                    combined_prep = _merge_prepared_datasets(all_preps)
                    combined_prep.name = ext.name
                    # Inject true raw/filtered counts into stats so the report
                    # shows them (compute_windowed_pe only knows post-filter).
                    combined_prep.stats["events_read"]         = n_read_total
                    combined_prep.stats["events_after_filter"] = n_filtered_total

                    lh5open(ext.output_path, "w") do ds
                        ds[:jlext] = combined_keys
                        _write_wpe_group(ds, combined_prep)
                        write_extraction_metadata(ds, ext.name, group_name, ext.filter_string, n_read_total, n_filtered_total)
                    end

                    wpe_results[ext.name] = combined_prep
                    ds_status[ext.name] = (n_read=n_read_total, n_filtered=n_filtered_total,
                                            n_chunks=length(chunk_files),
                                            status=isempty(errs) ? "ok" : "partial",
                                            error_msgs=errs)
                    @info "$(ext.name): $(combined_prep.n_events) events (read: $n_read_total, filtered: $n_filtered_total)"
                end
            end

            rm(chunk_dir; recursive=true, force=true)
            nworkers() > 1 && rmprocs(workers())
            show(timer); println()
        else
            @warn "No filekeys found"
        end
    end

    # ── Plots ────────────────────────────────────────────────────────────
    if !isempty(wpe_results)
        plot_dir = get_plot_dir(processing_config, group_name, :extraction)
        for (ds_name, ds) in wpe_results
            preliminary = get(get(processing_config, :plots, PropDict()), :preliminary, true)
            save_extraction_plots(ds, group_name, plot_dir, all_datasets[ds_name];
                                   preliminary, l200, group_def)
            save_t0_diff_plots(ds, group_name, plot_dir; preliminary)
        end
    end

    # ── Report ALWAYS — so failures are immediately visible ──────────────
    try
        generate_extraction_report(wpe_results, ds_status,
            get_report_dir(processing_config, group_name), group_name, l200)
    catch e
        @error "Failed to write extraction report" exception=(e, catch_backtrace())
    end

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    n_ok = count(s -> s.status in ("ok",), values(ds_status))
    n_total = length(ds_status)
    @info "Extraction complete: $n_ok/$n_total datasets ok"
    return n_ok == n_total
end
