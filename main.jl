#!/usr/bin/env -S julia

#############################
# ML-based LAr Veto Training
#############################

println("""
╔════════════════════════════════════════╗
║           ML-based LAr Veto            ║
╚════════════════════════════════════════╝
""")
flush(stdout); flush(stderr)

import Pkg

@info "Using Julia $VERSION"
@info "Using Julia project $(dirname(Pkg.project().path))"
flush(stderr)

# Skip instantiate/precompile when using external environment
if get(ENV, "SKIP_PKG_SETUP", "") != "1"
    Pkg.instantiate()
    Pkg.precompile()
end

# Load packages and config
@info "Loading packages…"; flush(stderr)
t_load = @elapsed include(joinpath(@__DIR__, "src", "startup.jl"))
@info "Packages loaded in $(round(t_load; digits=1)) s"; flush(stderr)

@info "Loading config…"; flush(stderr)
include(joinpath(@__DIR__, "src", "config.jl"))

# Get processing config
l200, processing_config, group_names = get_processingconfig()
@info "Start ML data processing for group(s): $group_names"; flush(stderr)

# Load processors (only include files whose processor is enabled — avoids heavy
# ML/CUDA precompilation when running CPU-only flow steps). Done once, shared
# across all groups in this Julia session.
let proc_dir = joinpath(@__DIR__, "processors")
    enabled = Set(string.(processing_config.enabled_steps))
    for f in sort(readdir(proc_dir))
        endswith(f, ".jl") || continue
        proc_name = replace(f, ".jl" => "")
        if proc_name in enabled || !haskey(processing_config.processors, Symbol(proc_name))
            @info "  Loading processor: $proc_name"; flush(stderr)
            t_proc = @elapsed Base.include(Main, joinpath(proc_dir, f))
            @info "  $proc_name loaded in $(round(t_proc; digits=1)) s"; flush(stderr)
        end
    end
end

# Execute enabled processors in order, looping over all configured groups.
# All groups share the same Julia session, model code, and precompile cache —
# so adding a 2nd group costs only its data + training time, not another
# precompile round.
flush(stdout)

for (gi, group_name) in enumerate(group_names)
    # Keep the legacy single-group field in sync with the loop variable, so
    # `load_metadata_config` (which reads `processing_config.group_name`)
    # picks the correct per-group YAML for each iteration.
    processing_config.group_name = group_name
    @info "╔════════════════════════════════════════╗"
    @info "║  Group $gi/$(length(group_names)): $group_name"
    @info "╚════════════════════════════════════════╝"; flush(stderr)

    for process in processing_config.enabled_steps
        @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        @info "Processor: $(string(process))  [group=$group_name]"

        # Check dependencies
        deps = Symbol.(get(processing_config.processors[process], :dependencies, []))
        completed_steps = Symbol[]
        for prev in processing_config.enabled_steps
            prev == process && break
            push!(completed_steps, prev)
        end

        # Check dependencies - all must be completed (enabled and ran before this step)
        missing_deps = filter(d -> d in processing_config.enabled_steps && !(d in completed_steps), deps)
        if !isempty(missing_deps)
            @error "Dependencies not met - required processors not completed" missing=missing_deps
            continue
        end

        # Get kwargs
        kwargs_dict = get(processing_config.processors[process], :kwargs, PropDict())
        kwargs = NamedTuple{Tuple(keys(kwargs_dict))}(values(kwargs_dict))

        # Call processor
        try
            process_func = getfield(Main, process)
            process_func(processing_config, l200, group_name; kwargs...)
        catch e
            @error "Processor failed" process group=group_name exception=(e, catch_backtrace())
        end
    end
end

@info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
@info "Processing complete (groups: $group_names)"
