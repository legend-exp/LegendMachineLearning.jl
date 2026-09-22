# GPU / device selection and memory utilities
#
# Provides:
#   select_device() → (dev_fn, using_gpu::Bool)
#   gpu_sync_gc()   — synchronize GPU + GC
#   _snap(x)        — copy GPU array to CPU

"""Synchronize GPU and force GC + finalization."""
function gpu_sync_gc()
    if _GPU_READY
        CUDA.synchronize()
    end
    GC.gc(true)
    GC.gc(false)
end

"""
    select_device() → (dev_fn, using_gpu::Bool)

Returns a device transfer function and a flag indicating GPU usage.
"""
function select_device()
    if _GPU_READY
        CUDA.allowscalar(false)
        @info "  Device: CUDA GPU — $(CUDA.name(CUDA.device()))"
        return x -> fmap(CuArray, x; exclude = a -> a isa AbstractArray), true
    end
    @info "  Device: CPU"
    return identity, false
end

"""Copy a GPU array to CPU; pass through non-arrays."""
_snap(x::AbstractArray) = Array(x)
_snap(x) = x
