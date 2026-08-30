# ML package imports
#
# Setting DISABLE_GPU=1 forces CPU mode.

using Printf: @sprintf, @printf
using Statistics: mean
using CUDA
using Lux
using NNlib: σ, relu, gelu, swish, leakyrelu
using Optimisers
using MLUtils: DataLoader
using Functors: fmap
using JLD2
# NB: do NOT `using cuDNN` here. cuDNN.jl on this cluster reports
# "not available for your platform (cuda+none)" at precompile but still loads,
# which triggers NNlib's NNlibCUDACUDNNExt extension. That extension routes
# softmax (used by Set-Transformer attention) and BatchNorm through cuDNN,
# which then crashes with `UndefVarError: libcudnn not defined`.
# Skipping the import keeps NNlib on its pure-CUDA softmax path. We don't
# call any cuDNN function explicitly in this codebase, and all BatchNorms
# are configured `batch_norm: false` for the same reason.
using Zygote
using ADTypes: AutoZygote
import Lux.Training

const _GPU_READY = get(ENV, "DISABLE_GPU", "") != "1" && CUDA.functional()
