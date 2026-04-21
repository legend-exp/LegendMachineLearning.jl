# ML package imports
#
# Setting DISABLE_GPU=1 forces CPU mode.

using Printf: @sprintf, @printf
using Statistics: mean
using CUDA
using Lux
using NNlib: σ, relu, gelu, swish
using Optimisers
using MLUtils: DataLoader
using Functors: fmap
using JLD2
using cuDNN
using Zygote
using ADTypes: AutoZygote
import Lux.Training

const _GPU_READY = get(ENV, "DISABLE_GPU", "") != "1" && CUDA.functional()
