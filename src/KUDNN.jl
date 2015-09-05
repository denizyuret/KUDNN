module KUDNN

using CUDArt,CUDNN
const libkudnn = Libdl.find_library(["libkudnn"], [Pkg.dir("KUDNN/src")])
isempty(libkudnn) && error("Cannot find libkudnn")

export kudnnConvolutionForward, kudnnConvolutionBackwardFilter, kudnnConvolutionBackwardData, kudnnGetConvolutionNdForwardOutputDim
export kudnnPoolingForward, kudnnPoolingBackward, kudnnGetPoolingNdForwardOutputDim

# high level C interface (following CUDNN.jl)

using CUDNN: cudnnHandle, ConvolutionDescriptor, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, TD, FD, cptr

# TODO: do not construct this every time:
kudnnConvolutionDescriptor(n)=ConvolutionDescriptor(; padding=ntuple(i->0,n), stride=ntuple(i->1,n), upscale=ntuple(i->1,n), mode=CUDNN_CROSS_CORRELATION)

function kudnnGetConvolutionNdForwardOutputDim(src, filter; convDesc=kudnnConvolutionDescriptor(ndims(src)-2))
    nd = ndims(src)
    nd == ndims(filter) || error("Dimension mismatch")
    size(src,nd-1) == size(filter,nd-1) || error("Dimension mismatch")
    od = Array(Int,nd)
    for i=1:nd-2
        od[i] = 1 + div(size(src,i) - size(filter,i) + 2*convDesc.padding[i], convDesc.stride[i])
    end
    od[nd-1] = size(filter,nd)
    od[nd] = size(src,nd)
    tuple(od...)
end

function kudnnConvolutionForward(src::AbstractCudaArray, filter::AbstractCudaArray, dest=nothing;
                                 handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                 convDesc=kudnnConvolutionDescriptor(ndims(src)-2),
                                 algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                 workSpace=Int8[], workSpaceSizeInBytes=0)
    @assert ndims(filter) == ndims(src)
    @assert eltype(filter) == eltype(src)
    osize = kudnnGetConvolutionNdForwardOutputDim(src,filter;convDesc=convDesc)
    (dest == nothing) && (dest = CudaArray(eltype(src), osize))
    @assert osize == size(dest)
    @assert eltype(dest) == eltype(src)
    kudnnConvolutionForward(handle,
                            cptr(alpha,src),TD(src),src,
                            FD(filter),filter,
                            convDesc,algorithm,workSpace,workSpaceSizeInBytes,
                            cptr(beta,dest),TD(dest),dest)
    return dest
end


function kudnnConvolutionBackwardFilter(src::AbstractCudaArray, diff::AbstractCudaArray, grad::AbstractCudaArray;
                                        handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                        convDesc=kudnnConvolutionDescriptor(ndims(src)-2))
    kudnnConvolutionBackwardFilter(handle,
                                   cptr(alpha,src),TD(src),src,
                                   TD(diff),diff,convDesc,
                                   cptr(beta,grad),FD(grad),grad)
    return grad
end


function kudnnConvolutionBackwardData(filter::AbstractCudaArray, diff::AbstractCudaArray, grad::AbstractCudaArray;
                                      handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                      convDesc=kudnnConvolutionDescriptor(ndims(filter)-2))
    kudnnConvolutionBackwardData(handle,cptr(alpha,diff),
                                 FD(filter),filter,
                                 TD(diff),diff,convDesc,
                                 cptr(beta,grad),TD(grad),grad)
    return grad
end


function kudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, input::AbstractCudaArray)
    n = ndims(input)
    ntuple(n) do i
        i >= n-1 ? size(input, i) :
        1 + ceil(Int, (size(input, i) + 2*pd.padding[i] - pd.dims[i]) / pd.stride[i])
    end
end

function kudnnPoolingForward(pd::PoolingDescriptor, src::AbstractCudaArray, dest::AbstractCudaArray; 
                             handle=cudnnHandle, alpha=1.0, beta=0.0)
    kudnnPoolingForward(handle, pd, 
                        cptr(alpha,src), TD(src), src,
                        cptr(beta,dest), TD(dest), dest)
    return dest
end

function kudnnPoolingBackward(pd::PoolingDescriptor, src::AbstractCudaArray, srcDiff::AbstractCudaArray, dest::AbstractCudaArray, destDiff::AbstractCudaArray; 
                              handle=cudnnHandle, alpha=1.0, beta=0.0)
    cudnnSetTensor(destDiff, 0)  # TODO: make sure kudnnPoolingBackward zeros out its output
    kudnnPoolingBackward(handle, pd, 
                         cptr(alpha,src), TD(src), src, 
                         TD(srcDiff), srcDiff, 
                         TD(dest), dest,
                         cptr(beta,destDiff), TD(destDiff), destDiff)
    return destDiff
end


# low level C interface (similar to libcudnn.jl)

using CUDNN: cudnnCheck,cudnnStatus_t,cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,cudnnPoolingDescriptor_t

function kudnnConvolutionForward(handle,alpha,srcDesc,srcData,filterDesc,filterData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,destDesc,destData)
    cudnnCheck(ccall((:kudnnConvolutionForward,libkudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,filterDesc,filterData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,destDesc,destData))
end

function kudnnConvolutionBackwardFilter(handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
    cudnnCheck(ccall((:kudnnConvolutionBackwardFilter,libkudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,beta,gradDesc,gradData))
end

function kudnnConvolutionBackwardData(handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
    cudnnCheck(ccall((:kudnnConvolutionBackwardData,libkudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,beta,gradDesc,gradData))
end

function kudnnPoolingForward(handle,poolingDesc,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:kudnnPoolingForward,libkudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function kudnnPoolingBackward(handle,poolingDesc,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
    cudnnCheck(ccall((:kudnnPoolingBackward,libkudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData))
end

end # module KUDNN
