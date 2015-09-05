using CUDArt,CUDNN,KUDNN,Base.Test
include("isapprox.jl")

cd4 = ConvolutionDescriptor(padding=(0,0), stride=(1,1), upscale=(1,1), mode=CUDNN_CROSS_CORRELATION)
x1 = CudaArray(rand(10,20,3,4)) # W,H,C,N
w1 = CudaArray(rand(5,4,3,2))   # X1,Y,C,K
y1 = kudnnConvolutionForward(x1, w1; convDesc=cd4)
y2 = cudnnConvolutionForward(x1, w1; convDesc=cd4)
@show map(size, (x1, w1, y1,y2))
@show to_host(y1)==to_host(y2)
@test @show isapprox(y1,y2)
@test @show size(y1) == kudnnGetConvolutionNdForwardOutputDim(x1, w1; convDesc=cd4)

dy1 = CudaArray(rand(size(y1)...))
dw1 = kudnnConvolutionBackwardFilter(x1, dy1, copy(w1); convDesc=cd4)
dw2 = cudnnConvolutionBackwardFilter(x1, dy1, copy(w1); convDesc=cd4)
@show map(size, (x1, w1, dy1, dw1, dw2))
@show to_host(dw1)==to_host(dw2)
@test @show isapprox(dw1,dw2)

dx1 = kudnnConvolutionBackwardData(w1, dy1, copy(x1); convDesc=cd4)
dx2 = cudnnConvolutionBackwardData(w1, dy1, copy(x1); convDesc=cd4)
@show map(size, (x1, w1, dy1, dx1, dx2))
@show to_host(dx1)==to_host(dx2)
@test @show isapprox(dx1,dx2)

cd5 = ConvolutionDescriptor(padding=(0,0,0), stride=(1,1,1), upscale=(1,1,1), mode=CUDNN_CROSS_CORRELATION)
x2 = CudaArray(rand(10,20,30,3,4)) # 1,2,3,C,N
w2 = CudaArray(rand(6,5,4,3,2))   # 1,2,3,C,K
y3 = kudnnConvolutionForward(x2, w2; convDesc=cd5)
@show map(size, (x2, w2, y3))
@test @show size(y3) == kudnnGetConvolutionNdForwardOutputDim(x2, w2; convDesc=cd5)
# y4 = cudnnConvolutionForward(x2, w2; convDesc=cd5)
# @show isapprox(y3,y4)
# @show to_host(y3)==to_host(y4)

dy2 = CudaArray(rand(size(y3)...))
dw3 = kudnnConvolutionBackwardFilter(x2, dy2, copy(w2); convDesc=cd5)
@show map(size, (x2, w2, dy2, dw3))
# dw4 = cudnnConvolutionBackwardFilter(x2, dy2, copy(w2); convDesc=cd5)
# @show isapprox(dw3,dw4)
# @show to_host(dw3)==to_host(dw4)

dx3 = kudnnConvolutionBackwardData(w2, dy2, copy(x2); convDesc=cd5)
@show map(size, (x2, w2, dy2, dx3))
# dx4 = cudnnConvolutionBackwardData(w2, dy2, copy(x2); convDesc=cd5)
# @show isapprox(dx3,dx4)
# @show to_host(dx3)==to_host(dx4)

pd4 = PoolingDescriptor((3,2))
x10 = CudaArray(rand(5,4,3,2)) # W,H,C,N
y10 = similar(x10, kudnnGetPoolingNdForwardOutputDim(pd4,x10))
y11 = kudnnPoolingForward(pd4, x10, copy(y10))
y12 = cudnnPoolingForward(pd4, x10, copy(y10))
@show map(size, (x10,y10))
@show to_host(y11)==to_host(y12)
@test @show isapprox(y11,y12)

dy10 = CudaArray(rand(size(y10)...))
dx10 = similar(x10)
dx11 = kudnnPoolingBackward(pd4, y11, dy10, x10, copy(dx10))
dx12 = cudnnPoolingBackward(pd4, y11, dy10, x10, copy(dx10))
@show to_host(dx11) == to_host(dx12)
@test @show isapprox(dx11,dx12)

pd5 = PoolingDescriptor((3,4,5))
x13 = CudaArray(rand(6,5,4,3,2)) # W,H,C,N
y13 = similar(x13, kudnnGetPoolingNdForwardOutputDim(pd5,x13))
y14 = kudnnPoolingForward(pd5, x13, copy(y13))
# y15 = cudnnPoolingForward(pd4, x10, copy(y10))
@show map(size, (x13,y14))
# @show to_host(y11)==to_host(y12)
# @test @show isapprox(y11,y12)

dy13 = CudaArray(rand(size(y13)...))
dx13 = similar(x13)
dx16 = kudnnPoolingBackward(pd5, y14, dy13, x13, copy(dx13))
# dx17 = cudnnPoolingBackward(pd5, y15, dy13, x13, copy(dx13))
# @show to_host(dx11) == to_host(dx12)
# @test @show isapprox(dx11,dx12)

:ok
