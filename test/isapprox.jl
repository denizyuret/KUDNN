using CUDArt
import Base: isapprox

function isapprox(x, y; 
                  maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                  rtol::Real=maxeps^(1/3), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    isa(x, CudaArray) && (x = to_host(x))
    isa(y, CudaArray) && (y = to_host(y))
    d = abs(x-y)
    s = abs(x)+abs(y)
    all(d .< (atol + rtol * s))
end
