module BlockCyclicOperators

import LinearAlgebra

export BlockCyclicOperator

# utils
ithblock(i::Int, n::Int) = ((i-1)*n + 1):(i*n)

# block cyclic matrix
struct BlockCyclicOperator{T, N, AT, VT} <: AbstractMatrix{T}
     As::NTuple{N, AT}
    tmp::NTuple{2, VT}
    function BlockCyclicOperator(As::NTuple{N, AT}, x::VT) where {N, AT, VT}
        n = size(As[1], 1)
        T = eltype(As[1])
        for A in As
              size(A) == (n, n) || throw(ArgumentError("invalid operator size"))
            eltype(A) == T      || throw(ArgumentError("invalid operator type"))
        end
        return new{T, N, AT, VT}(As, (similar(x), similar(x)))
    end
end

Base.getindex(A::BlockCyclicOperator, i::Int) = A.As[i]
Base.size(A::BlockCyclicOperator{T, N}) where {T, N} = (N*size(A[1], 1), N*size(A[1], 2))
LinearAlgebra.issymmetric(A::BlockCyclicOperator) = false

##  matmul
Base.:*(A::BlockCyclicOperator, v::AbstractVector) = LinearAlgebra.mul!(similar(v), A, v)

function LinearAlgebra.mul!(out::AbstractVector,
                              A::BlockCyclicOperator{T, N},
                              v::AbstractVector) where {T, N}
    n = size(A[1], 1)
    @assert length(out) == length(v) == N*n
    for i = 1:N
        @inbounds A.tmp[1] .= v[ithblock(mod1(i-1, N), n)]
        LinearAlgebra.mul!(A.tmp[2], A.As[mod1(i-1, N)], A.tmp[1])
        @inbounds out[ithblock(i, n)] .= A.tmp[2]
    end
    return out
end

end