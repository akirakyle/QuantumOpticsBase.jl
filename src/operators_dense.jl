import Base: +, -, *, /, Broadcast
import Adapt
using Base.Cartesian

const DenseOpPureType{B} = Operator{B,<:Matrix}
const DenseOpAdjType{B} = Operator{B,<:Adjoint{<:Number,<:Matrix}}
const DenseOpType{B} = Union{DenseOpPureType{B},DenseOpAdjType{B}}
const AdjointOperator{B} = Operator{B,<:Adjoint}

"""
    DenseOperator(b1[, b2, data])

Dense array implementation of Operator. Converts any given data to a dense `Matrix`.
"""
DenseOperator(basis::OperatorBasis, data::T) where T = Operator(basis, Matrix(data))
DenseOperator(basis::OperatorBasis, data::Matrix) = Operator(basis, data)
DenseOperator(b::Basis, data) = DenseOperator(KetBraBasis(b, b), data)
# FIXME
#DenseOperator(::Type{T},b1::Basis,b2::Basis) where T = Operator(b1,b2,zeros(T,length(b1),length(b2)))
#DenseOperator(::Type{T},b::Basis) where T = Operator(b,b,zeros(T,length(b),length(b)))
#DenseOperator(b1::Basis, b2::Basis) = DenseOperator(ComplexF64, b1, b2)
#DenseOperator(b::Basis) = DenseOperator(ComplexF64, b)
DenseOperator(op::Operator) = DenseOperator(op.basis, Matrix(op.data))

"""
    dense(op::AbstractOperator)

Convert an arbitrary Operator into a [`DenseOperator`](@ref).
"""
dense(x::AbstractOperator) = DenseOperator(x)

"""
    exp(op::DenseOpType)

Operator exponential used, for example, to calculate displacement operators.
Uses LinearAlgebra's `Base.exp`.

If you only need the result of the exponential acting on a vector,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
exp(op::DenseOpType) = DenseOperator(op.basis, exp(op.data))

# TODO: does this even make sense when length(b1) != length(b2)?
identityoperator(::Type{S}, ::Type{T}, b1::Basis, b2::Basis) where {S<:DenseOpType,T<:Number} =
    Operator(KetBraBasis(b1, b2), Matrix{T}(I, length(b1), length(b2)))

function ptrace(a::DataOperator, indices)
    check_ptrace_arguments(a, indices)
    rank = length(a.basis_l.shape)
    result = _ptrace(Val{rank}, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end
ptrace(op::AdjointOperator, indices) = dagger(ptrace(op, indices))

function ptrace(psi::Ket, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_ket(Val{rank}, psi.data, b.shape, indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end

function ptrace(psi::Bra, indices)
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_bra(Val{rank}, psi.data, b.shape, indices)::Matrix{eltype(psi)}
    return Operator(b_, b_, result)
end


# Partial trace implementation for dense operators.
function _strides(shape)
    N = length(shape)
    S = zeros(eltype(shape), N)
    S[1] = 1
    for m=2:N
        S[m] = S[m-1]*shape[m-1]
    end
    return S
end

function _strides(shape::Ty)::Ty where Ty <: Tuple
    accumulate(*, (1,Base.front(shape)...))
end

# Dense operator version
@generated function _ptrace(::Type{Val{RANK}}, a,
                            shape_l, shape_r,
                            indices) where RANK
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = copy(shape_l)
        @inbounds for idx ∈ indices
            result_shape_l[idx] = 1
        end
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = copy(shape_r)
        @inbounds for idx ∈ indices
            result_shape_r[idx] = 1
        end
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(eltype(a), N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

@generated function _ptrace_ket(::Type{Val{RANK}}, a,
                            shape, indices) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        @inbounds for idx ∈ indices
            result_shape[idx] = 1
        end
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(eltype(a), N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0]*conj(a[Ir_0])
            end
        end
        return result
    end
end

@generated function _ptrace_bra(::Type{Val{RANK}}, a,
                            shape, indices) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        @inbounds for idx ∈ indices
            result_shape[idx] = 1
        end
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(eltype(a), N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += conj(a[Il_0])*a[Ir_0]
            end
        end
        return result
    end
end

"""
    mul!(Y::DataOperator,A::AbstractOperator,B::DataOperator,alpha,beta) -> Y
    mul!(Y::StateVector,A::AbstractOperator,B::StateVector,alpha,beta) -> Y

Fast in-place multiplication of operators/state vectors. Updates `Y` as
`Y = alpha*A*B + beta*Y`. In most cases, the call gets forwarded to
Julia's 5-arg mul! implementation on the underlying data.
See also [`LinearAlgebra.mul!`](@ref).
"""
mul!(result::Operator{B1,B3},a::Operator{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3} = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Ket{B1},a::Operator{B1,B2},b::Ket{B2},alpha,beta) where {B1,B2} = (LinearAlgebra.mul!(result.data,a.data,b.data,alpha,beta); result)
mul!(result::Bra{B2},a::Bra{B1},b::Operator{B1,B2},alpha,beta) where {B1,B2} = (LinearAlgebra.mul!(result.data,transpose(b.data),a.data,alpha,beta); result)
rmul!(op::Operator, x) = (rmul!(op.data, x); op)

# Multiplication for Operators in terms of their gemv! implementation
function mul!(result::Operator{B1,B3},M::AbstractOperator{B1,B2},b::Operator{B2,B3},alpha,beta) where {B1,B2,B3}
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        mul!(resultket,M,bket,alpha,beta)
        result.data[:,i] = resultket.data
    end
    return result
end

function mul!(result::Operator{B1,B3},b::Operator{B1,B2},M::AbstractOperator{B2,B3},alpha,beta) where {B1,B2,B3}
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        mul!(resultbra,bbra,M,alpha,beta)
        result.data[i,:] = resultbra.data
    end
    return result
end

