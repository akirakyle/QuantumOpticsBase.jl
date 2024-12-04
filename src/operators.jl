import Base: ==, +, -, *, /, ^, length, one, exp, conj, conj!, transpose
import LinearAlgebra: tr, ishermitian
import SparseArrays: sparse
import QuantumInterface: AbstractOperator

"""
    Operator{B,T} <: DataOperator{B}

Operator type that stores the representation of an operator on the Hilbert spaces
given by `B<:OperatorBasis` (e.g. a Matrix).
"""
mutable struct Operator{B,T} <: AbstractOperator{B}
    basis::B
    data::T
    function Operator{B,T}(basis::B,data::T) where {B,T}
        (size(basis)==size(data)) || throw(DimensionMismatch("Tried to assign data of size $(size(data)) to basis of size $(size(basis))!"))
        new(basis,data)
    end
end
Operator{B}(basis::B,data::T) where {B,T} = Operator{B,T}(basis,data)
Operator(basis::B,data::T) where {B,T} = Operator{B,T}(basis,data)
Operator(b::Basis,data) = Operator(KetBraBasis(b,b),data)
# FIXME
#Operator(qet1::Ket, qetva::Ket...) = Operator(qet1.basis, GenericBasis(length(qetva)+1), qet1, qetva...)
#Operator(basis_r::Basis,qet1::Ket,qetva::Ket...) = Operator(qet1.basis, basis_r, qet1, qetva...)
#Operator(basis_l::BL,basis_r::BR,qet1::Ket,qetva::Ket...) where {BL,BR} = Operator{BL,BR}(basis_l, basis_r, hcat(qet1.data, getfield.(qetva,:data)...))
#Operator(qets::AbstractVector{<:Ket}) = Operator(first(qets).basis, GenericBasis(length(qets)), qets)
#Operator(basis_r::Basis,qets::AbstractVector{<:Ket}) = Operator(first(qets).basis, basis_r, qets)
#Operator(basis_l::BL,basis_r::BR,qets::AbstractVector{<:Ket}) where {BL,BR} = Operator{BL,BR}(basis_l, basis_r, reduce(hcat, getfield.(qets, :data)))

basis(op::Operator) = op.basis

Base.isequal(x::Operator{B}, y::Operator{B}) where {B} = isequal(x.data, y.data)
Base.(==)(x::Operator{B}, y::Operator{B}) where {B} = x.data==y.data
Base.(==)(x::Operator, y::Operator) = false
Base.isapprox(x::Operator{B}, y::Operator{B}; kwargs...) where {B} = isapprox(x.data, y.data; kwargs...)
Base.isapprox(x::Operator, y::Operator; kwargs...) = false

Base.copy(x::Operator) = Operator(x.basis, copy(x.data))
Base.zero(op::Operator) = Operator(op.basis,zero(op.data))
Base.eltype(op::Operator) = eltype(op.data)
Base.eltype(::Type{T}) where {B,D,T<:Operator{B,D}} = eltype(D)
Base.size(op::Operator) = size(op.data)
Base.size(op::Operator, d) = size(op.data, d)

function Base.convert(::Type{Operator{B,T}}, op::Operator{B,S}) where {B,T,S}
    if T==S
        return op
    else
        return Operator{B,T}(op.basis, convert(T, op.data))
    end
end

# Convert data to CuArray with cu(::Operator)
Adapt.adapt_structure(to, x::Operator) = Operator(x.basis, Adapt.adapt(to, x.data))

# Arithmetic operations
+(a::Operator{B}, b::Operator{B}) where {B} = Operator(a.basis, a.data+b.data)
+(a::Operator, b::Operator) = throw(IncompatibleBases())

-(a::Operator) = Operator(a.basis_l, a.basis_r, -a.data)
-(a::Operator{B}, b::Operator{B}) where {B} = Operator(a.basis, a.data-b.data)
-(a::Operator, b::Operator) = throw(IncompatibleBases())

*(a::Operator{KetBraBasis{BL,BR}}, b::Ket{BR}) where {BL,BR} = Ket{BL}(bases(a).left, a.data*b.data)
*(a::Operator, b::Ket) = throw(IncompatibleBases())
*(a::Bra{BL}, b::Operator{KetBraBasis{BL,BR}}) where {BL,BR} = Bra{BR}(bases(b).right, transpose(b.data)*a.data)
*(a::Bra, b::DataOperator) = throw(IncompatibleBases())
*(a::Operator{KetBraBasis{B1,B2}}, b::Operator{KetBraBasis{B2,B3}}) where {B1,B2,B3} = Operator(KetBraBasis(bases(a).left, bases(b).right), a.data*b.data)
*(a::DataOperator, b::DataOperator) = throw(IncompatibleBases())
#*(a::DataOperator{B1, B2}, b::Operator{B2, B3, T}) where {B1, B2, B3, T} = error("no `*` method defined for DataOperator subtype $(typeof(a))") # defined to avoid method ambiguity
#*(a::Operator{B1, B2, T}, b::DataOperator{B2, B3}) where {B1, B2, B3, T} = error("no `*` method defined for DataOperator subtype $(typeof(b))") # defined to avoid method ambiguity
*(a::Operator, b::Number) = Operator(basis(a), b*a.data)
*(a::Number, b::Operator) = Operator(basis(b), a*b.data)
function *(op1::AbstractOperator{KetBraBasis{B1,B2}}, op2::Operator{KetBraBasis{B2,B3},T}) where {B1,B2,B3,T}
    result = Operator(KetBraBasis(basis(op1).left, basis(op2).right), similar(_parent(op2.data),promote_type(eltype(op1),eltype(op2)),length(basis(op1).left),length(basis(op2).right)))
    mul!(result,op1,op2)
    return result
end
function *(op1::Operator{KetBraBasis{B1,B2},T}, op2::AbstractOperator{KetBraBasis{B2,B3}}) where {B1,B2,B3,T}
    result = Operator(KetBraBasis(basis(op1).left, basis(op2).right), similar(_parent(op1.data),promote_type(eltype(op1),eltype(op2)),length(basis(op1).left),length(basis(op2).right)))
    mul!(result,op1,op2)
    return result
end
function *(op::AbstractOperator{KetBraBasis{BL,BR}}, psi::Ket{BR,T}) where {BL,BR,T}
    result = Ket{BL,T}(basis(op).left, similar(psi.data,length(basis(op).left)))
    mul!(result,op,psi)
    return result
end
function *(psi::Bra{BL,T}, op::AbstractOperator{KetBraBasis{BL,BR}}) where {BL,BR,T}
    result = Bra{BR,T}(basis(op).right, similar(psi.data,length(basis(op).right)))
    mul!(result,psi,op)
    return result
end

_parent(x::T, x_parent::T) where T = x
_parent(x, x_parent) = _parent(x_parent, parent(x_parent))
_parent(x) = _parent(x, parent(x))

/(a::Operator, b::Number) = Operator(a.basis, a.data ./ b)

dagger(x::Operator{KetBraBasis}) = Operator(basis(x).right,basis(x).left,adjoint(x.data))
transpose(x::Operator) = Operator(x.basis_r,x.basis_l,transpose(x.data))
ishermitian(A::DataOperator) = false
ishermitian(A::DataOperator{B,B}) where B = ishermitian(A.data)
Base.collect(A::Operator) = Operator(A.basis_l, A.basis_r, collect(A.data))

tensor(a::Operator, b::Operator) = Operator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

conj(a::Operator) = Operator(a.basis_l, a.basis_r, conj(a.data))
conj!(a::Operator) = (conj!(a.data); a)


"""
    projector(a::Ket, b::Bra)

Projection operator ``|a⟩⟨b|``.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
"""
    projector(a::Ket)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Ket) = Operator(a.basis, a.data*a.data')
"""
    projector(a::Bra)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Bra) = projector(a')

"""
    dm(a::StateVector)

Create density matrix ``|a⟩⟨a|``. Same as `projector(a)`.
"""
dm(x::Ket) = tensor(x, dagger(x))
dm(x::Bra) = tensor(dagger(x), x)



"""
    tensor(x::Ket, y::Bra)

Outer product ``|x⟩⟨y|`` of the given states.
"""
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), length(a.basis), length(b.basis)))

"""
    tensor(a::AbstractOperator, b::Bra)
    tensor(a::Bra, b::AbstractOperator)
    tensor(a::AbstractOperator, b::Ket)
    tensor(a::Ket, b::AbstractOperator)

Products of operators and state vectors ``a ⊗ <b|``. The result is an isometry
in case the operator is unitary and state is normalized.
"""
function tensor(a::AbstractOperator, b::Bra)
    # upgrade the bra to an operator that projects onto a dim-1 space
    # NOTE: copy() works around non-sparse-preserving kron in case b.data is a SparseVector.
    b_op = Operator(GenericBasis(1), basis(b), copy(reshape(b.data, (1,:))))
    ab_op = tensor(a, b_op)
    # squeeze out the trivial dimension
    Operator(a.basis_l, ab_op.basis_r, ab_op.data)
end

function tensor(a::Bra, b::AbstractOperator)
    # upgrade the bra to an operator that projects onto a dim-1 space
    a_op = Operator(GenericBasis(1), basis(a), copy(reshape(a.data, (1,:))))
    ab_op = tensor(a_op, b)
    # squeeze out the trivial dimension
    Operator(b.basis_l, ab_op.basis_r, ab_op.data)
end

function tensor(a::AbstractOperator, b::Ket)
    # upgrade the bra to an operator that projects onto a dim-1 space
    b_op = Operator(basis(b), GenericBasis(1), copy(reshape(b.data, (:,1))))
    ab_op = tensor(a, b_op)
    # squeeze out the trivial dimension
    Operator(ab_op.basis_l, a.basis_r, ab_op.data)
end

function tensor(a::Ket, b::AbstractOperator)
    # upgrade the bra to an operator that projects onto a dim-1 space
    a_op = Operator(basis(a), GenericBasis(1), copy(reshape(a.data, (:,1))))
    ab_op = tensor(a_op, b)
    # squeeze out the trivial dimension
    Operator(ab_op.basis_l, b.basis_r, ab_op.data)
end

tr(op::Operator{B,B}) where B = tr(op.data)

normalize!(op::Operator) = (rmul!(op.data, 1.0/tr(op)); op)

function expect(op::DataOperator{B,B}, state::Ket{B}) where B
    dot(state.data, op.data, state.data)
end

function expect(op::DataOperator{KetBraBasis{B,B}}, state::DataOperator{KetBraBasis{B,B}}) where {B}
    result = zero(promote_type(eltype(op),eltype(state)))
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

# Common error messages
using QuantumInterface: arithmetic_binary_error, arithmetic_unary_error, addnumbererror


"""
    embed(basis1[, basis2], indices::Vector, op::AbstractOperator)

Embed operator acting on a joint Hilbert space where missing indices are filled up with identity operators.
"""
function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
               indices, op::T) where T<:DataOperator
    N = nsubsystems(basis_l)
    @assert nsubsystems(basis_r) == N

    reduce(tensor, bases(basis_l)[indices]) == op.basis_l || throw(IncompatibleBases())
    reduce(tensor, bases(basis_r)[indices]) == op.basis_r || throw(IncompatibleBases())

    index_order = [idx for idx in 1:length(basis_l.bases) if idx ∉ indices]
    all_operators = AbstractOperator[identityoperator(T, eltype(op), basis_l.bases[i], basis_r.bases[i]) for i in index_order]

    for idx in indices
        pushfirst!(index_order, idx)
    end
    push!(all_operators, op)

    check_indices(N, indices)

    # Create the operator.
    permuted_op = tensor(all_operators...)

    # Reorient the matrix to act in the correctly ordered basis.
    # Get the dimensions necessary for index permuting.
    dims_l = [b.shape[1] for b in basis_l.bases]
    dims_r = [b.shape[1] for b in basis_r.bases]

    # Get the order of indices to use in the first reshape. Julia indices go in
    # reverse order.
    expand_order = index_order[end:-1:1]
    # Get the dimensions associated with those indices.
    expand_dims_l = dims_l[expand_order]
    expand_dims_r = dims_r[expand_order]

    # Prepare the permutation to the correctly ordered basis.
    perm_order_l = [indexin(idx, expand_order)[1] for idx in 1:length(dims_l)]
    perm_order_r = [indexin(idx, expand_order)[1] for idx in 1:length(dims_r)]

    # Perform the index expansion, the permutation, and the index collapse.
    M = (reshape(permuted_op.data, tuple([expand_dims_l; expand_dims_r]...)) |>
         x -> permutedims(x, [perm_order_l; perm_order_r .+ length(dims_l)]) |>
         x -> sparse(reshape(x, (prod(dims_l), prod(dims_r)))))

    # Create operator with proper data and bases
    constructor = Base.typename(T)
    unpermuted_op = constructor.wrapper(basis_l, basis_r, M)

    return unpermuted_op
end

function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
                index::Integer, op::T) where {T<:DataOperator}

    N = length(basis_l.bases)

    # Check stuff
    @assert N==length(basis_r.bases)
    basis_l.bases[index] == op.basis_l || throw(IncompatibleBases())
    basis_r.bases[index] == op.basis_r || throw(IncompatibleBases())
    check_indices(N, index)

    # Build data
    Tnum = eltype(op)
    data = similar(sparse(op.data),1,1)
    data[1] = one(Tnum)
    i = N
    while i > 0
        if i == index
            data = kron(data, op.data)
            i -= length(index)
        else
            bl = basis_l.bases[i]
            br = basis_r.bases[i]
            id = SparseMatrixCSC{Tnum}(I, length(bl), length(br))
            data = kron(data, id)
            i -= 1
        end
    end

    return Operator(basis_l, basis_r, data)
end

"""
    expect(op, state)

Expectation value of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
expect(op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{B}) where {B} = dot(state.data, (op * state).data)
# TODO upstream this one
# expect(op::AbstractOperator{B,B}, state::AbstractKet{B}) where B = norm(op * state) ^ 2

function expect(indices, op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{B2}) where {B,B2<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = complement(N, indices)
    expect(op, ptrace(state, indices_))
end

expect(index::Integer, op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{B2}) where {B,B2<:CompositeBasis} = expect([index], op, state)

"""
    variance(op, state)

Variance of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function variance(op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{B}) where {B}
    x = op*state
    state.data'*(op*x).data - (state.data'*x.data)^2
end

function variance(indices, op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{BC}) where {B,BC<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = complement(N, indices)
    variance(op, ptrace(state, indices_))
end

variance(index::Integer, op::AbstractOperator{KetBraBasis{B,B}}, state::Ket{BC}) where {B,BC<:CompositeBasis} = variance([index], op, state)

# Helper functions to check validity of arguments
function check_ptrace_arguments(a::AbstractOperator, indices)
    if !isa(a.basis_l, CompositeBasis) || !isa(a.basis_r, CompositeBasis)
        throw(ArgumentError("Partial trace can only be applied onto operators with composite bases."))
    end
    rank = length(a.basis_l.shape)
    if rank != length(a.basis_r.shape)
        throw(ArgumentError("Partial trace can only be applied onto operators wich have the same number of subsystems in the left basis and right basis."))
    end
    if rank == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(a.basis_l.shape), indices)
    for i=indices
        if a.basis_l.shape[i] != a.basis_r.shape[i]
            throw(ArgumentError("Partial trace can only be applied onto subsystems that have the same left and right dimension."))
        end
    end
end
function check_ptrace_arguments(a::StateVector, indices)
    if length(basis(a).shape) == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    check_indices(length(basis(a).shape), indices)
end

function permutesystems(a::Operator{KetBraBasis{B1,B2}}, perm) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(a.data, [a.basis_l.shape; a.basis_r.shape]...)
    data = permutedims(data, [perm; perm .+ length(perm)])
    data = reshape(data, length(a.basis_l), length(a.basis_r))
    return Operator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), data)
end
permutesystems(a::AdjointOperator{KetBraBasis{M,B1,B2}}, perm) where {M,B1<:CompositeBasis,B2<:CompositeBasis} = dagger(permutesystems(dagger(a),perm))

# Broadcasting
@inline Base.axes(A::DataOperator) = axes(A.data)
Base.broadcastable(A::DataOperator) = A

# Custom broadcasting styles
abstract type DataOperatorStyle{BL,BR} <: Broadcast.BroadcastStyle end
struct OperatorStyle{BL,BR} <: DataOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Operator{BL,BR}}) where {BL,BR} = OperatorStyle{BL,BR}()
Broadcast.BroadcastStyle(::OperatorStyle{B1,B2}, ::OperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

# Broadcast with scalars (of use in ODE solvers checking for tolerances, e.g. `.* reltol .+ abstol`)
Broadcast.BroadcastStyle(::T, ::Broadcast.DefaultArrayStyle{0}) where {Bl<:Basis, Br<:Basis, T<:OperatorStyle{Bl,Br}} = T()

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:OperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    T = find_dType(bcf)
    data = zeros(T, length(bl), length(br))
    @inbounds @simd for I in eachindex(bcf)
        data[I] = bcf[I]
    end
    return Operator{BL,BR}(bl, br, data)
end

find_basis(a::DataOperator, rest) = (a.basis_l, a.basis_r)
find_dType(a::DataOperator, rest) = eltype(a)
@inline Base.getindex(a::DataOperator, idx) = getindex(a.data, idx)
Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(x::DataOperator, i) = x.data[i]
Base.iterate(a::DataOperator) = iterate(a.data)
Base.iterate(a::DataOperator, idx) = iterate(a.data, idx)

# In-place broadcasting
@inline function Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle{BL,BR},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    dest′ = dest.data
    @inbounds @simd for I in eachindex(bc′)
        dest′[I] = bc′[I]
    end
    return dest
end
@inline Base.copyto!(A::DataOperator{BL,BR},B::DataOperator{BL,BR}) where {BL,BR} = (copyto!(A.data,B.data); A)
@inline Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:DataOperatorStyle,Axes,F,Args} =
    throw(IncompatibleBases())

# A few more standard interfaces: These do not necessarily make sense for a StateVector, but enable transparent use of DifferentialEquations.jl
Base.eltype(::Type{Operator{Bl,Br,A}}) where {Bl,Br,N,A<:AbstractMatrix{N}} = N # ODE init
Base.any(f::Function, x::Operator; kwargs...) = any(f, x.data; kwargs...) # ODE nan checks
Base.all(f::Function, x::Operator; kwargs...) = all(f, x.data; kwargs...)
Base.fill!(x::Operator, a) = typeof(x)(x.basis_l, x.basis_r, fill!(x.data, a))
Base.ndims(x::Type{Operator{Bl,Br,A}}) where {Bl,Br,N,A<:AbstractMatrix{N}} = ndims(A)
Base.similar(x::Operator, t) = typeof(x)(x.basis_l, x.basis_r, copy(x.data))
RecursiveArrayTools.recursivecopy!(dest::Operator{Bl,Br,A},src::Operator{Bl,Br,A}) where {Bl,Br,A} = copyto!(dest,src) # ODE in-place equations
RecursiveArrayTools.recursivecopy(x::Operator) = copy(x)
RecursiveArrayTools.recursivecopy(x::AbstractArray{T}) where {T<:Operator} = copy(x)
RecursiveArrayTools.recursivefill!(x::Operator, a) = fill!(x, a)
