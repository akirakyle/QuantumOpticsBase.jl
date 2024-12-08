import QuantumInterface: KetBraBasis, ChoiRefSysBasis, ChoiOutSysBasis
import FastExpm: fastExpm

"""
    SuperOperator <: AbstractOperator

SuperOperator stored as representation, e.g. as a Matrix.
"""
mutable struct SuperOperator{T} <: AbstractOperator
    basis_l::KetBraBasis
    basis_r::KetBraBasis
    data::T
    function SuperOperator{T}(basis_l::KetBraBasis, basis_r::KetBraBasis, data::T) where {T}
        if (length(basis_l) != size(data, 1) || length(basis_r) != size(data, 2))
            throw(DimensionMismatch("Tried to assign data of size $(size(data)) to Hilbert spaces of sizes $(size(basis_l)), $(size(basis_r))"))
        end
        new(basis_l, basis_r, data)
    end
end
SuperOperator(b1,b2,data::T) where {T} = SuperOperator{T}(b1,b2,data)
SuperOperator(b,data) = SuperOperator(b,b,data)

basis_l(op::SuperOperator) = op.basis_l
basis_r(op::SuperOperator) = op.basis_r

const DenseSuperOpType = SuperOperator{<:Matrix}
const SparseSuperOpType = SuperOperator{<:SparseMatrixCSC}

"""
    DenseSuperOperator(b1[, b2, data])
    DenseSuperOperator([T=ComplexF64,], b1[, b2])

SuperOperator stored as dense matrix.
"""
DenseSuperOperator(basis_l,basis_r,data) = SuperOperator(basis_l, basis_r, Matrix(data))
function DenseSuperOperator(::Type{T}, basis_l, basis_r) where T
    data = zeros(T, length(basis_l), length(basis_r))
    DenseSuperOperator(basis_l, basis_r, data)
end
DenseSuperOperator(basis_l, basis_r) = DenseSuperOperator(ComplexF64, basis_l, basis_r)
DenseSuperOperator(::Type{T}, b) where T = DenseSuperOperator(T, b, b)
DenseSuperOperator(b) = DenseSuperOperator(b,b)


"""
    SparseSuperOperator(b1[, b2, data])
    SparseSuperOperator([T=ComplexF64,], b1[, b2])

SuperOperator stored as sparse matrix.
"""
SparseSuperOperator(basis_l, basis_r, data) = SuperOperator(basis_l, basis_r, sparse(data))

function SparseSuperOperator(::Type{T}, basis_l, basis_r) where T
    data = spzeros(T, length(basis_l), length(basis_r))
    SparseSuperOperator(basis_l, basis_r, data)
end
SparseSuperOperator(basis_l, basis_r) = SparseSuperOperator(ComplexF64, basis_l, basis_r)
SparseSuperOperator(::Type{T}, b) where T = SparseSuperOperator(T, b, b)
SparseSuperOperator(b) = DenseSuperOperator(b,b)

Base.copy(a::T) where {T<:SuperOperator} = T(a.basis_l, a.basis_r, copy(a.data))

dense(a::SuperOperator) = DenseSuperOperator(a.basis_l, a.basis_r, a.data)
sparse(a::SuperOperator) = SuperOperator(a.basis_l, a.basis_r, sparse(a.data))

==(a::SuperOperator, b::SuperOperator) = (addible(a,b) && a.data == b.data)

Base.length(a::SuperOperator) = length(a.basis_l)*length(a.basis_r)

vec(op::Operator) = Ket(KetBraBasis(basis_l(op), basis_r(op)), reshape(op.data, length(op.data)))

# Arithmetic operations
#*(a::SuperOperator, b::Operator) = a*vec(b)
# TODO unify this and multiplicitation for DataOperators
function *(a::SuperOperator, b::Operator)
    check_multiplicable(a,vec(b))
    data = a.data*reshape(b.data, length(b.data))
    return Operator(basis_l(a.basis_l), basis_r(a.basis_l),
                    reshape(data, length(basis_l(a.basis_l)), length(basis_r(a.basis_l))))
end

*(a::SuperOperator, b::SuperOperator) = (check_multiplicable(a,b);
                                         SuperOperator(a.basis_l, b.basis_r, a.data*b.data))

*(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data*b)
*(a::Number, b::SuperOperator) = b*a

/(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data ./ b)

+(a::SuperOperator, b::SuperOperator) = (check_addible(a,b); SuperOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::SuperOperator, b::SuperOperator) = (check_addible(a,b); SuperOperator(a.basis_l, a.basis_r, a.data-b.data))
-(a::SuperOperator) = SuperOperator(a.basis_l, a.basis_r, -a.data)

identitysuperoperator(b::Basis) =
    SuperOperator(KetBraBasis(b,b), KetBraBasis(b,b), Eye{ComplexF64}(length(b)^2))

identitysuperoperator(op::DenseSuperOpType) = 
    SuperOperator(op.basis_l, op.basis_r, Matrix(one(eltype(op.data))I, size(op.data)))

identitysuperoperator(op::SparseSuperOpType) = 
    SuperOperator(op.basis_l, op.basis_r, sparse(one(eltype(op.data))I, size(op.data)))

dagger(x::DenseSuperOpType) = SuperOperator(x.basis_r, x.basis_l, copy(adjoint(x.data)))
dagger(x::SparseSuperOpType) = SuperOperator(x.basis_r, x.basis_l, sparse(adjoint(x.data)))


"""
    spre(op)

Create a super-operator equivalent for right side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spre}(A) B = A B
```

holds. `op` can be a dense or a sparse operator.
"""
function spre(op::AbstractOperator)
    if basis_l(op) != basis_r(op)
        throw(ArgumentError("It's not clear what spre of a non-square operator should be. See issue #113"))
    end
    SuperOperator(KetBraBasis(basis(op)), tensor(op, identityoperator(op)).data)
end

"""
    spost(op)

Create a super-operator equivalent for left side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spost}(A) B = B A
```

holds. `op` can be a dense or a sparse operator.
"""
function spost(op::AbstractOperator)
    if basis_l(op) != basis_r(op)
        throw(ArgumentError("It's not clear what spost of a non-square operator should be. See issue #113"))
    end
    SuperOperator(KetBraBasis(basis(op)), kron(permutedims(op.data), identityoperator(op).data))
end

"""
    sprepost(op)

Create a super-operator equivalent for left and right side operator multiplication.

For operators ``A``, ``B``, ``C`` the relation

```math
    \\mathrm{sprepost}(A, B) C = A C B
```

holds. `A` ond `B` can be dense or a sparse operators.
"""
sprepost(A::AbstractOperator, B::AbstractOperator) =
    SuperOperator(KetBraBasis(A.basis_l, B.basis_r), KetBraBasis(A.basis_r, B.basis_l),
                  kron(permutedims(B.data), A.data))

function _check_input(H::BLROperator{B1,B2}, J::Vector, Jdagger::Vector, rates) where {B1,B2}
    for j=J
        @assert isa(j, BLROperator{B1,B2})
    end
    for j=Jdagger
        @assert isa(j, BLROperator{B1,B2})
    end
    @assert length(J)==length(Jdagger)
    if isa(rates, Matrix{<:Number})
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif isa(rates, Vector{<:Number})
        @assert length(rates) == length(J)
    end
end


"""
    liouvillian(H, J; rates, Jdagger)

Create a super-operator equivalent to the master equation so that ``\\dot ρ = S ρ``.

The super-operator ``S`` is defined by

```math
S ρ = -\\frac{i}{ħ} [H, ρ] + \\sum_i J_i ρ J_i^† - \\frac{1}{2} J_i^† J_i ρ - \\frac{1}{2} ρ J_i^† J_i
```

# Arguments
* `H`: Hamiltonian.
* `J`: Vector containing the jump operators.
* `rates`: Vector or matrix specifying the coefficients for the jump operators.
* `Jdagger`: Vector containing the hermitian conjugates of the jump operators. If they
             are not given they are calculated automatically.
"""
function liouvillian(H, J; rates=ones(length(J)), Jdagger=dagger.(J))
    _check_input(H, J, Jdagger, rates)
    L = spre(-1im*H) + spost(1im*H)
    if isa(rates, AbstractMatrix)
        for i=1:length(J), j=1:length(J)
            jdagger_j = rates[i,j]/2*Jdagger[j]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i,j]*J[i]) * spost(Jdagger[j])
        end
    elseif isa(rates, AbstractVector)
        for i=1:length(J)
            jdagger_j = rates[i]/2*Jdagger[i]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i]*J[i]) * spost(Jdagger[i])
        end
    end
    return L
end

"""
    exp(op::DenseSuperOperator)

Superoperator exponential which can, for example, be used to calculate time evolutions.
Uses LinearAlgebra's `Base.exp`.

If you only need the result of the exponential acting on an operator,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
Base.exp(op::DenseSuperOpType) = DenseSuperOperator(op.basis_l, op.basis_r, exp(op.data))

"""
    exp(op::SparseSuperOperator; opts...)

Superoperator exponential which can, for example, be used to calculate time evolutions.
Uses [`FastExpm.jl.jl`](https://github.com/fmentink/FastExpm.jl) which will return a sparse
or dense operator depending on which is more efficient.
All optional arguments are passed to `fastExpm` and can be used to specify tolerances.

If you only need the result of the exponential acting on an operator,
consider using much faster implicit methods that do not calculate the entire exponential.
"""
function Base.exp(op::SparseSuperOpType; opts...)
    if iszero(op)
        return identitysuperoperator(op)
    else
        return SuperOperator(op.basis_l, op.basis_r, fastExpm(op.data; opts...))
    end
end

# Array-like functions
Base.zero(A::SuperOperator) = SuperOperator(A.basis_l, A.basis_r, zero(A.data))
Base.size(A::SuperOperator) = size(A.data)
@inline Base.axes(A::SuperOperator) = axes(A.data)
Base.ndims(A::SuperOperator) = 2
Base.ndims(::Type{<:SuperOperator}) = 2

# Broadcasting
Base.broadcastable(A::SuperOperator) = A

# Custom broadcasting styles
struct SuperOperatorStyle <: Broadcast.BroadcastStyle end
# struct DenseSuperOperatorStyle{BL,BR} <: SuperOperatorStyle{BL,BR} end
# struct SparseSuperOperatorStyle{BL,BR} <: SuperOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:SuperOperator}) = SuperOperatorStyle()
# Broadcast.BroadcastStyle(::Type{<:SparseSuperOperator{BL,BR}}) where {BL,BR} = SparseSuperOperatorStyle{BL,BR}()
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B1,B2}) where {B1,B2} = DenseSuperOperatorStyle{B1,B2}()
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::DenseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
# Broadcast.BroadcastStyle(::SparseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())
# Broadcast.BroadcastStyle(::DenseSuperOperatorStyle{B1,B2}, ::SparseSuperOperatorStyle{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {Style<:SuperOperatorStyle,Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    return SuperOperator(bl, br, copy(bc_))
end
# @inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL,BR,Style<:SparseSuperOperatorStyle{BL,BR},Axes,F,Args<:Tuple}
#     bcf = Broadcast.flatten(bc)
#     bl,br = find_basis(bcf.args)
#     bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
#     return SuperOperator{BL,BR}(bl, br, copy(bc_))
# end
find_basis(a::SuperOperator, rest) = (a.basis_l, a.basis_r)

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*),typeof(/)}
function Broadcasted_restrict_f(f::BasicMathFunc, args::Tuple{Vararg{<:SuperOperator}}, axes)
    args_ = Tuple(a.data for a=args)
    return Broadcast.Broadcasted(f, args_, axes)
end

# In-place broadcasting
@inline function Base.copyto!(dest::SuperOperator, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {Style<:SuperOperatorStyle,Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:SuperOperator}) # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    # Get the underlying data fields of operators and broadcast them as arrays
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    copyto!(dest.data, bc_)
    return dest
end
@inline Base.copyto!(A::SuperOperator,B::SuperOperator) = (copyto!(A.data,B.data); A)

# TODO make sure copyto! checks basis appropriately and throws error
#@inline function Base.copyto!(dest::SuperOperator, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {Style<:SuperOperatorStyle,Axes,F,Args}
#    throw(IncompatibleBases())
#end

"""
    ChoiState <: AbstractOperator

Superoperator represented as a choi state.

The convention is chosen such that the reference/input operators live in `(basis_l[1], basis_r[1])` while
the output operators live in `(basis_r[2], basis_r[2])`.
"""
mutable struct ChoiState{T} <: AbstractOperator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    data::T
    function ChoiState{T}(basis_l::CompositeBasis, basis_r::CompositeBasis, data::T) where {T}
        if !(nsubsystems(basis_l) == nsubsystems(basis_r) == 2 &&
            basis_l[1] isa ChoiRefSysBasis && basis_r[1] isa ChoiRefSysBasis &&
            basis_l[2] isa ChoiOutSysBasis && basis_r[2] isa ChoiOutSysBasis)
            throw(ArgumentError("Choi State must be have appropriate bases..."))
        end
        if (length(basis_l) != size(data, 1) || length(basis_r) != size(data, 2))
            throw(DimensionMismatch("Tried to assign data of size $(size(data)) to Hilbert spaces of sizes $(size(basis_l)), $(size(basis_r))"))
        end
        new(basis_l, basis_r, data)
    end
end
ChoiState(b1, b2, data::T) where {T} = ChoiState{T}(b1, b2, data)

basis_l(op::ChoiState) = op.basis_l
basis_r(op::ChoiState) = op.basis_r

dense(a::ChoiState) = ChoiState(a.basis_l, a.basis_r, Matrix(a.data))
sparse(a::ChoiState) = ChoiState(a.basis_l, a.basis_r, sparse(a.data))
dagger(a::ChoiState) = ChoiState(dagger(SuperOperator(a)))
*(a::ChoiState, b::ChoiState) = ChoiState(SuperOperator(a)*SuperOperator(b))
*(a::ChoiState, b::Operator) = SuperOperator(a)*b
==(a::ChoiState, b::ChoiState) = (addible(a,b); a.data == b.data)

# reshape swaps within systems due to colum major ordering
# https://docs.qojulia.org/quantumobjects/operators/#tensor_order
function _super_choi((l1, l2), (r1, r2), data)
    data = reshape(data, map(length, (l2, l1, r2, r1)))
    (l1, l2), (r1, r2) = (r2, l2), (r1, l1)
    data = permutedims(data, (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    return (l1, l2), (r1, r2), data
end

function _super_choi((r2, l2), (r1, l1), data::SparseMatrixCSC)
    data = _permutedims(data, map(length, (l2, r2, l1, r1)), (1, 3, 2, 4))
    data = reshape(data, map(length, (l1⊗l2, r1⊗r2)))
    # sparse(data) is necessary since reshape of a sparse array returns a
    # ReshapedSparseArray which is not a subtype of AbstractArray and so
    # _permutedims fails to acces the ".m" field
    # https://github.com/qojulia/QuantumOpticsBase.jl/pull/83
    # https://github.com/JuliaSparse/SparseArrays.jl/issues/24
    # permutedims in SparseArrays.jl only implements perm (2,1) and so
    # _permutedims should probably be upstreamed
    # https://github.com/JuliaLang/julia/issues/26534
    return (l1, l2), (r1, r2), sparse(data)
end

function ChoiState(op::SuperOperator)
    #ChoiState(_super_choi(op.basis_l, op.basis_r, op.data)...)
    bl = (basis_l(op.basis_l), basis_r(op.basis_l))
    br = (basis_l(op.basis_r), basis_r(op.basis_r))
    bl, br, d = _super_choi(bl, br, op.data)
    ChoiState(ChoiRefSysBasis(bl[1])⊗ChoiOutSysBasis(bl[2]),
              ChoiRefSysBasis(br[1])⊗ChoiOutSysBasis(br[2]), d)
end

function SuperOperator(op::ChoiState)
    #SuperOperator(_super_choi(op.basis_l, op.basis_r, op.data)...)
    bl = (op.basis_l[1].basis, op.basis_l[2].basis)
    br = (op.basis_r[1].basis, op.basis_r[2].basis)
    bl, br, d = _super_choi(bl, br, op.data)
    SuperOperator(KetBraBasis(bl[1],bl[2]),
                  KetBraBasis(br[1],br[2]), d)
end

*(a::ChoiState, b::SuperOperator) = SuperOperator(a)*b
*(a::SuperOperator, b::ChoiState) = a*SuperOperator(b)

