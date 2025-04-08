import QuantumInterface: AbstractOperator, AbstractBra, AbstractKet

abstract type QSymbolic{T} end

#const SymQObj = QSymbolic{<:Union{StateVector,AbstractOperator}}

# possibly move to simplified typing after this issue is resolved
# https://github.com/Roger-luo/Moshi.jl/issues/33 
@data BasicQSymExpr{T} <: QSymbolic{T} begin
    struct Const
        val::Number = 1
    end
    struct CNum
        val::Number = 1
    end
    struct QSym
        name::Symbol = :OOF
    end
    struct Term
        f::Function = identity
        arguments::Vector{BasicQSymExpr.Type} = BasicQSymExpr.Type[]
    end
    struct Add
        terms::Set{BasicQSymExpr.Type} = Set{BasicQSymExpr.Type}()
    end
    struct Pow
        base::BasicQSymExpr.Type = BasicQSymExpr.Const(1)
        exp::BasicQSymExpr.Type = BasicQSymExpr.Const(1)
    end
    struct Mul
        coeff::BasicQSymExpr.Type = BasicQSymExpr.Const(0)
        terms::Vector{BasicQSymExpr.Type} = BasicQSymExpr.Type[]
    end
    struct Tensor
        coeff::BasicQSymExpr.Type = BasicQSymExpr.Const(0)
        terms::Vector{BasicQSymExpr.Type} = BasicQSymExpr.Type[]
    end
    struct Sum
        coeff::BasicQSymExpr.Type = BasicQSymExpr.Const(0)
        terms::Vector{BasicQSymExpr.Type} = BasicQSymExpr.Type[]
    end
end

const BasicQSymbolic = BasicQSymExpr.Type
const Const = BasicQSymExpr.Const
const CNum = BasicQSymExpr.CNum
const QSym = BasicQSymExpr.QSym
const Term = BasicQSymExpr.Term
const Add = BasicQSymExpr.Add
const Pow = BasicQSymExpr.Pow
const Mul = BasicQSymExpr.Mul
const Tensor = BasicQSymExpr.Tensor
const Sum = BasicQSymExpr.Sum

const SBra = QSym{AbstractBra}
const SKet = QSym{AbstractKet}
const SOperator = QSym{AbstractOperator}
ishermitian(x::SOperator) = false
isunitary(x::SOperator) = false
