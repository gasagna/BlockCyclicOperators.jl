using BlockCyclicOperators
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Test


@testset "tests                                 " begin

    A1 = [1 2; 3 4]
    A2 = [5 6; 7 8]
    A3 = [9 0; 1 2]
    A = [0 0 0 0 9 0;
         0 0 0 0 1 2;
         1 2 0 0 0 0;
         3 4 0 0 0 0;
         0 0 5 6 0 0;
         0 0 7 8 0 0]
     A_ = BlockCyclicOperator((A1, A2, A3), zeros(Int64, 2))
    
    v  = [1, 2, 3, 4, 5, 6]

    @testset "matmul" begin
        @test A_*v == A*v
        out = similar(v)
        fun(out, A, v) = @allocated LinearAlgebra.mul!(out, A, v)
        @test fun(out, A, v) == 0
    end

    @testset "interface" begin
        @test size(A_) == (6, 6)
        @test A_[1] == A1
        @test A_[2] == A2
        @test A_[3] == A3
    end

end