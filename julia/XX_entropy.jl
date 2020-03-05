#=
This file calculates the entanglement entropy of the XX model Hamiltonian (static system).
@author: William Berdanier
=#

using Distributions
function construct_mat(w_h,w_J,N)
    # Disordered couplings:
    dist_exp_h = Exponential(w_h)
    dist_exp_J = Exponential(w_J)
    hs = exp.(-rand(dist_exp_h,div(N,2)))
    Js = exp.(-rand(dist_exp_J,div(N-1,2)))
    J = zeros(N-1)
    for i in range(1,N-1)
        if (i-1) % 2 == 0
            J[i] = hs[div(i+1,2)]
        else
            J[i] = Js[div(i,2)]
        end
    end

#     J = 0.5 * ones(N-1) # Clean couplings (translationally invariant).

    return Symmetric(Array(SymTridiagonal(zeros(N), J)))
end

function EE(H,l,eps)
    #=
    This function calculates the entanglement entropy of the ground state of H,
    for a cut of length l, with regularization parameter epsilon (Schmidt values S < epsilon are set to 0).
    =#
    eigH = eigfact(H)
    v = eigH[:vectors]
    v = transpose(v)
    w = eigH[:values]
    sort = sortperm(w) # Sort the eigenvalues from least to greatest.
    v = v[sort,:]
    w = w[sort]
    Nf = div(l,2) # Fill the Fermi sea, i.e. all negative eigenvalue modes. Due to symmetry, this is half the spectrum.
#     Nf = sum([1 for n in w if n<0]) # We can also explicitly fill the negative ones.
    ell = div(l,2) # Half system cut for the entanglement calculation. This minimizes finite-size effects, see e.g. Cardy-Calabrese papers.

    P = v[1:ell,1:Nf]
    C = Symmetric(P*transpose(P))
    eigC = eigfact(C)[:values]
    tot = 0.0
    for e in eigC
        if ((e<eps) || e>(1.0-eps))
            continue
        else
            tot += (-(e.*log.(e)+(1.0-e).*log.(1.0-e)))
        end
    end
    return tot
end

max_L = 7
L =[4,8,12,16,24,36,52,76,108,156,224,324,468,512,672]
@show L
w_h = 3.
w_J = 3.
dis_r = 500
eps = 10.0^(-10)
EE_r_tot = [0 for l in L]
EE_avg = []
EE_L = []
c = 1
println("r_tot: $dis_r")
for r in range(1,dis_r)
    if (r-1) % 5 == 0
        println("r: $r")
#         display(r) # why is this so buggy in julia
    end

    EE_r = [EE(construct_mat(w_h,w_J,l),l,eps) for l in L]
    EE_r_tot += EE_r
    EE_avg = EE_r_tot / r

    EE_L = []
    for i in 1:length(L)
        push!(EE_L,Dict("L"=>L[i],"EE"=>EE_avg[i]))
    end
    c += 1
end

writedlm("EE_avg_w=$w_J-r=$dis_r.out",EE_avg)

using Plots

plot(L,[x["EE"] for x in EE_L],xscale=:log10,xlabel="L",ylabel="S(L)",title="$dis_r realizations, w=$w_J",label="data",marker="o")
plot!(L,log(2)/6*log.(L)+0.29,label="c=ln2")
plot!(L,1/6*log.(L)+0.29,label="c=1")

dis_r = 3000
L =[4,8,12,16,24,36,52,76,108,156,224,324,468,512,672]
using Plots
plotly()
plot(L,EE_avgs,xscale=:log10,xlabel="L",ylabel="S(L)",title="$dis_r realizations, w=$w_J",label="data",marker="o")
plot!(L,log(2)/6*log.(L)+0.29,label="c=ln2")
plot!(L,1/6*log.(L)+0.29,label="c=1")
