#=
@author: William Berdanier

This file calculates the entanglement entropy of the driven XX model (Floquet system).
=#

using Distributions
using Plots

function get_onebody_eigs(F)
    eigF = eigfact(F)
    v = eigF[:vectors]
    w = sort(real(-im * log.(eigF[:values]))) / pi
    plot(w,marker="o")
end

function construct_mat(w_h,w_J,N,epsilon_max=0.1,delta_max=0.1)

    # Disordered couplings
    dist_exp_h = Exponential(w_h)
    dist_exp_J = Exponential(w_J)
    hs = epsilon_max * pi * exp.(-rand(dist_exp_h,div(N,2)))
    Js = delta_max * pi * exp.(-rand(dist_exp_J,div(N-1,2)))

#     # Clean couplings (translation invariant)
#     Jtilde = 0.5
#     @assert Jtilde < pi
#     l1 = Jtilde
#     l2 = Jtilde
#     hs = l1 * pi * ones(div(N,2))
#     Js = l2 * pi * ones(div(N-1,2))

    J1 = zeros(N-1)
    for i in range(1,N-1)
        if (i-1) % 2 == 0
            J1[i] = -hs[div(i+1,2)]
        else
            continue
        end
    end

    J2 = zeros(N-1)
    for i in range(1,N-1)
        if (i-1) % 2 == 0
            continue
        else
            J2[i] = -Js[div(i,2)]
        end
    end

    H1 = Symmetric(Array(SymTridiagonal(zeros(N), J1)))
    H2 = Symmetric(Array(SymTridiagonal(zeros(N), J2)))

    U1 = expm(-im * H1)
    U2 = expm(-im * H2)

    F = U2 * U1
    return F
end

function EE(F,l,eps)
    eigF = eigfact(F)
#     get_onebody_eigs(F)

    v = eigF[:vectors]
    w = real(-im * log.(eigF[:values]))

#     HF = v' * diagm(w) * v
#     HF = 1/2 * (HF + HF') # force HF to be hermitian
#     eigHF = eigfact(HF)
#     v = eigHF[:vectors]'
#     w = eigHF[:values]

    sorted = sortperm(w) # sort the eigenvalues
    v = v[:,sorted]
    w = w[sorted]
    Nf = div(l,2) # fill the fermi sea
#     Nf = sum([1 for n in w if n<0])
    ell = div(l,2) # half system cut

    P = v[1:ell,1:Nf]
    C = P*P'
    eigC = real(eigvals(C))
    half = div(ell,2)

#     # ensure the eigenvalues of C are symmetric
#     symm = [eigC[i] + eigC[ell+1-i] for i in range(1,ell)]
# #     println(minimum(symm))
#     tolerance = 1.0
#     @assert minimum(symm) > 1-tolerance && maximum(symm) < 1 + tolerance

    # method 1 - sum over half the eigenvalues, multiply by 2
    tota = 0.0
    totb = 0.0
    for e in eigC[1:half]
        if ((e<eps) || e>(1.0-eps))
            continue
        else
            tota += 2 * (-(e.*log.(e)+(1.0-e).*log.(1.0-e)))
        end
    end
    for e in eigC[half+1:ell]
        if ((e<eps) || e>(1.0-eps))
            continue
        else
            totb += 2 * (-(e.*log.(e)+(1.0-e).*log.(1.0-e)))
        end
    end
    tot = maximum([tota,totb])

#     # method 2 - sum over all eigenvalues
#     tot2 = 0.0
#     for e in eigC
#         if ((e<eps) || e>(1.0-eps))
#             continue
#         else
#             tot2 += (-(e.*log.(e)+(1.0-e).*log.(1.0-e)))
#         end
#     end
#     tot = tot2

    return tot
end

L =[4,8,12,16,24,36,52,76,108,156,224,324,468,512,672]
@show L
w_h = 1 # don't go above 2 here, too unstable.
w_J = w_h
dis_r = 1000
eps = 10.0^(-10)
epsilon_max = 1.0
delta_max = 1.0

EE_r_tot = [0 for l in L]
EE_avg = []
EE_L = []
c = 1
F = []

println("r_tot: $dis_r")
for r in range(1,dis_r)
        # c = number of successful realizations
        if (r-1) % 5 == 0
            println("r: $r")
        end

        EE_r = [EE(construct_mat(w_h,w_J,l,epsilon_max,delta_max),l,eps) for l in L]

        EE_r_tot += EE_r
        EE_avg = EE_r_tot / c

        EE_L = []

        for i in 1:length(L)
            push!(EE_L,Dict("L"=>L[i],"EE"=>EE_avg[i]))
        end
        c += 1
end
c -= 1

EE_avg = EE_r_tot / c

writedlm("data/EE_avg_w=$w_J-r=$c.out",EE_avg)

using Plots
plotly()

plot(L,[x["EE"] for x in EE_L],xscale=:log10,xlabel="L",ylabel="S(L)",title="$dis_r realizations, w=$w_J",label="data",marker="o")
plot!(L,log(2)/6*log.(L)+0.5,label="c=ln2")
plot!(L,1/6*log.(L)+0.5,label="c=1")
plot!(L,2/6*log.(L)+0.38,label="c=2")
plot!(L,log(2)*2/6*log.(L)+0.38,label="c=2ln2")
