"""
An `efficientIV` model consists of outcomes, `y` ∈ (0,1),  regressors
`x` and instruments `z`.  The moment condition is
``E[ g(z)(y-x β) ] = 0``

The dimensions of `x`, `y`, and `z` must be such that
`length(y) == size(x,1) == size(z,1)`.
In this version, `size(x,2) = size(z,2)=1`.
"""
mutable struct efficientIV <: GMMModel
  x::Vector{Float64}
  y::Vector{Float64}
  z::Vector{Float64}
end

"""
efficientIV(n::Integer, β::Real, ρ::Real, π::Real, a::AbstractVector)
Simulate an efficientIV model.

# Arguments
- `n` number of observations
- `β` coefficients on `x`
- `π` first stage coefficients `x = z*π + v`
- `ρ` correlation betwwen `v` and `ϵ`
- `a` values that z take
Returns an efficientIV model.
"""

function efficientIV(n::Integer, β::Real, ρ::Real, π::Real, a::AbstractVector)
    # (a) generate z
    w=rand(n)
    w1=(w.<0.2)*1.0
    w2=(w.>=0.2).*(w.<0.4)*1.0
    w3=(w.>=0.4).*(w.<0.6)*1.0
    w4=(w.>=0.6)*1.0
    z=a[1].*w1+a[2].*w2+a[3].*w3+a[4].*w4

    # (b) Matrix of Errors
    A=[1 ρ; ρ 1]
    ev=randn(n,2)*sqrt(A)
    one=ones(n)
    u=(one+z).*ev[:,1]

    # (c) Creating the X, Y vectors
    x = π.*z+ ev[:, 2]
    y = x.*β+u

    #Return the simulated data
    return (efficientIV(x,y,z))
end

number_parameters(model::efficientIV) = size(model.x,2)
number_observations(model::efficientIV) = length(model.y)
number_moments(model::efficientIV) = size(model.x,2)

function gz_hat(model::efficientIV)
    n=number_observations(model)
    β_2sls = (model.z'*model.x)^(-1)*model.z'*model.y
    uhat_2sls = model.y .- model.x .* β_2sls

    zunique=unique(model.z)
    xz4=ones((size(zunique)))
    u2z4=ones((size(zunique)))
    for i=1:size(zunique,1)
        xz4[i]=(sum(model.x.*(model.z.==zunique[i]),dims=1)/sum((model.z.==zunique[i]),dims=1))[1]
    end
    for i=1:size(zunique,1)
        u2z4[i]=(sum(uhat_2sls.*uhat_2sls.*(model.z.==zunique[i]),dims=1)/sum((model.z.==zunique[i]),dims=1))[1]
    end
    ratio4=xz4./u2z4
    gz_hat=ones(n)
    for i=1:n
            for j=1:size(zunique,1)
                if model.z[i]==zunique[j]
                    gz_hat[i]=ratio4[j]
                else
                    gz_hat[i]=gz_hat[i]
                end
            end
        end

    return gz_hat
end

function get_gi(model::efficientIV)
   function(β)
        gz_hat=gz_hat(model::efficientIV)
        ξ=y-model.x.*β
        m=ξ.*gz_hat
   end
end

function estimation(model::efficientIV)
    gzhat=gz_hat(model::efficientIV)
    β_gz_hat=(gzhat'*model.x)^(-1)*gzhat'*model.y
    return (β_gz_hat)
end
