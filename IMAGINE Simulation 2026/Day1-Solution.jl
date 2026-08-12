## Simple exercises using KomaMRI.jl
using KomaMRI
using FFTW
using CUDA

sys = Scanner()

# change us to a different integer value for more spins. Beware, us=6 produces 3M spins
phantom = brain_phantom2D(us = 2)

# Visualize T1
display(plot_phantom_map(phantom, :T1))

# Visualize T2
display(plot_phantom_map(phantom, :T2))

# Visualize T2s
display(plot_phantom_map(phantom, :T2s))

# Visualize Δw
display(plot_phantom_map(phantom, :Δw))

# Visualize ρ
display(plot_phantom_map(phantom, :ρ))

## Using phantom indexing.

phantom = brain_phantom2D(us = 1)

# Visualize only a part of the phantom
display(plot_phantom_map(phantom[1000:3000], :T1))

x = zeros(length(phantom))
y = zeros(length(phantom))
z = zeros(length(phantom))
T1 = zeros(length(phantom))
T2 = zeros(length(phantom))
T2s = zeros(length(phantom))
Δw = zeros(length(phantom))
ρ = zeros(length(phantom))
for (index, spin_i) in enumerate(phantom)
    x[index] = phantom[index].x[1]
    y[index] = phantom[index].y[1]
    z[index] = phantom[index].z[1]
    if index >= 1000 && index <= 3000
        T1[index] = phantom[index].T1[1]*0.01
    end
    T2[index] = phantom[index].T2[1]
    T2s[index] = phantom[index].T2s[1]
    ρ[index] = phantom[index].ρ[1]
    Δw[index] = phantom[index].Δw[1]

end

new_phantom = Phantom(x = x, y = y, z = z, T1 = T1, T2 = T2, T2s = T2s, ρ = ρ, Δw = Δw)
display(plot_phantom_map(new_phantom, :T1))

## Creating a phantom

function square_phantom(L, Δx, Δy, T1, T2, T2s, ρ, Δw)
    # Number of points along each dimension
    Nx = floor(Int, L / Δx) + 1
    Ny = floor(Int, L / Δy) + 1

    # Coordinates
    x = Float64[]
    y = Float64[]
    z = Float64[]

    for i in 0:Nx-1
        for j in 0:Ny-1
            push!(x, Float64(i * Δx))
            push!(y, Float64(j * Δy))
            push!(z, 0.0f0)
        end
    end

    N = length(x)

    return Phantom{Float64}(
        x = x,
        y = y,
        z = z,
        T1 = fill(T1, N),
        T2 = fill(T2, N),
        T2s = fill(T2s, N),
        ρ = fill(ρ, N),
        Δw = fill(Δw, N)
    )
end
sq_phantom = square_phantom(
    128e-3,   # 128 mm side length
    1e-3,     # 1 mm spacing in x
    1e-3,     # 1 mm spacing in y
    1000e-3,  # T1 = 1000 ms
    100e-3,   # T2 = 100 ms
    80e-3,    # T2* = 80 ms
    1.0f0,    # proton density
    0.0f0     # frequency offset
)

display(plot_phantom_map(sq_phantom, :T1))