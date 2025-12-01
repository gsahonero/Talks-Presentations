using KomaMRI
using FFTW
using CUDA

sys = Scanner()

# change us to a different integer value for more spins. Beware, us=6 produces 3M spins
phantom = brain_phantom2D(us = 1)

# we read the pulseq file produced by PyPulseq T1w/T2w.ipynb
seq = read_seq("gre_pypulseq.seq")

sim_params = Dict("return_type" => "mat", "method" => Bloch())

data = simulate(phantom, seq, sys; sim_params = sim_params)

# Reshaping to Nx and Ny
reshaped_data = reshape(data, (128, 128))

# We use log for better visualization
plot_image(log.(abs.(reshaped_data)))

plot_image(abs.(ifftshift(ifft(fftshift(reshaped_data)))))


## Modifying the phantom and observing the effects on the simulation

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
plot_phantom_map(new_phantom, :T1)

new_data = simulate(new_phantom, seq, sys; sim_params = sim_params)

reshaped_new_data = reshape(new_data, (128, 128))

plot_image(log.(abs.(reshaped_new_data)))

plot_image(abs.(ifftshift(ifft(fftshift(reshaped_new_data)))))
