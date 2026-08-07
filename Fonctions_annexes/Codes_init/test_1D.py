import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def simulate_diffusion_1d(n_steps=1000, L=100, base_rate=1.0,
                          defect_density=0.1, defect_strength=0.1, seed=None):
    """
    Simulate the random trajectory of a particle diffusing in a 1D landscape with defects.
    - Each site i has a local jump rate ω_i.
    - Waiting time between jumps is exponential with mean 1/ω_i.
    """
    rng = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    
    # Define spatial map of jump rates
    rates = np.full(L, base_rate)
    n_defects = int(defect_density * L)
    defect_sites = rng.choice(L, n_defects, replace=False)
            
    # rates[defect_sites] *= rng.uniform(0.001,1.0,n_defects)*defect_strength  # defects slow diffusion
    rates[defect_sites] *= defect_strength  # defects slow diffusion
    
    # Initialize state
    pos = L // 2
    positions, times, local_rates = [pos], [0.0], [rates[pos]]
    
    # plt.figure()
    # plt.plot(rates)
    
    t = 0.0
    for _ in range(n_steps):
        rate = rates[pos]
        dt = rng1.exponential(1 / rate)  # draw waiting time
        t += dt
        pos = (pos + rng2.choice([-1, 1])) % L  # random jump, periodic BC
        positions.append(pos)
        times.append(t)
        local_rates.append(rate)
    
    return np.array(positions), np.array(times), rates


def compute_msd(positions, times, L, n_lags=100):

    x_unwrapped = np.unwrap(2 * np.pi * positions / L) * L / (2 * np.pi)


    i_idx, j_idx = np.triu_indices(len(times), k=1)

    pair_t = np.abs(times[j_idx] - times[i_idx])
    pair_p = (x_unwrapped[j_idx] - x_unwrapped[i_idx])**2.0

    # pair_t = pair_t0[pair_t0>1]
    # pair_p = pair_p0[pair_t0>1]
    
    dt_bins = np.logspace(np.log10(pair_t.min()), np.log10(pair_t.max()), n_lags + 1)
    
    dt_centers = np.sqrt(dt_bins[:-1] * dt_bins[1:])  # geometric mean (for log bins)
    
    # Bin the data by Δt
    counts, _ = np.histogram(pair_t, bins=dt_bins)
    sums, _ = np.histogram(pair_t, bins=dt_bins, weights=pair_p)
 
    msd = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    return np.array(dt_centers), np.array(msd)


    

def displacement_pdf(times, positions, n_lags=10):
    
    x_unwrapped = np.unwrap(2 * np.pi * positions / L) * L / (2 * np.pi)


    i_idx, j_idx = np.triu_indices(len(times), k=1)

    pair_t = np.abs(times[j_idx] - times[i_idx])
    pair_p = (x_unwrapped[j_idx] - x_unwrapped[i_idx])

    # pair_t = pair_t0[pair_t0>1]
    # pair_p = pair_p0[pair_t0>1]
    
    dt_bins = np.logspace(np.log10(1.0), np.log10(pair_t.max()), n_lags + 1)
    
    dt_centers = np.sqrt(dt_bins[:-1] * dt_bins[1:])  # geometric mean (for log bins)    

    pdfs = {}

    plt.figure()
    axx=plt.gca()
    # For each target Δt, select pairs and build histogram
    for i in range(n_lags):
        sel = (pair_t > dt_bins[i]) & (pair_t < dt_bins[i+1])
        dx_sel = pair_p [sel]

        if len(dx_sel) == 0:
            continue

        hist, edges = np.histogram(dx_sel,bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        pdfs[i] = (centers, hist)

        # Plot directly
        axx.plot(centers, hist, label="$\delta t =$"+str(dt_bins[i]))

    axx.set_xlabel("$\delta x$")
    axx.set_ylabel("$p(\delta x | \delta t)$")
    axx.set_yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return pdfs

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams['xtick.labelsize']=30
#mpl.rcParams['text.fontsize']=20
mpl.rcParams['axes.labelsize']=35
mpl.rcParams['legend.fontsize']=30
mpl.rcParams['axes.titlesize']=35
mpl.rcParams['figure.figsize']=(12,12)

plt.close('all')

L=300
# Example simulation
positions, times, rates = simulate_diffusion_1d(
    n_steps=20000, L=L, base_rate=1.0, defect_density=0.2,
    defect_strength=0.8, seed=None
)

pos_unwrapped = np.unwrap(2 * np.pi * positions / L) * L / (2 * np.pi)



# Plot trajectory
plt.figure(figsize=(12, 12))
plt.plot(times, pos_unwrapped, lw=1)
plt.xlabel("Time $(t)$")
plt.ylabel("Position $(x)$")
plt.grid(True)
plt.show()



L = len(rates)
fig, ax = plt.subplots(figsize=(12, 12))

# Plot color-coded landscape (rates)
x = np.arange(L)
rate_img = ax.imshow(rates[np.newaxis, :], aspect='auto', cmap='viridis',
                     extent=[0, L, 0, times[-1]], origin='lower')
# ax.set_yticks([])
ax.set_ylabel('Time')
ax.set_xlabel("Position")
ax.plot(positions,times,'.')
# Add colorbar for rates
cbar = plt.colorbar(rate_img, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label("Local jump rate $\omega (x)$", rotation=270, labelpad=20)


# %%
# --- Run simulations for different defect densities ---
densities = [0.0, 0.1, 0.3, 0.5,0.8]
results = {}


plt.figure()
ax0 = plt.gca()

L = 400
for d in densities:
    positions, times, rates = simulate_diffusion_1d(
        n_steps=20000, L=L, base_rate=1.0,
        defect_density=d, defect_strength=0.01, seed=None
    )
    t, msd = compute_msd(positions, times, 200)
    x_unwrapped = np.unwrap(2 * np.pi * positions / L) * L / (2 * np.pi)

    ax0.plot(times,x_unwrapped,label=f"Defect density = {d}")

    
    plt.figure()
    rate_img = plt.imshow(rates[np.newaxis, :], aspect='auto', cmap='viridis',
                         extent=[0, L, 0, times[-1]], origin='lower')
    plt.ylabel('Time')
    plt.xlabel("Position")
    plt.plot(positions,times,'.')
    # Add colorbar for rates
    ax= plt.gca()
    cbar = plt.colorbar(rate_img, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label("Local jump rate $\omega (x)$", rotation=270, labelpad=20)

    results[d] = (t, msd)
ax0.legend()

# --- Plot MSD ---
plt.figure(figsize=(12, 12))
for d, (t, msd) in results.items():
    plt.loglog(t, msd, 'o',label=f"Defect density = {d}")

plt.loglog(t,(t),lw=3,color='k')
plt.xlabel("Time")
plt.ylabel("Mean squared displacement $⟨\Delta x^2⟩$")
plt.legend()
plt.grid(True, which="both", ls="--")


# displacement_pdf(times, positions, n_lags=10)



