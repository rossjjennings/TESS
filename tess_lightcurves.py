


def tess_lightcurves(filename,TIC):

    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    
    tpf_url = str(filename)
    with fits.open(tpf_url) as hdus:
        tpf = hdus[1].data
        tpf_hdr = hdus[1].header
        tpf_mask = hdus[2].data
    
    time = tpf["TIME"]
    flux = tpf["FLUX"]
    m = np.any(np.isfinite(flux), axis=(1, 2)) & (tpf["QUALITY"] == 0)
    time = np.ascontiguousarray(time[m] - np.min(time[m]), dtype=np.float64)
    flux = np.ascontiguousarray(flux[m], dtype=np.float64)
    
    mean_img = np.median(flux, axis=0)
    plt.imshow(mean_img.T, cmap="gray_r")
    plt.title("TESS image of Pi Men")
    plt.xticks([])
    plt.yticks([]);
    
    from scipy.signal import savgol_filter
    
    # Sort the pixels by median brightness
    order = np.argsort(mean_img.flatten())[::-1]
    
    # A function to estimate the windowed scatter in a lightcurve
    def estimate_scatter_with_mask(mask):
        f = np.sum(flux[:, mask], axis=-1)
        smooth = savgol_filter(f, 1001, polyorder=5)
        return 1e6 * np.sqrt(np.median((f / smooth - 1)**2))
    
    # Loop over pixels ordered by brightness and add them one-by-one
    # to the aperture
    masks, scatters = [], []
    for i in range(10, 100):
        msk = np.zeros_like(tpf_mask, dtype=bool)
        msk[np.unravel_index(order[:i], mean_img.shape)] = True
        scatter = estimate_scatter_with_mask(msk)
        masks.append(msk)
        scatters.append(scatter)
    
    # Choose the aperture that minimizes the scatter
    pix_mask = masks[np.argmin(scatters)]
    
    # Plot the selected aperture
    plt.imshow(mean_img.T, cmap="gray_r")
    plt.imshow(pix_mask.T, cmap="Reds", alpha=0.3)
    plt.title("TIC"+str(TIC)+": selected aperture ")
    plt.xticks([])
    plt.yticks([])
#    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_aperture.pdf')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sap_flux = np.sum(flux[:, pix_mask], axis=-1)
    sap_flux = (sap_flux / np.median(sap_flux) - 1) * 1e3
    plt.plot(time, sap_flux, "k")
    plt.xlabel("time [days]")
    plt.ylabel("relative flux [ppt]")
    plt.title("TIC"+str(TIC)+": raw light curve")
    plt.xlim(time.min(), time.max())
#    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_raw_lc.pdf')
    plt.close()
    
    # Build the first order PLD basis
    X_pld = np.reshape(flux[:, pix_mask], (len(flux), -1))
    X_pld = X_pld / np.sum(flux[:, pix_mask], axis=-1)[:, None]
    
    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, :X_pld.shape[1]]
    
    # Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((np.ones((len(flux), 1)), X_pld, X2_pld), axis=-1)
    XTX = np.dot(X_pld.T, X_pld)
    w_pld = np.linalg.solve(XTX, np.dot(X_pld.T, sap_flux))
    pld_flux = np.dot(X_pld, w_pld)
    
    # Plot the de-trended light curve
    plt.figure(figsize=(10, 5))
    plt.plot(time, sap_flux-pld_flux, "k")
    plt.xlabel("time [days]")
    plt.ylabel("de-trended flux [ppt]")
    plt.title("TIC"+str(TIC)+": initial de-trended light curve")
    plt.xlim(time.min(), time.max())
#    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_detrend_lc.pdf')
    plt.close()
    
    from astropy.stats import BoxLeastSquares
    
    period_grid = np.exp(np.linspace(np.log(1), np.log(15), 50000))
    
    bls = BoxLeastSquares(time, sap_flux - pld_flux)
    bls_power = bls.power(period_grid, 0.1, oversample=20)
    
    # Save the highest peak as the planet candidate
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index]
    bls_t0 = bls_power.transit_time[index]
    bls_depth = bls_power.depth[index]
    transit_mask = bls.transit_mask(time, bls_period, 0.2, bls_t0)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot the periodogram
    ax = axes[0]
    ax.axvline(np.log10(bls_period), color="C1", lw=5, alpha=0.8)
    ax.plot(np.log10(bls_power.period), bls_power.power, "k")
    ax.annotate("period = {0:.4f} d".format(bls_period),
                (0, 1), xycoords="axes fraction",
                xytext=(5, -5), textcoords="offset points",
                va="top", ha="left", fontsize=12)
    ax.set_ylabel("bls power")
    ax.set_yticks([])
    ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
    ax.set_xlabel("log10(period)")
    
    # Plot the folded transit
    ax = axes[1]
    x_fold = (time - bls_t0 + 0.5*bls_period)%bls_period - 0.5*bls_period
    m = np.abs(x_fold) < 0.4
    ax.plot(x_fold[m], sap_flux[m] - pld_flux[m], ".k")
    
    # Overplot the phase binned light curve
    bins = np.linspace(-0.41, 0.41, 32)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=sap_flux - pld_flux)
    denom[num == 0] = 1.0
    ax.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1")
    
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylabel("de-trended flux [ppt]")
    ax.set_xlabel("time since transit")
    plt.title("TIC"+str(TIC))
    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_period_detrend_lc.pdf')
    plt.close()
    
    m = ~transit_mask
    XTX = np.dot(X_pld[m].T, X_pld[m])
    w_pld = np.linalg.solve(XTX, np.dot(X_pld[m].T, sap_flux[m]))
    pld_flux = np.dot(X_pld, w_pld)
    
    x = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(sap_flux-pld_flux, dtype=np.float64)
    
    plt.figure(figsize=(10, 5))
    plt.plot(time, y, "k")
    plt.xlabel("time [days]")
    plt.ylabel("de-trended flux [ppt]")
    plt.title("TIC"+str(TIC)+": final de-trended light curve")
    plt.xlim(time.min(), time.max())
#    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_detrend_lc_final.pdf')
    plt.close()

    
    plt.figure(figsize=(10, 5))
    
    x_fold = (x - bls_t0 + 0.5*bls_period) % bls_period - 0.5*bls_period
    m = np.abs(x_fold) < 0.3
    plt.plot(x_fold[m], pld_flux[m], ".k", ms=4)
    
    bins = np.linspace(-0.5, 0.5, 60)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=pld_flux)
    denom[num == 0] = 1.0
    plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1", lw=2)
    plt.xlim(-0.2, 0.2)
    plt.title("TIC"+str(TIC))
    plt.xlabel("time since transit")
    plt.ylabel("PLD model flux")
#    plt.savefig('/Users/Admin/Documents/Research_Lisa/TESS/plots/'+str(TIC)+'_PLD_model.pdf')
    plt.close()

name = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0002/6113/6679/tess2018206045859-s0001-0000000261136679-0120-s_tp.fits"
name2 = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0003/0003/3922/tess2018206045859-s0001-0000000300033922-0120-s_tp.fits"
name3 = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0002/6113/6679/tess*_tp.fits"

tess_lightcurves(name,261136679)