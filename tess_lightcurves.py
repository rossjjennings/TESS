import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from astropy.stats import BoxLeastSquares # astropy.timeseries?
import os

def tess_lightcurves(filename, TIC, output_dir):
    
    time, flux, tpf_hdr, tpf_mask = unpack(filename)
    mean_img = np.median(flux, axis=0)
    plot_img(mean_img)
    
    pix_mask = get_aperture(flux, mean_img, tpf_mask)
    plot_aperture(TIC, mean_img, pix_mask)
    sap_flux = simple_aperture_photometry(flux, pix_mask)
    plot_lightcurve(TIC, time, sap_flux, "raw light curve")
    
    pld_flux = pixel_level_deconvolution(flux, sap_flux, pix_mask)
    lightcurve = sap_flux - pld_flux
    plot_lightcurve(TIC, time, lightcurve, "initial de-trended light curve")
    bls, bls_power, period_grid = periodogram(time, lightcurve)
    
    # Save the highest peak as the planet candidate
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index]
    bls_t0 = bls_power.transit_time[index]
    bls_depth = bls_power.depth[index]
    transit_mask = bls.transit_mask(time, bls_period, 0.2, bls_t0)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_periodogram(axes[0], bls_power, bls_period, period_grid)
    plot_transit(axes[1], time, lightcurve, bls_period, bls_t0)
    
    plt.title("TIC"+str(TIC))
    plt.savefig(os.path.join(output_dir, str(TIC)+'_period_detrend_lc.pdf'))
    plt.close()
    
    pld_flux = pixel_level_deconvolution(flux, sap_flux, pix_mask, transit_mask)
    final_lightcurve = sap_flux - pld_flux
    plot_lightcurve(TIC, time, final_lightcurve, "final de-trended light curve")
    plot_folded_pldmodel(TIC, time, pld_flux, bls_period, bls_t0)

def unpack(filename):
    
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
    
    return time, flux, tpf_hdr, tpf_mask

def plot_img(img):
    
    plt.imshow(img.T, cmap="gray_r")
    plt.title("TESS image of Pi Men")
    plt.xticks([])
    plt.yticks([]);
    
def get_aperture(flux, img, tpf_mask):
    
    mean_img = np.median(flux, axis=0)
    
    # Sort the pixels by median brightness
    order = np.argsort(img.flatten())[::-1]
    
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
    return pix_mask

def plot_aperture(TIC, img, pix_mask):
    # Plot the selected aperture
    plt.imshow(img.T, cmap="gray_r")
    plt.imshow(pix_mask.T, cmap="Reds", alpha=0.3)
    plt.title("TIC"+str(TIC)+": selected aperture ")
    plt.xticks([])
    plt.yticks([])
    plt.close()

def simple_aperture_photometry(flux, pix_mask):
    
    sap_flux = np.sum(flux[:, pix_mask], axis=-1)
    sap_flux = (sap_flux / np.median(sap_flux) - 1)
    return sap_flux

def plot_lightcurve(TIC, time, lightcurve, descr):
    
    plt.figure(figsize=(10, 5))
    plt.plot(time, lightcurve*100, "k")
    plt.xlabel("time [days]")
    plt.ylabel("relative flux [percent]")
    plt.title("TIC"+str(TIC)+": "+descr)
    plt.xlim(time.min(), time.max())
    plt.close()

def pixel_level_deconvolution(flux, sap_flux, pix_mask, transit_mask=None):
    # Build the first order PLD basis
    X_pld = np.reshape(flux[:, pix_mask], (len(flux), -1))
    X_pld = X_pld / np.sum(flux[:, pix_mask], axis=-1)[:, None]
    
    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, :X_pld.shape[1]]
    
    # Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((np.ones((len(flux), 1)), X_pld, X2_pld), axis=-1)
    if transit_mask is not None:
        m = ~transit_mask
        XTX = np.dot(X_pld[m].T, X_pld[m])
        w_pld = np.linalg.solve(XTX, np.dot(X_pld[m].T, sap_flux[m]))
    else:
        XTX = np.dot(X_pld.T, X_pld)
        w_pld = np.linalg.solve(XTX, np.dot(X_pld.T, sap_flux))
    pld_flux = np.dot(X_pld, w_pld)
    
    return pld_flux
    
def periodogram(time, lightcurve):
    
    period_grid = np.exp(np.linspace(np.log(1), np.log(15), 50000))
    
    bls = BoxLeastSquares(time, lightcurve)
    bls_power = bls.power(period_grid, 0.1, oversample=20)
    return bls, bls_power, period_grid

def plot_periodogram(ax, bls_power, bls_period, period_grid):
    # Plot the periodogram
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
    return ax

def plot_transit(ax, time, lightcurve, cand_period, cand_t0):
    # Plot the folded transit
    x_fold = (time - cand_t0 + 0.5*cand_period)%cand_period - 0.5*cand_period
    m = np.abs(x_fold) < 0.4
    ax.plot(x_fold[m], lightcurve[m]*100, ".k")
    
    # Overplot the phase binned light curve
    bins = np.linspace(-0.41, 0.41, 32)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=lightcurve*100)
    denom[num == 0] = 1.0
    ax.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1")
    
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylabel("de-trended flux [percent]")
    ax.set_xlabel("time since transit")
    return ax

def plot_folded_pldmodel(TIC, time, pld_flux, cand_period, cand_t0):
    
    plt.figure(figsize=(10, 5))
    
    x_fold = (time - cand_t0 + 0.5*cand_period) % cand_period - 0.5*cand_period
    m = np.abs(x_fold) < 0.3
    plt.plot(x_fold[m], pld_flux[m]*100, ".k", ms=4)
    
    bins = np.linspace(-0.5, 0.5, 60)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=pld_flux*100)
    denom[num == 0] = 1.0
    plt.plot(0.5*(bins[1:] + bins[:-1]), num / denom, color="C1", lw=2)
    plt.xlim(-0.2, 0.2)
    plt.title("TIC"+str(TIC))
    plt.xlabel("time since transit")
    plt.ylabel("PLD model flux [percent]")
    plt.close()

name = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0002/6113/6679/tess2018206045859-s0001-0000000261136679-0120-s_tp.fits"
name2 = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0003/0003/3922/tess2018206045859-s0001-0000000300033922-0120-s_tp.fits"
name3 = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0002/6113/6679/tess*_tp.fits"

tess_lightcurves(name, 261136679, os.getcwd())
