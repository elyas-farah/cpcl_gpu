import pymaster as nmt
import numpy as np
import healpy as hp
import wget
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import simpson
cmap = plt.get_cmap('twilight_shifted')



def gen_maps_spin00(cl_list, nside, lmax):
    """Generate a spin-0 map from a given power spectrum. cl_list could include cross-correlation spectrea as well."""
    alm_1, alm_2 = hp.synalm(cls=cl_list)
    map1 = hp.alm2map(alm_1, nside=nside, lmax=lmax)
    map2 = hp.alm2map(alm_2, nside=nside, lmax=lmax)
    map1 = np.atleast_2d(map1)
    map2 = np.atleast_2d(map2)    
    return map1, map2

def gen_maps_spin22(cl_list, nside, lmax):
    alm_1, alm_2 = hp.synalm(cls=cl_list)
    maps1 = hp.alm2map_spin([alm_1, alm_1*0], nside=nside, lmax=lmax, spin=2)
    maps2 = hp.alm2map_spin([alm_2, alm_2*0], nside=nside, lmax=lmax,spin=2)    
    
    return maps1, maps2
    
def gen_maps_spin20(cl_list, nside, lmax, spin, spin2):
    '''I will enter the correlation factors where the first field is always shear ŷdn the second is the di
    dipersin measure.'''
    if spin == 2 and spin2==0:
        alm_1, alm_2 = hp.synalm(cls=cl_list)
        map1 = hp.alm2map_spin([alm_1, alm_1*0], nside=nside, lmax=lmax, spin=2)    
        map2 = hp.alm2map(alm_2, nside=nside, lmax=lmax)
        map1 = np.atleast_2d(map1)
        map2 = np.atleast_2d(map2)    
    else:
        alm_1, alm_2 = hp.synalm(cls=cl_list)
        map1 = hp.alm2map(alm_1, nside=nside, lmax=lmax)
        map2 = hp.alm2map_spin([alm_2, alm_2*0], nside=nside, lmax=lmax, spin=2)    
        map1 = np.atleast_2d(map1)
        map2 = np.atleast_2d(map2) 
    return map1, map2


# Simulator
def gen_sim(spin=0, spin2 = None, cl_true = None, cl_true_2 = None,cl_true_12 = None, nside = None, lmax_nside = None):
    if cl_true_2 is not None:
        if spin == 0 and spin2 == 0:
            cls_list = [cl_true, cl_true_12, cl_true_2]
            map1, map2 = gen_maps_spin00(cls_list, nside, lmax_nside)
            
            return np.atleast_2d(map1), np.atleast_2d(map2)
        elif spin == 2 and spin2 == 2:
            cls_list = [cl_true, cl_true_12, cl_true_2]
            maps1, maps2 = gen_maps_spin22(cls_list, nside, lmax_nside)
            return np.atleast_2d(maps1), np.atleast_2d(maps2)
        else:
            cls_list = [cl_true, cl_true_12, cl_true_2]
            maps1, map2 = gen_maps_spin20(cls_list, nside, lmax_nside, spin, spin2)
            
            return np.atleast_2d(maps1), np.atleast_2d(map2)
    
    else:
        if spin == 0:
            alms = hp.synalm(cls=cl_true)
            map = hp.alm2map(alms, nside)
            map = np.atleast_2d(map)
            return np.array(map)
        else:
            alms = hp.synalm(cls=cl_true)
            map = hp.alm2map_spin([alms, alms*0], nside, spin, lmax_nside)
            return np.array(map)
            

def find_cell_prediction(cls_th,  w: nmt.NmtWorkspace , pixwin, spin, spin2 = None):
    
    if spin2 is None:
        if spin == 0:
            cl_pred = w.decouple_cell(w.couple_cell(np.atleast_2d(cls_th)*pixwin**2))
            return cl_pred
        if spin == 2:
            cl_list = np.array([cls_th*pixwin**2, np.zeros_like(cls_th), np.zeros_like(cls_th), np.zeros_like(cls_th)])
            cl_pred = w.decouple_cell(w.couple_cell(cl_list))
            return np.atleast_2d(cl_pred[0])
    
    else:
        if spin == 0 and spin2 == 0:
            cl_pred = w.decouple_cell(w.couple_cell(np.atleast_2d(cls_th)*pixwin**2))
            return cl_pred
        elif spin == 2 and spin2 == 2:
            cl_list = np.array([cls_th*pixwin**2, np.zeros_like(cls_th), np.zeros_like(cls_th), np.zeros_like(cls_th)])
            cl_pred = w.decouple_cell(w.couple_cell(cl_list))
            return cl_pred
        else:
            cl_list = np.array([cls_th*pixwin**2, np.zeros_like(cls_th)])
            cl_pred = w.decouple_cell(w.couple_cell(cl_list))
            return cl_pred
        
            
            
        
        
        
        
    




# Code to generate simulated fields
def gen_fixed_field(spin, spin2, return_map=False, ipix_1=None,ipix_2=None, pos_data_1 = None,pos_data_2 = None, w_data=None,
                      w_data_2=None, lmax=None,
                      cl_true = None,cl_true_2 = None,cl_true_12 = None, nside = None, lmax_nside = None , mp_1 = None, mp_2 = None,
                      with_noise_1=None, with_noise_2 = None):
    has_map = False
    if mp_1 is None:
        if cl_true_2 is not None:
            cls_list = [cl_true, cl_true_12, cl_true_2]
            mp_1, mp_2 = gen_sim(spin=spin, spin2=spin2, cl_true=cl_true, cl_true_2=cl_true_2, cl_true_12 = cl_true_12, nside=nside, lmax_nside = lmax_nside)
            
            mp_1 = np.atleast_2d(mp_1)
            mp_2 = np.atleast_2d(mp_2)
            
            if with_noise_1 is not None:
                nval = np.random.randn(*(mp_1.shape)) * with_noise_1
                mp_1 += nval
            if with_noise_2 is not None:
                nval = np.random.randn(*(mp_2.shape)) * with_noise_2
                mp_2 += nval
        
        else:
            mp_1 = np.atleast_2d(gen_sim(spin=spin, cl_true=cl_true, nside=nside,lmax_nside=lmax_nside))
            if with_noise_1 is not None:
                nval = np.random.randn(*(mp_1.shape)) * with_noise_1
                mp_1 += nval
        # Transpose, extract the pixels, and transpose again. 
    else:
        has_map = True
    if cl_true_2 is not None:  
        fval_1 = mp_1[:,ipix_1]
        fval_2 = mp_2[:,ipix_2]
    else:
        fval = mp_1[:, ipix_1]

    if cl_true_2 is not None:
        
        fld_1 = nmt.NmtFieldCatalog(pos_data_1, w_data, fval_1, spin=spin,
                                lmax=lmax, lonlat=True)
        
        
        fld_2 = nmt.NmtFieldCatalog(pos_data_2, w_data_2, fval_2, spin=spin2,
                                lmax=lmax, lonlat=True)
        if return_map:
            return fld_1, fld_2, mp_1, mp_2
        return fld_1, fld_2
    else:
        if spin == 0:
            fval = fval - np.mean(fval)
        else:
            fval = fval - np.mean(fval, axis=1)[:, np.newaxis]
        
        fld = nmt.NmtFieldCatalog(pos_data_1, w_data, fval- np.mean(fval), spin=spin,
                                lmax=lmax, lonlat=True)
        if return_map:
            return fld, mp_1
        return fld


def get_pos(nsources, mode, sel, nside=None, fname_cat = None):
    nsources_save = np.copy(nsources)
    # nsources *= 10
    if mode == 'random' or mode == 'fixed' or mode == "field_variance":        
        nside_sel = hp.npix2nside(len(sel))
        th_ran = np.arccos(-1+2*np.random.rand(nsources))
        phi_ran = 2*np.pi*np.random.rand(nsources)
        u_ran = np.random.rand(nsources)
        ipix_ran = hp.ang2pix(nside_sel, th_ran, phi_ran)
        keep = u_ran <= sel[ipix_ran] #
        nsrc = nsources_save#np.sum(keep)
        # Assign RA, Dec, and weights
        pos_data = np.array([np.degrees((phi_ran[keep])[:nsources_save]), 90-np.degrees((th_ran[keep])[:nsources_save])])
        
    elif mode == 'catalog':
            
        cat = fits.open(fname_cat)[1].data
        # Select only sources with redshifts z < 1.47 (first bin of Alonso et al. 2023)
        cat = cat[cat['redshift_quaia'] < 1.47]

        cat_sel = np.unique(np.random.randint(low=0, high=cat.shape[0], size=nsources))
        cat = cat[cat_sel]
        nsrc = len(cat)
        pos_data = np.array([cat['ra'], cat['dec']])
        
    elif mode == 'hp_grid':
        pos_data = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        sel_udg = hp.ud_grade(sel, nside_out=nside).astype('bool')
        pos_data = np.array([pos_data[0][sel_udg], pos_data[1][sel_udg]])
        nsrc = pos_data[0].shape[0]
    
    return pos_data, nsrc

def get_snr(mean,cov):
    return np.sqrt(np.sum(np.dot(mean,np.dot(np.linalg.inv(cov),mean))))

def get_tangential_shear(map1, map2):
    shear_field = np.complex128(map1, map2)
    
    shear_tangential = -np.real(shear_field)
    shear_cross = -np.imag(shear_field)
    return shear_tangential, shear_cross    
def factor(spin):
    if spin == 0:
        return 1
    elif spin == 2:
        return 2

def generate_mocks(cat_sizes,
                   mode,
                   nsims,
                   nside,
                   ell_extern,
                   cl_extern,
                   sel,
                   spin,
                   spin2,
                   edges,
                   i_tomo=0,
                   j_tomo=0,
                   lmin = 40,
                   nell = 10,
                   noise_level_i_tomo = None,
                   noise_level_j_tomo = None):
    cross_correlation = False
    if i_tomo != j_tomo:
        cross_correlation = True
    if cross_correlation:
        cl_mean_rnd = []
        cl_prediction =[]
        covmat = []
        field_variance_cov = []
        analytic_cov = []
        npix = hp.nside2npix(nside)
        lmax = 3*nside-1
        lmax_nside = 3*nside-1
        ls = ell_extern[:lmax_nside+1]# np.arange(lmax_nside+1)
        pixwin = hp.pixwin(nside)
        cl_true_1 = cl_extern[:lmax_nside+1,i_tomo, i_tomo]
        cl_true_2 = cl_extern[:lmax_nside+1,j_tomo, j_tomo]
        cl_true_12 = cl_extern[:lmax_nside+1,i_tomo, j_tomo]
        field_variance1 = simpson(cl_true_1*ls,x = ls)/2./np.pi
        field_variance2 = simpson(cl_true_2*ls,x = ls)/2./np.pi
        field_variance_observed = 0    
        index = 0
        counting = 0
        for nsources in cat_sizes:
            if isinstance(nsources, list):
                nsources_1 = nsources[0]
                nsources_2 = nsources[1]
            else:
                nsources_1 = nsources
                nsources_2 = nsources
            if mode=="fixed":
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
                pos_data_1, nsrc_1 = get_pos(nsources_1, mode, sel)
                pos_data_2, nsrc_2 = get_pos(nsources_2, mode, sel)
                mp_1, mp_2 = gen_sim(spin=spin, spin2=spin2, cl_true=cl_true_1,cl_true_2=cl_true_2, cl_true_12 = cl_true_12, nside=nside, lmax_nside = lmax_nside)
                
                
                w_data_1 = np.ones(nsrc_1)
                w_data_2 = np.ones(nsrc_2)

                ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                fval_1 = mp_1[:, ipix_1]
                
                fval_2 = mp_2[:, ipix_2]
                fld_1 = nmt.NmtFieldCatalog(pos_data_1, w_data_1, fval_1, lmax=b.lmax, lonlat=True)
                fld_2 = nmt.NmtFieldCatalog(pos_data_2, w_data_2, fval_2, lmax=b.lmax, lonlat=True)
                
                # edges = np.geomspace(lmin,3*nside,nell).astype(int)                
                

                leff = b.get_effective_ells()
                cls = []
                for i in range(nsims):
                    fld_1, fld_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1, ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2, cl_true_12 = cl_true_12,nside=nside, lmax_nside = lmax_nside,
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                    w = nmt.NmtWorkspace.from_fields(fld_1, fld_2, b)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld_1, fld_2)))
                cls = np.array(cls)
            if mode=="random":
                pos_data_1, nsrc_1 = get_pos(nsources_1, mode, sel)
                pos_data_2, nsrc_2 = get_pos(nsources_2, mode, sel)
                w_data_1 = np.ones(nsrc_1)
                w_data_2 = np.ones(nsrc_2)

                mp_1, mp_2 = gen_sim(spin=spin, spin2=spin2, cl_true=cl_true_1,cl_true_2=cl_true_2,nside=nside, lmax_nside = lmax_nside)
                ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                fval_1 = mp_1[:, ipix_1]
                fval_2 = mp_2[:, ipix_2]
                fld_1 = nmt.NmtFieldCatalog(pos_data_1, w_data_1, fval_1, lmax=lmax_nside, lonlat=True)
                fld_2 = nmt.NmtFieldCatalog(pos_data_2, w_data_2, fval_2, lmax=lmax_nside, lonlat=True)
                
                # edges = np.geomspace(40,3*nside,10).astype(int)
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

                leff = b.get_effective_ells()
                
                cls = []
                for i in range(nsims):
                    pos_data_1, nsrc = get_pos(nsources_1, mode, sel)
                    pos_data_2, nsrc = get_pos(nsources_2, mode, sel)
                    ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                    ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                    fld_1, fld_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1,ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2, cl_true_12 = cl_true_12,nside=nside, lmax_nside = lmax_nside,
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                    w = nmt.NmtWorkspace.from_fields(fld_1, fld_2, b)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld_1, fld_2)))
                cls = np.array(cls)

            if mode=="field_variance":
                pos_data_1, nsrc_1 = get_pos(nsources_1, mode, sel)
                pos_data_2, nsrc_2 = get_pos(nsources_2, mode, sel)
                w_data_1 = np.ones(nsrc_1)
                w_data_2 = np.ones(nsrc_2)

                
                ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                # edges = np.geomspace(lmin,3*nside,nell).astype(int)
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
                leff = b.get_effective_ells()
                # print(spin, spin2, i_tomo, j_tomo)
                fld_1, fld_2, map_fixed_1, map_fixed_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1,ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2,nside=nside,  cl_true_12 = cl_true_12,lmax_nside = lmax_nside, return_map=True,
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                cls = []
                w = nmt.NmtWorkspace.from_fields(fld_1, fld_2, b)                
                for i in range(nsims):

                    pos_data_1, nsrc = get_pos(nsources_1, mode, sel)
                    pos_data_2, nsrc = get_pos(nsources_2, mode, sel)
                    #ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                    #ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                    fld_1, fld_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1,ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2, cl_true_12 = cl_true_12,nside=nside, lmax_nside = lmax_nside, mp_1=map_fixed_1, mp_2=map_fixed_2, 
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld_1, fld_2)))
                cls = np.array(cls)

            if mode=="chime":
                pos_data_1, nsrc_1 = get_pos(nsources_1, "fixed", sel)
                pos_data_2, nsrc_2 = get_pos(nsources_2, "fixed", sel)
                w_data_1 = np.ones(nsrc_1)
                w_data_2 = np.ones(nsrc_2)

                
                ipix_1 = hp.ang2pix(nside, pos_data_1[0], pos_data_1[1], lonlat=True)
                ipix_2 = hp.ang2pix(nside, pos_data_2[0], pos_data_2[1], lonlat=True)
                # edges = np.geomspace(lmin,3*nside,nell).astype(int)
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

                leff = b.get_effective_ells()
                
                #leff = b.get_effective_ells()
                fld_1, fld_2, map_fixed_1, map_fixed_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1,ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2,nside=nside,  cl_true_12 = cl_true_12,lmax_nside = lmax_nside, return_map=True,
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                cls = []
                w = nmt.NmtWorkspace.from_fields(fld_1, fld_2, b)                
                for i in range(nsims):
                    np.random.shuffle(pos_data_1[0,:])
                    np.random.shuffle(pos_data_1[1,:])
                    fld_1, fld_2 = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix_1,ipix_2=ipix_2, pos_data_1=pos_data_1,pos_data_2=pos_data_2,w_data=w_data_1,w_data_2=w_data_2,lmax=lmax,
                                            cl_true=cl_true_1,cl_true_2=cl_true_2, cl_true_12 = cl_true_12,nside=nside, lmax_nside = lmax_nside, mp_1=map_fixed_1, mp_2=map_fixed_2, 
                                            with_noise_1 = noise_level_i_tomo,
                                            with_noise_2 = noise_level_j_tomo)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld_1, fld_2)))
                cls = np.array(cls)

            
            
            
            # Fix the inputs power spectra such tha it correspond to spin-2 or spin-0 fields, taking into consideration the mode coupling matrix.
            cl_pred = find_cell_prediction(cl_true_12, w, pixwin, spin = spin, spin2 = spin2)
            cl_pred11 = find_cell_prediction(cl_true_1, w, pixwin, spin = spin, spin2 = spin2)
            cl_pred22 = find_cell_prediction(cl_true_2, w, pixwin, spin = spin, spin2 = spin2)
            
            
            cl_prediction.append(cl_pred[0,:])
            cl_mean = np.mean(cls, axis=0)
            cl_mean_rnd.append(cl_mean[0])
            delta_ell = np.zeros_like(leff)
            delta_ell[:-1] = leff[1:] - leff[:-1]
            delta_ell[-1] = delta_ell[-2]
            covmat.append(np.cov((cls[:,index,:]).T))
            ndens_sources1 = nsources_1/4.0/np.pi
            ndens_sources2 = nsources_2/4.0/np.pi
            analytic_cov.append((cl_pred[0,:]**2 + (cl_pred11[0, :] + field_variance1/(factor(spin)*ndens_sources1))*(cl_pred22[0, :] + field_variance2/(factor(spin2)*ndens_sources2)))/(2*leff + 1)/delta_ell)
            field_variance_cov.append(((field_variance1/(factor(spin)*ndens_sources1)) * (factor(spin2)*ndens_sources2))/(2*leff + 1)/delta_ell)
        return leff, cl_mean_rnd, cl_prediction, covmat, analytic_cov, field_variance_cov, cls
    else:
        cl_mean_rnd = []
        cl_prediction =[]
        covmat = []
        field_variance_cov = []
        analytic_cov = []
        Nf_list = []
        
        npix = hp.nside2npix(nside)
        lmax = 3*nside-1
        lmax_nside = 3*nside-1
        ls = ell_extern[:lmax_nside+1]# np.arange(lmax_nside+1)
        pixwin = hp.pixwin(nside)
        cl_true = cl_extern[:lmax_nside+1,i_tomo, j_tomo] #1/(ls+10)**exponent
        field_variance = simpson(cl_true*ls,x = ls)/2./np.pi

        index = 0
        for nsources in cat_sizes:
            if mode=="fixed":
                pos_data, nsrc = get_pos(nsources, mode, sel)
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
                    
                # print(f"The catalog has {nsrc} sources.")
                mp = gen_sim(spin = spin, cl_true=cl_true,nside=nside, lmax_nside = lmax_nside)

                # Catalog weights
                w_data = np.ones(nsrc)

                # Generate continuous map
                # Calculate pixel indices for each souce
                ipix = hp.ang2pix(nside, pos_data[0], pos_data[1], lonlat=True)
                # Assign field values from sky positions
                # print(type(ipix), ipix, mp.shape)
                fval = mp[:, ipix]

                fld = nmt.NmtFieldCatalog(pos_data, w_data, fval, lmax=b.lmax, lonlat=True)
                
                # edges = np.unique(np.geomspace(lmin,3*nside,nell).astype(int))
                
                cls = []
                w = nmt.NmtWorkspace.from_fields(fld, fld, b)
                for i in range(nsims):
                    fld = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix, pos_data_1=pos_data,w_data=w_data,lmax=b.lmax,
                                            cl_true=cl_true,nside=nside, lmax_nside = b.lmax)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld, fld)))
                    Nf_list.append(fld.Nf)
                    
                cls = np.array(cls)
            if mode=="random":
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
                
                pos_data, nsrc = get_pos(nsources, mode, sel)
                cls = []
                # Catalog weights
                w_data = np.ones(nsrc)

                # Generate continuous map
                mp = gen_sim(cl_true=cl_true,nside=nside, lmax_nside = lmax_nside)
                # Calculate pixel indices for each souce
                ipix = hp.ang2pix(nside, pos_data[0], pos_data[1], lonlat=True)
                # Assign field values from sky positions
                fval = mp[:, ipix]
                fld = nmt.NmtFieldCatalog(pos_data, w_data, fval, lmax=b.lmax, lonlat=True)
                
                # edges = np.unique(np.geomspace(lmin,3*nside,nell).astype(int))
                cls = []
                for i in range(nsims):
                    pos_data, nsrc = get_pos(nsources, mode, sel)
                    ipix = hp.ang2pix(nside, pos_data[0], pos_data[1], lonlat=True)    
                    
                    fld = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix, pos_data_1=pos_data,w_data=w_data,lmax=b.lmax,
                                            cl_true=cl_true,nside=nside, lmax_nside = b.lmax)
                    w = nmt.NmtWorkspace.from_fields(fld, fld, b)
                    Nf_list.append(fld.Nf)
                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld, fld)))
                cls = np.array(cls)

            if mode=="field_variance":
                pos_data, nsrc = get_pos(nsources, mode, sel=sel)
                cls = []
                # Catalog weights
                w_data = np.ones(nsrc)
                b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

                # Generate continuous map
                mp = gen_sim(cl_true=cl_true,nside=nside, lmax_nside = lmax_nside)
                # Calculate pixel indices for each souce
                ipix = hp.ang2pix(nside, pos_data[0], pos_data[1], lonlat=True)
                # Assign field values from sky positions
                fval = mp[:, ipix]
                fld = nmt.NmtFieldCatalog(pos_data, w_data, fval, lmax=lmax_nside, lonlat=True)
                
                # edges = np.unique(np.geomspace(lmin,3*nside,nell).astype(int))
                leff = b.get_effective_ells()
                fld, map_fixed = gen_fixed_field(spin = spin, spin2=spin2, ipix_1=ipix, pos_data_1=pos_data,w_data=w_data,lmax=b.lmax,
                                            cl_true=cl_true,nside=nside, lmax_nside = b.lmax, return_map=True)
                w = nmt.NmtWorkspace.from_fields(fld, fld, b)
                for i in range(nsims):
                    pos_data, nsrc = get_pos(nsources, mode, sel)
                    # IF YOU UNCOMMENT THIS; YOU WILL NOT GET THE FIELD VARIANCE
                    #ipix = hp.ang2pix(nside, pos_data[0], pos_data[1], lonlat=True)
    

                    fld = gen_fixed_field(spin=spin, spin2=spin2, ipix_1=ipix, pos_data_1=pos_data,w_data=w_data,lmax=b.lmax,
                                            cl_true=cl_true,nside=nside, lmax_nside = b.lmax, mp_1=map_fixed)
                    Nf_list.append(fld.Nf)

                    cls.append(w.decouple_cell(nmt.compute_coupled_cell(fld, fld)))
                cls = np.array(cls)

            leff = b.get_effective_ells()
            cl_pred = find_cell_prediction(cl_true, w, pixwin, spin)
            cl_prediction.append(cl_pred[0,:])
            cl_mean = np.mean(cls, axis=0)
            cl_mean_rnd.append(cl_mean[0])
            delta_ell = np.zeros_like(leff)
            delta_ell[:-1] = leff[1:] - leff[:-1]
            delta_ell[-1] = delta_ell[-2]
            covmat.append(np.cov((cls[:,index,:]).T))
            ndens_sources = nsources/4.0/np.pi
            Nf_avg = np.ones_like(leff)*np.mean(Nf_list)
            Nf_var = np.ones_like(leff)*np.var(Nf_list)
            analytic_cov.append(2*(cl_pred[0,:]+ field_variance/(factor(spin)*ndens_sources))**2/(2*leff + 1)/delta_ell)
            field_variance_cov.append(2*(field_variance/(factor(spin)*ndens_sources))**2/(2*leff + 1)/delta_ell)
        return leff, cl_mean_rnd, cl_prediction, covmat, analytic_cov, field_variance_cov, Nf_avg, Nf_var, cls