import sys
import warnings
from collections import OrderedDict

import numpy as np
from numpy.linalg import norm
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import griddata

from matminer.featurizers.base import BaseFeaturizer

__author__ = "Anubhav Jain <ajain@lbl.gov>, Jade Chongsathapornpong"
        
def finite_second_derivative(E, dk: float) -> float:
    """5-point stencil for 2nd derivative, assumes equally-spaced points
        Args:
            E: energy eigenvalues, length 5.
            dk: k-point spacing between each of these eigenvalues"""
    assert len(E) == 5
    num = -E[4] + 16*E[3] - 30*E[2] + 16*E[1] - E[0]
    den = 12 * dk * dk
    return num / den
    

class BandFeaturizer(BaseFeaturizer):
    """
    Featurizes a pymatgen band structure object.

    Args:
        kpoints ([1x3 numpy array]): list of fractional coordinates of
                k-points at which energy is extracted.
        find_method (str): the method for finding or interpolating for energy
            at given kpoints. It does nothing if kpoints is None.
            options are:
                'nearest': the energy of the nearest available k-point to
                    the input k-point is returned.
                'linear': the result of linear interpolation is returned
                see the documentation for scipy.interpolate.griddata
        nbands (int): the number of valence/conduction bands to be featurized
    """

    def __init__(self, find_method='nearest', curvmode='stat', nbands=2):
        assert curvmode == 'stat' or curvmode == 'vals'
        self.find_method = find_method
        self.mode = curvmode # 'stat' - return curvature mean/min/max / 'vals' - return curvature values 
        self.nbands = nbands

        # Due to a bug in the multiprocessing library, featurizing large numbers
        # of band structures with multiprocessing on Python < 3.8 can result in
        # an error. See https://github.com/hackingmaterials/matminer/issues/417
        if sys.version_info.major == 3 and sys.version_info.minor < 8:
            warnings.warn(
                "Multiprocessing for band structure featurizers "
                "is not recommended for Python versions < 3.8."
                "Setting n_jobs to 1."
            )
            self.set_n_jobs(1)

    def featurize(self, bs):
        """
        Args:
            bs (pymatgen BandStructure or BandStructureSymmLine or their dict):
                The band structure to featurize. To obtain all features, bs
                should include the pymatgen structure attribute.

        Returns:
            OrderedDict of band structure features. If not bs.structure,
                features that require the structure will be returned as NaN.
            Currently supported features:
                band_gap (eV): the difference between the CBM and VBM energy
                is_gap_direct (0.0|1.0): whether the band gap is direct or not
                direct_gap (eV): the minimum direct distance of the last
                    valence band and the first conduction band
                p_ex1_norm (float): k-space distance between Gamma point
                    and k-point of VBM
                n_ex1_norm (float): k-space distance between Gamma point
                    and k-point of CBM
                p_ex1_degen: degeneracy of VBM
                n_ex1_degen: degeneracy of CBM
                        
            Features added for Jade-Shomik-Zhen:
                {band index}_vbm_curv (float): approximation to 2nd derivative along
                    symmetry line, of energy w.r.t. k at VBM
                {band index}_cbm_curv (float): ^ for CBM
        """
        if isinstance(bs, dict):
            bs = BandStructure.from_dict(bs)
        if bs.is_metal():
            raise ValueError("Cannot featurize a metallic band structure!")
        bs_kpts = [k.frac_coords for k in bs.kpoints]
        cvd = {"p": bs.get_vbm(), "n": bs.get_cbm()}
        for itp, tp in enumerate(["p", "n"]): # whyyyy
            cvd[tp]["k"] = bs.kpoints[cvd[tp]["kpoint_index"][0]].frac_coords
            cvd[tp]["bidx"], cvd[tp]["sidx"] = self.get_bindex_bspin(cvd[tp], is_cbm=bool(itp))
            cvd[tp]["Es"] = np.array(bs.bands[cvd[tp]["sidx"]][cvd[tp]["bidx"]])
        band_gap = bs.get_band_gap()
        

        # featurize
        feat = OrderedDict()
        feat["band_gap"] = band_gap["energy"]
        feat["is_gap_direct"] = band_gap["direct"]
        feat["direct_gap"] = min(cvd["n"]["Es"] - cvd["p"]["Es"])
        for tp in ["p", "n"]:
            feat[f"{tp}_ex1_norm"] = norm(cvd[tp]["k"])
            if bs.structure:
                feat[f"{tp}_ex1_degen"] = bs.get_kpoint_degeneracy(cvd[tp]["k"])
            else:
                feat[f"{tp}_ex1_degen"] = float("NaN")
                
        # custom feature:
        # band curvatures (spin-up band), assumes 1D band structure along symmline
        cbm_curvs = []
        vbm_curvs = []
        
        cbm_info = bs.get_cbm()
        band_indices = cbm_info['band_index'][Spin.up]
        kpoint_index = cbm_info['kpoint_index'][0]
        for bidx in band_indices:
            # get energy eigenvals at 5 k points around the index of cbm
            pts = bs.bands[Spin.up][bidx][kpoint_index-2:kpoint_index+3]
            # get spacing between k points
            dk = np.linalg.norm(bs.kpoints[kpoint_index].cart_coords - bs.kpoints[kpoint_index - 1].cart_coords)
            cbm_curvature = finite_second_derivative(pts, dk)
            cbm_curvs.append(cbm_curvature)
            if self.mode == 'vals':
                feat[f"{bidx}_cbm_curv"] = cbm_curvature
            
        
        vbm_info = bs.get_vbm()
        band_indices = vbm_info['band_index'][Spin.up]
        kpoint_index = vbm_info['kpoint_index'][0]
        for bidx in band_indices:
            # get energy eigenvals at 5 k points around the index of cbm
            pts = bs.bands[Spin.up][bidx][kpoint_index-2:kpoint_index+3]
            # get spacing between k points
            dk = np.linalg.norm(bs.kpoints[kpoint_index].cart_coords - bs.kpoints[kpoint_index - 1].cart_coords)
            vbm_curvature = finite_second_derivative(pts, dk)
            vbm_curvs.append(vbm_curvature)
            if self.mode == 'vals':
                feat[f"{bidx}_vbm_curv"] = vbm_curvature
        
        if self.mode == 'stat':
            feat["N_cbm"] = len(cbm_curvs)
            feat["mean_cbm_curv"] = np.mean(cbm_curvs)
            feat["min_cbm_curv"] = np.min(cbm_curvs)
            feat["max_cbm_curv"= = np.max(cbm_curvs)
            feat["N_vbm"] = len(vbm_curvs)
            feat["mean_vbm_curv"] = np.mean(vbm_curvs)
            feat["min_vbm_curv"] = np.min(vbm_curvs)
            feat["max_vbm_curv"] = np.max(vbm_curvs)

        return feat    
    
    def feature_labels(self):
        print("Feature labels for Jade's custom band featurizer are inaccurate, but this method is included to accommodate BaseFeaturizer.")
        labels = [
            "band_gap",
            "is_gap_direct",
            "direct_gap",
            "p_ex1_norm",
            "p_ex1_degen",
            "n_ex1_norm",
            "n_ex1_degen",
        ]
        
        return labels

    @staticmethod
    def get_bindex_bspin(extremum, is_cbm):
        """
        Returns the band index and spin of band extremum

        Args:
            extremum (dict): dictionary containing the CBM/VBM, i.e. output of
                Bandstructure.get_cbm()
            is_cbm (bool): whether the extremum is the CBM or not
        """

        idx = int(is_cbm) - 1  # 0 for CBM and -1 for VBM
        try:
            bidx = extremum["band_index"][Spin.up][idx]
            bspin = Spin.up
        except IndexError:
            bidx = extremum["band_index"][Spin.down][idx]
            bspin = Spin.down
        return bidx, bspin

    def citations(self):
        return ["@article{in_progress, title={{In progress}} year={2017}}"]

    def implementors(self):
        return ["Alireza Faghaninia", "Anubhav Jain"]
