from sft import sft_qrt_gb, isft_qrt_gb
from utils import uniform_disk
import numpy as np

class Coronograph:
    """
    Coronograph class to compute the electric field in the image plane
    of a classical Lyot coronagraph with four planes (A: entrance pupil, 
    B: intermediate focal plane, C: relayed pupil before stop, L: relayed 
    pupil after stop, D: final image plane)
    """
    
    def __init__(self, nPup=100, mB=2.252, mD=45.0, ID=0.14, rho0=0.0, rho1=20.0, tau=0.5, nFPM=50, CtrBtwnPix=False): 
        """
        Initializes the Coronograph class with the given parameters.

        Parameters
        ----------
        nPup : int
            Size of the entrance pupil array.
        
        mB : float
            Maximum spatial frequency in the intermediate focal plane (in units of lambda/D).
        
        mD : float
            Maximum spatial frequency in the final image plane (in units of lambda/D).

        ID : float
            Secondary mirror size in pupil diameter.

        rho0 : float
            Inner edge of the dark zone in units of lambda/D.

        rho1 : float
            Outer edge of the dark zone in units of lambda/D.        
        
        tau : float
            Integrated pupil transmission.
        
        nFPM : int
            Size of the focal plane mask array.
        
        CtrBtwnPix : bool, optional
            If True, the disk is centered between four pixels. Default is False.
        """

        self.nPup = nPup
        self.mB = mB
        self.rMask = mB
        
        self.mD = mD
        self.Fmask2d = mD
        self.nImg2d = 2*int(self.Fmask2d)
        
        self.ID = ID
        self.rho0 = rho0
        self.rho1 = rho1

        self.tau = tau
        self.nFPM = nFPM
        self.CtrBtwnPix = CtrBtwnPix

        ### Entrance pupil

        #self.Pupil2d = uniform_disk(self.nPup, self.nPup/2., CtrBtwnPix=CtrBtwnPix)-uniform_disk(self.nPup, self.ID*self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)
        #self.Pupil2d_qrt = self.Pupil2d[self.nPup//2:,self.nPup//2:]

        ### Focal plane mask
        #self.mask2d = uniform_disk(self.nFPM, self.nFPM/2., CtrBtwnPix=self.CtrBtwnPix)
        #self.mask2d_qrt = self.mask2d[self.nFPM//2:,self.nFPM//2:]  

        ### Entrance pupil
        self.Pupil2d = uniform_disk(self.nPup, self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)-uniform_disk(self.nPup, self.ID*self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)
        self.Pupil2d_qrt = self.Pupil2d[self.nPup//2:,self.nPup//2:]
        self.TR = np.sum(self.Pupil2d_qrt)
        self.idx_out2d_qrt = self.Pupil2d_qrt == 0.0      

        ### Focal plane mask
        self.mask2d = uniform_disk(self.nFPM, self.nFPM/2., CtrBtwnPix=self.CtrBtwnPix)
        self.mask2d_qrt = self.mask2d[self.nFPM//2:,self.nFPM//2:]

        ### Lyot Stop
        self.LyotStop2d = uniform_disk(self.nPup, self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)-uniform_disk(self.nPup, self.ID*self.nPup/2., CtrBtwnPix=self.CtrBtwnPix)
        self.LyotStop2d_qrt = self.LyotStop2d[self.nPup//2:,self.nPup//2:]

        # Select pixels in pupil and image planes for optimization purposes


def compute_corono_field_2d_qrt(Apod2d_qrt, Pupil2d_qrt, LyotStop2d_qrt, mask2d_qrt):
    """
    Computes the coronagraph electric field for a classical Lyot coronagraph
    with four planes (A: entrance pupil, B: intermediate focal plane, 
    C: relayed pupil before stop, L: relayed pupil after stop, 
    D: final image plane).
    Resolution element are given in :math:`\lambda_0/D` where 
    :math:`\lambda_0` and :math:`D` denote the central and the telescope 
    diameter.

    Parameters
    ---------- 
    Apod2d : array_like 
        Entrance pupil apodization :math:`\Phi`
        
    Returns    
    ----------    
    field_Dtmp : array_like
        Coronagraphic electric field :math:`\Psi_D` in the final image plane
        at all the wavelengths
        
    """        
    field_A_qrt    = Pupil2d_qrt.reshape(-1)*Apod2d_qrt.reshape(-1)
                                      
    field_B_qrt    = mask2d_qrt.reshape(-1)*sft_qrt_gb(field_A_qrt, nFPM, mB, 
                                    CtrBtwnPix=True)
    field_C_qrt    = field_A_qrt - isft_qrt_gb(field_B_qrt, nPup, mB, 
                                   CtrBtwnPix=True)
    field_L_qrt    = LyotStop2d_qrt.reshape(-1)*field_C_qrt
    field_Dtmp_qrt = sft_qrt_gb(field_L_qrt, nImg2d, mD, 
              CtrBtwnPix=True)
                     
    return field_Dtmp_qrt