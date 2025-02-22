import bilby
import numpy as np
from bilby.core.prior import Cosine, Constraint, PowerLaw
import bilby.core.prior.analytical
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio, Uniform
from gwpy.timeseries import TimeSeries
from matplotlib import pyplot as plt

tov= 1259157749.53
start=tov -118
end= tov+12
duration= 120

Hdata= TimeSeries.fetch_open_data('H1', start, end, sample_rate=16384,cache= True)
Ldata= TimeSeries.fetch_open_data('L1', start, end, sample_rate=16384,cache= True)
Vdata= TimeSeries.fetch_open_data('V1', start, end, sample_rate=16384,cache= True)

Hdata=Hdata.bandpass(50, 2048)
Ldata=Ldata.bandpass(50, 2048)
Vdata=Vdata.bandpass(50, 2048)

H1= bilby.gw.detector.get_empty_interferometer("H1")
L1= bilby.gw.detector.get_empty_interferometer("L1")
V1= bilby.gw.detector.get_empty_interferometer("V1")

H1.set_strain_data_from_gwpy_timeseries(Hdata)
L1.set_strain_data_from_gwpy_timeseries(Ldata)
V1.set_strain_data_from_gwpy_timeseries(Vdata)

psduration= duration*24
psdstart= tov+ psduration

H1_psd_data = TimeSeries.fetch_open_data("H1", psdstart, psdstart + psduration, sample_rate=16384, cache=True, verbose=True)
L1_psd_data = TimeSeries.fetch_open_data("L1", psdstart, psdstart + psduration, sample_rate=16384, cache=True, verbose=True)
V1_psd_data= TimeSeries.fetch_open_data('V1', psdstart, psdstart + psduration, sample_rate=16384, cache=True, verbose=True)

Hpsd= H1_psd_data.bandpass(50, 2048)
Lpsd= L1_psd_data.bandpass(50, 2048)
Vpsd= V1_psd_data.bandpass(50, 2048)

H1psd= H1_psd_data.psd(fftlength=24, overlap=0.5, window=('tukey'), method='median')
L1psd= L1_psd_data.psd(fftlength=24, overlap=0.5, window=('tukey'), method='median')
V1psd= V1_psd_data.psd(fftlength=32, overlap=0.5, window=('tukey'), method='median')

H1.power_spectral_density= bilby.gw.detector.PowerSpectralDensity(frequency_array= H1psd.frequencies.value, psd_array=H1psd.value)
L1.power_spectral_density= bilby.gw.detector.PowerSpectralDensity(frequency_array= L1psd.frequencies.value, psd_array=L1psd.value)
V1.power_spectral_density= bilby.gw.detector.PowerSpectralDensity(frequency_array= V1psd.frequencies.value, psd_array=V1psd.value)

H1.minimum_frequency= 85
L1.minimum_frequency= 85
V1.minimum_frequency= 85


priors= bilby.core.prior.PriorDict()
priors['chirp_mass']= UniformInComponentsChirpMass(name='chirp_mass', minimum=0.24, maximum=0.26, latex_label='$\mathcal{M}_c$')
priors['mass_ratio']= UniformInComponentsMassRatio(name='mass_ratio', minimum=0.02, maximum=1, latex_label='$q$')
priors['mass_1']= Constraint(name= 'mass_1', minimum= 0.03, maximum= 4, latex_label='$m_1$')
priors['mass_2']= Constraint(name= 'mass_2', minimum= 0.05, maximum= 0.2, latex_label='$m_2$')
priors['dec']= Cosine(name='dec', minimum= -np.pi/2, maximum= np.pi/2, latex_label='$\delta$')
#declination- angular distance, north or south of celestial equator
priors['ra']= Uniform(name='ra', minimum=0, maximum= 2*np.pi, boundary= 'periodic', latex_label=r'$\alpha$')
priors['cos_theta_jn']= Uniform(name= 'cos_theta_jn', minimum=-1, maximum=1, latex_label=r'$\theta_{JN}$')
priors['psi']= Uniform(name='psi', minimum=0, maximum=np.pi, boundary= 'periodic', latex_label='$\psi$')
priors['phase']= Uniform(name='phase', minimum=0, maximum= 2*np.pi, boundary='periodic', latex_label='$\phi$')
priors['a_1']= Uniform(name= 'a_1', minimum= 0, maximum= 1, latex_label='$\chi_1$')
priors['a_2']= Uniform(name= 'a_2', minimum= 0, maximum= 1, latex_label='$\chi_2$')
priors['tilt_1']= bilby.core.prior.analytical.Sine(name= 'tilt_1', minimum=0, maximum=np.pi, latex_label=r'$\theta_1$')
priors['tilt_2']= bilby.core.prior.analytical.Sine(name= 'tilt_2', minimum=0, maximum=np.pi, latex_label=r'$\theta_2$')
priors['phi_12']=Uniform(name='phi_12', minimum=np.pi, maximum=2*np.pi, boundary='periodic', latex_label='$\phi_{12}$')
priors['phi_jl']= Uniform(name='phi_jl', minimum=0, maximum=2*np.pi, boundary='periodic', latex_label='$\phi_{jl}$')
priors['luminosity_distance']= PowerLaw(name='luminosity_distance', minimum=5, maximum=300, alpha=2, latex_label='$d_L$')
# create likelihood
ifos= [H1, L1]
apx= 'IMRPhenomPv2'
all= ['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2', 'dec', 'ra', 'cos_theta_jn', 'psi', 'phase', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'H1_matched_filter_snr', 'L1_matched_filter_snr']

waveargs= dict(waveform_approximant= apx, reference_frequency=100, catch_waveform_errors=True)
wavegen= bilby.gw.WaveformGenerator(frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
                                    waveform_arguments= waveargs,
                                    parameter_conversion= convert_to_lal_binary_black_hole_parameters)
likelihood= bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform_generator= wavegen, priors= priors, time_marginalization=True, phase_marginalization=True, distance_marginalization=True)
#run sampler
if __name__ == '__main__':
    result= bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', outdir='MM_PE',
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters, label= 'no_virgo_mpriors_spriors01',npool=6, clean=False, resume= True)

    result.plot_corner(outdir='corners', filename='dL_thetajn3', parameters=['luminosity_distance', 'cos_theta_jn'],
                       labels=['$d_L$', r'$\theta_{JN}$'], save=True, priors= True)
    result.plot_corner(outdir='corners', filename='phi_12_jl3', parameters=['phi_12', 'phi_jl'],
                       labels=['$\phi_{12}$', '$\phi_{jl}$'], save=True, priors= True)
    result.plot_corner(outdir='corners', filename='ra_dec3', parameters=['ra', 'dec'], labels=[r'$\alpha$', '$\delta$'],
                       save=True, priors= True)
    result.plot_corner(outdir='corners', filename='spins_1_23', parameters=['a_1', 'a_2'], labels=['$\chi_1$', '$\chi_2$'],
                       save=True, priors= True)
    result.plot_corner(outdir='corners', filename= 'spins_masses3', parameters= ['mass_1', 'mass_2', 'a_1', 'a_2'],
                       labels=['$m_1$', '$m_2$', '$\chi_1$', '$\chi_2$'], save= True, priors= True)
    result.plot_corner(outdir='corners', filename='tilts_1_23', parameters=['tilt_1', 'tilt_2'],
                       labels=[r'$\theta_1$', r'$\theta_2$'], save=True, priors= True)
    result.plot_corner(outdir='corners', filename='psi_phase3', parameters=['psi', 'phase'], labels=['$\psi$', '$\phi$'],
                       save=True, priors= True)
    result.plot_corner(outdir='corners', filename='masses3', parameters=['mass_1', 'mass_2', 'chirp_mass', 'mass_ratio'],
                       labels=[r'$m_1$', r'$m_2$', '$\mathcal{M}_c$', '$q$'], save=True, priors= True)
    result.plot_corner(outdir='corners', filename= 'all_spins3', parameters=['a_1', 'a_2', 'phi_jl'], save=True)
    result.plot_corner(outdir= 'corners', parameters = ["H1_matched_filter_snr","L1_matched_filter_snr"],  labels=[
                              r"$SNR_{H1}$", r"$SNR_{L1}$"], filename= 'matched_filter')
    result.plot_corner(outdir='corners', priors=True, parameters=all, filename='all_parameters')
  
