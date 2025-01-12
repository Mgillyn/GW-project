import bilby
import numpy as np
from bilby.core.prior import Cosine, Constraint, PowerLaw
import bilby.core.prior.analytical
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio, Uniform
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

#gps= event_gps('GW') <--insert event after GW, might need to try removing 'GW' if it fails
tov= 1267725971.02
start=tov -118
end= tov+2
duration= 120
#so it appears the event hasn't been released on gwosc
#does the paper analyze 256s in the PE? try that...
Hdata= TimeSeries.fetch_open_data('H1', start, end, sample_rate=4096,cache= True)
Ldata= TimeSeries.fetch_open_data('L1', start, end, sample_rate=4096,cache= True)
Vdata= TimeSeries.fetch_open_data('V1', start, end, sample_rate=4096,cache= True)
H1= bilby.gw.detector.get_empty_interferometer("H1")
L1= bilby.gw.detector.get_empty_interferometer("L1")
V1= bilby.gw.detector.get_empty_interferometer("V1")
H1.set_strain_data_from_gwpy_timeseries(Hdata)
L1.set_strain_data_from_gwpy_timeseries(Ldata)
V1.set_strain_data_from_gwpy_timeseries(Vdata)
psduration= duration*32 #use 120 without ROQ- still not sure what we will use
#do they use 256 seconds of strain data for the analysis?
psdstart= start- psduration

H1_psd_data = TimeSeries.fetch_open_data("H1", psdstart, psdstart + psduration, sample_rate=4096, cache=True)
L1_psd_data = TimeSeries.fetch_open_data("L1", psdstart, psdstart + psduration, sample_rate=4096, cache=True)
V1_psd_data= TimeSeries.fetch_open_data("V1", psdstart, psdstart + psduration, sample_rate=4096, cache=True)
psd_alpha = 2 * H1.strain_data.roll_off / duration

H1psd = H1_psd_data.psd(fftlength=4, overlap=0, window=("tukey", psd_alpha), method="median")
L1psd = L1_psd_data.psd(fftlength=4, overlap=0, window=("tukey", psd_alpha), method="median")
V1psd= V1_psd_data.psd(fftlength=4, overlap=0, window=('tukey', psd_alpha), method= 'median')
H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=H1psd.frequencies.value, psd_array=H1psd.value)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=L1psd.frequencies.value, psd_array=L1psd.value)
V1.power_spectral_density= bilby.gw.detector.PowerSpectralDensity(
    frequency_array=V1psd.frequencies.value, psd_array=V1psd.value)

priors= bilby.core.prior.PriorDict()
priors['chirp_mass']= UniformInComponentsChirpMass(name='chirp_mass', minimum= 0.351, maximum= 0.355, latex_label='$\mathcal{M}_c$')
priors['mass_ratio']= UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1.0, latex_label='$q$')
priors['mass_1']= Constraint(name= 'mass_1', minimum= 0.142, maximum= 10, latex_label='$m_1$')
priors['mass_2']= Constraint(name= 'mass_2', minimum= 0.142, maximum= 10, latex_label='$m_2$')
priors['dec']= Cosine(name='dec', minimum= -np.pi/2, maximum= np.pi/2, latex_label='$\delta$')
#declination- angular distance, north or south of celestial equator
priors['ra']= Uniform(name='ra', minimum=0, maximum= 2*np.pi, boundary= 'periodic', latex_label=r'$\alpha$')

priors['theta_jn']= Uniform(name= 'theta_jn', minimum=-1, maximum=1, latex_label=r'$\theta_JN$')

priors['psi']= Uniform(name='psi', minimum=0, maximum=np.pi, boundary= 'periodic', latex_label='$\psi$')
priors['phase']= Uniform(name='phase', minimum=0, maximum= 2*np.pi, boundary='periodic', latex_label='$\phi$')
priors['a_1']= Uniform(name= 'a_1', minimum=0, maximum=0.8, latex_label=r'$a_1$')
priors['a_2']= Uniform(name= 'a_2', minimum=0, maximum=0.8, latex_label=r'$a_2$')
priors['tilt_1']= bilby.core.prior.analytical.Sine(name= 'tilt_1', minimum=0, maximum=np.pi, latex_label=r'$\theta_1$')
priors['tilt_2']= bilby.core.prior.analytical.Sine(name= 'tilt_2', minimum=0, maximum=np.pi, latex_label=r'$\theta_2$')
priors['Difference-azimuthal-spin-angle']=Uniform(name='difference between azimuthal spin angles', minimum=0, maximum=2*np.pi, boundary='periodic', latex_label='$\phi_12$')
priors['phase-orbital-angular-momenta']= Uniform(name='phase between orbital and angular momenta', minimum=0, maximum=2*np.pi, boundary='periodic', latex_label='$\phi_jl$')
priors['luminosity_distance']= PowerLaw(name='luminosity_distance', minimum=5, maximum=300, alpha=2, latex_label='$d_L$')
# create likelihood
ifos= [H1, L1, V1]
apx= 'IMRPhenomPv2'

waveargs= dict(waveform_approximant= apx, reference_frequency=100, catch_waveform_errors=True)
wavegen= bilby.gw.WaveformGenerator(frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
                                    waveform_arguments= waveargs,
                                    parameter_conversion= convert_to_lal_binary_black_hole_parameters)
likelihood= bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform_generator= wavegen, priors= priors, time_marginalization=True, phase_marginalization=True, distance_marginalization=True)
#run sampler
result= bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', outdir='prunier_PE',
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters, label= 'Prunier Event', printdt= 30, clean=True)
result.plot_corner()
