global np

from multiprocessing import Process

#import nest.raster_plot
import raster_plot

import os
import json
import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from threading import Thread

global nest, time, plt, gridspec
global to2D
global N, tInterval, tStart, wEEInter, wEEIntra, events, tTotal, getCV
global tau_m, t_ref, V_th, V_reset, ratio_nuExt_nuThr, J, g, gamma, C_e, w_e, C_e_inter_1Dto2D, C_e_inter_2Dto1D
global widthStimulus, w_e_intra
global convert
global syn_exc
global ts, events, nR1e, nR2e, nF1e, nF2e, nLe, directory, ns
global rate, rates, spikeSenders, spikeTimes
global stretchGaussian, factor_inter_1Dto2D
global neighborhoods1D, neighborhoods2D, N_neighborhoods, neighborhoods_inter_gauss_1Dto2D, neighborhoods_inter_gauss_and_uniform_2Dto1D
global connect_intra_1D, connect_intra_2D, connect_inter_f, connect_inter_l
global winner





#nThreads = 32/nExe+1
nThreads = max(2, 32/nExe)
'''
if idRun == 0:
	nThreads = 1
if idRun == 1:
	nThreads = 2
if idRun == 2:
	nThreads = 4
if idRun == 3:
	nThreads = 8
'''

#baseseed = 21234

baseseed = 200000000*idRun

'''
if idRun == 0:
	baseseed = 1000000
if idRun == 1:
	baseseed = 2000000
if idRun == 2:
	baseseed = 3000000
if idRun == 3:
	baseseed = 4000000
'''



'''
"Variable" parameters
'''

N=300
#N=50

'''
if idRun == 0:
	N=300
	nThreads = 4
if idRun == 1:
	N=500
	nThreads = 4
if idRun == 2:
	N=800
	nThreads = 8
if idRun == 3:
	N=1000
	nThreads = 16
'''

factorNuExt = 0.7
'''
if idRun == 0:
	factorNuExt = .65
if idRun == 1:
	factorNuExt = .67
if idRun == 2:
	factorNuExt = .7
if idRun == 3:
	factorNuExt = .8
'''

factor_e_to_i = 5.5 # 6
'''
if idRun == 0:
	factor_e_to_i = 5
if idRun == 1:
	factor_e_to_i = 6
if idRun == 2:
	factor_e_to_i = 8
if idRun == 3:
	factor_e_to_i = 10
'''

ratio_stimulus_noise = 1.1		# ( rate(noise) + rate(stimulus) ) / rate(noise)
'''
if idRun == 0:
	ratio_stimulus_noise = 1.0
if idRun == 1:
	ratio_stimulus_noise = 1.1
if idRun == 2:
	ratio_stimulus_noise = 1.3
if idRun == 3:
	ratio_stimulus_noise = 1.5
'''

stretchGaussian = 1
'''
if idRun == 0:
	stretchGaussian = 1
if idRun == 1:
	stretchGaussian = 2
if idRun == 2:
	stretchGaussian = 3
if idRun == 3:
	stretchGaussian = 4
'''

'''
factor_intra = .95

if idRun == 0:
	factor_intra = 1.
if idRun == 1:
	factor_intra = .95
if idRun == 2:
	factor_intra = .9
if idRun == 3:
	factor_intra = .8
'''

factor_attention = 3. #3.
'''
if idRun == 0:
	factor_attention = 3.
if idRun == 1:
	factor_attention = 4.
if idRun == 2:
	factor_attention = 5.
if idRun == 3:
	factor_attention = 6.
'''




C_e = 100
'''
if idRun ==0:
	C_e=95
if idRun ==1:
	C_e=100
if idRun ==2:
	C_e=105
if idRun ==3:
	C_e=110
'''





factor_inter_1Dto2D = 0.01 # 0.015
'''
if idRun == 0:
	factor_inter_1Dto2D = .01
if idRun == 1:
	factor_inter_1Dto2D = .015
if idRun == 2:
	factor_inter_1Dto2D = .02
if idRun == 3:
	factor_inter_1Dto2D = .03
'''

factor_inter_2Dto1D = 1.5 # 1.2
'''
if idRun == 0:
	factor_inter_2Dto1D = 1.2
if idRun == 1:
	factor_inter_2Dto1D = 1.25
if idRun == 2:
	factor_inter_2Dto1D = 1.3
if idRun == 3:
	factor_inter_2Dto1D = 1.4
'''







'''
Simulation parameters
'''

#tInterval = "eternity"
tInterval = 50

if case == "short-very":
	tInit = 4*tInterval
	tStimulus = 1*tInterval
	tFinal = 11*tInterval
elif case == "short":
	tInit = 4*tInterval
	tStimulus = 2*tInterval
	tFinal = 10*tInterval
else:
	tInit = 4*tInterval
	tStimulus = 8*tInterval
	tFinal = 4*tInterval



attentionL = "none"
attentionF1 = 1
'''
if idRun == 0:
	attentionL = "none"
if idRun == 1:
	attentionL = 1
if idRun == 2:
	attentionF1 = 1
if idRun == 3:
	attentionF1 = 2
'''


N_stimuli = 2

o1l=0.3
o1f1=0.3
o1f2=0.3
o2l=0.6
o2f1=0.8
o2f2=0.45

widthStimulus = 0.05



'''
End parameters
'''


if not ('idRun' in globals()):
	idRun = 0


executionTimeStart = time.time()
executionClockStart = time.clock()
print("status "+str(idRun)+": start script")

nest.set_verbosity("M_WARNING")
nest.ResetKernel()


def rate(countSpikesSingle):
	return countSpikesSingle * 1000.0/tInterval

def rates(countSpikes, pop):
	return [rate(countSpikes[n-1]) for n in pop]

def getCV(times):
	if len(times) < 2:
		return 0
	isi=[]
	for i in range(len(times)-1):
		isi.append(times[i+1]-times[i])
	E_isi = np.mean(isi)
	Var_isi = np.var(isi)
	#f = 1./Eisi
	cv = np.sqrt(Var_isi) / E_isi
	#print("f="+str(f)+";cv="+str(cv))
	return cv

def timeToStr(seconds):
	minutes = seconds/60
	hours = minutes/60
	return str(seconds)+" s     = "+str(minutes)+" min     = "+str(hours)+" h"


'''
Process parameters
'''


g = 5.
'''
if idRun == 0:
	g = 3.
if idRun == 1:
	g = 4.
if idRun == 2:
	g = 5.
if idRun == 3:
	g = 8.
'''


ratio_nuExt_nuThr = 1.5

'''
if idRun == 0:
	ratio_nuExt_nuThr = 1.5
if idRun == 1:
	ratio_nuExt_nuThr = 1.4
if idRun == 2:
	ratio_nuExt_nuThr = 1.3
if idRun == 3:
	ratio_nuExt_nuThr = 1.2
if idRun == 4:
	ratio_nuExt_nuThr = 1.1
if idRun == 5:
	ratio_nuExt_nuThr = 1.0
'''

C_i = C_e / 4

N1D = N
N2D = N*N


tTotal = tInit + tStimulus + tFinal
ts = np.arange(tInterval,tTotal+1,tInterval)

'''
"Fixed" neuron parameters from brunel
'''

E_L = 0. # mV
C_m = 1.0 # pF
tau_m_ms = 20. # ms
t_ref_ms = 2. # ms
V_th = 20. # mV
V_reset = 10. # mV

delay = 2.

tau_m_s = tau_m_ms / 1000.
t_ref_s = t_ref_ms / 1000.


neuron_pars = {
	"V_m":		0.0,
	"E_L":		E_L,
	"C_m":		C_m,
	"tau_m":	tau_m_ms,
	"t_ref":	t_ref_ms,
	"V_th":		V_th,
	"V_reset":	V_reset,
	"I_e":		0.0
}

'''
Other "fixed" parameters from brunel
'''

gamma = 0.25
J = 0.1 # mV



nuThr_Hz = V_th / (C_e * J * tau_m_s)
nuExt_Hz = ratio_nuExt_nuThr * nuThr_Hz
nuExt_kHz = nuExt_Hz / 1000.

nu_estimation = False

if nu_estimation:
	from scipy import optimize
	import siegert_interpolator
	from estimate_nu import *
	nuInit = 10.
	nu_estimated = estimate_nu(nuInit=nuInit, nuExt=nuExt_kHz, C_e=C_e, V_reset=V_reset, V_th=V_th, J=J, tau_m=tau_m_s, t_ref=t_ref_s, g=g, gamma=gamma)
	print("estimated nu: "+str(nu_estimated))
	import sys
	sys.exit("stop script......")

print("parameters:\n nuExt="+str(nuExt_Hz)+" Hz\n nuExt="+str(nuExt_kHz)+" kHz\n nuThr="+str(nuThr_Hz)+" Hz\n C_e="+str(C_e)+"\n V_reset="+str(V_reset)+"\n V_th="+str(V_th)+"\n J="+str(J)+"\n tau_m="+str(tau_m_ms)+"\n t_ref="+str(t_ref_ms)+"\n g="+str(g)+"\n gamma="+str(gamma))

'''
End definition brunel parameters. What follows is specific to this project.
'''




'''
factor_inter_1Dto2D = 1. - factor_intra

C_e_intra = int(C_e * factor_intra)
C_e_inter_1Dto2D = int(C_e * factor_inter_1Dto2D)
'''

C_e_intra = C_e


C_e_inter_1Dto2D = int(C_e * factor_inter_1Dto2D)
C_e_inter_2Dto1D = int(C_e * factor_inter_2Dto1D)

A_stimulus = N_stimuli * widthStimulus*widthStimulus



ratePoissonExt = nuExt_Hz*C_e * factorNuExt

ratePoissonNoise = ratePoissonExt / (A_stimulus * (ratio_stimulus_noise-1) + 1)
ratePoissonStimulus = ratePoissonNoise * (ratio_stimulus_noise-1)
ratePoissonAttention = ratePoissonStimulus * factor_attention

print("ratePoissonExt = "+str(ratePoissonExt)+"\nratePoissonNoise = "+str(ratePoissonNoise)+"\nratePoissonStimulus = "+str(ratePoissonStimulus))


conn_exc_e_to_i = {'rule': 'fixed_indegree', 'indegree': int(C_e * factor_e_to_i)}
conn_inh = {'rule': 'fixed_indegree', 'indegree': C_i}

w_e = J # according to brunel, each neuron should have C_e incoming connections with weight J from other excitatory neurons
w_i = -g * w_e

syn_inh = {"weight":w_i}



nest.SetKernelStatus({'local_num_threads': nThreads})  # sets number of cpus
baseseedRange = range(baseseed+nThreads+1, baseseed+1+2*nThreads)
nest.SetKernelStatus({'grng_seed' : baseseed+nThreads, 'rng_seeds': baseseedRange})

'''
Create nodes
'''

print("status "+str(idRun)+": create neurons")

nR1e = nest.Create("iaf_psc_delta", N*N)
nR1i = nest.Create("iaf_psc_delta", int((N*N)*gamma)) # in order to have 20% inh, 80% exc
nR2e = nest.Create("iaf_psc_delta", N*N)
nR2i = nest.Create("iaf_psc_delta", int((N*N)*gamma)) # in order to have 20% inh, 80% exc

nF1e = nest.Create("iaf_psc_delta", N)
nF1i = nest.Create("iaf_psc_delta", int(N*gamma))
nF2e = nest.Create("iaf_psc_delta", N)
nF2i = nest.Create("iaf_psc_delta", int(N*gamma))

nLe = nest.Create("iaf_psc_delta", N)
nLi = nest.Create("iaf_psc_delta", int(N*gamma))

ns1D = nF1e+nF1i+nF2e+nF2i+nLe+nLi
ns2D = nR1e+nR1i+nR2e+nR2i
ns = ns1D + ns2D

poissonNoise = nest.Create("poisson_generator")
poissonStimulus = nest.Create("poisson_generator")
poissonAttention = nest.Create("poisson_generator")


spd = nest.Create("spike_detector")

'''
Set status of nodes
'''

nest.SetStatus(poissonNoise, "rate", ratePoissonNoise)

nest.SetStatus(ns, neuron_pars)

'''
Connect neurons
'''

print("status "+str(idRun)+": connect neurons")

'''
Connect poisson neurons as external input to neurons
'''

nest.SetDefaults("static_synapse", {"delay" : delay, "weight" : w_e})

nest.Connect(poissonNoise, ns)


def convert(pos):
	return np.int(pos*(N-1))

nWidthStimulus = np.int(np.ceil(widthStimulus * N/2 - 1))

n_R1_o1 = []
n_R2_o1 = []
n_R1_o2 = []
n_R2_o2 = []

n_R1_other = []
n_R2_other = []

n_F2_o1 = []
n_F2_o2 = []

nAttention = []
for fOffset in np.arange(-nWidthStimulus, nWidthStimulus+1) :
	for lOffset in np.arange(-nWidthStimulus, nWidthStimulus+1) :
		n_R1_o1.append( nR1e[((convert(o1f1)+fOffset)%N)*N + (convert(o1l)+lOffset)%N] )
		n_R1_o2.append( nR1e[((convert(o2f1)+fOffset)%N)*N + (convert(o2l)+lOffset)%N] )
		n_R2_o1.append( nR2e[((convert(o1f2)+fOffset)%N)*N + (convert(o1l)+lOffset)%N] )
		n_R2_o2.append( nR2e[((convert(o2f2)+fOffset)%N)*N + (convert(o2l)+lOffset)%N] )
		
		n_R1_other.append( nR1e[((0+fOffset)%N)*N + (0+lOffset)%N] )
		n_R2_other.append( nR2e[((0+fOffset)%N)*N + (0+lOffset)%N] )

	if attentionL == 1:
		nAttention.append( nLe[(convert(o1l)+fOffset)%N] )
	elif attentionL == 2:
		nAttention.append( nLe[(convert(o1l)+fOffset)%N] )
	if attentionF1 == 1:
		nAttention.append( nF1e[(convert(o1f1)+fOffset)%N] )
	elif attentionF1 == 2:
		nAttention.append( nF1e[(convert(o2f1)+fOffset)%N] )
	
	n_F2_o1.append( nF2e[(convert(o1f2)+fOffset)%N] )
	n_F2_o2.append( nF2e[(convert(o2f2)+fOffset)%N] )

nStimulus = np.concatenate([n_R1_o1, n_R2_o1, n_R1_o2, n_R2_o2])



nest.Connect(poissonStimulus, nStimulus.tolist())
nest.Connect(poissonAttention, nAttention)

'''
Connect actual network internally
'''
# structure of nRx: nRx[f*N+l]

'''
Create neighborhoods
'''
N_neighborhoods = 1000
neighborhoods1D = []
neighborhoods2D = []
neighborhoods_inter_gauss_1Dto2D = []
neighborhoods_inter_gauss_and_uniform_2Dto1D = []

t1connect = time.time()

for i in range(N_neighborhoods):
	neighborhood = []
	for j in range(C_e_intra):
		d = np.random.normal() * widthStimulus * N/2 * stretchGaussian
		d = int(np.sign(d)) * (abs(int(d))+1)
		neighborhood.append(d)
	neighborhoods1D.append(neighborhood)

for i in range(N_neighborhoods):
	neighborhood = []
	for j in range(C_e_intra):
		f_d = np.random.normal() * widthStimulus * N/2 * stretchGaussian
		f_d = int(np.sign(f_d)) * (abs(int(f_d))+1)
		l_d = np.random.normal() * widthStimulus * N/2 * stretchGaussian
		l_d = int(np.sign(l_d)) * (abs(int(l_d))+1)
		neighborhood.append((f_d, l_d))
	neighborhoods2D.append(neighborhood)

for i in range(N_neighborhoods):
	neighborhood = []
	for j in range(C_e_inter_1Dto2D):
		d = np.random.normal() * widthStimulus * N/2 * stretchGaussian
		d = int(np.sign(d)) * (abs(int(d))+1)
		neighborhood.append(d)
	neighborhoods_inter_gauss_1Dto2D.append(neighborhood)


for i in range(N_neighborhoods):
	neighborhood = []
	for j in range(C_e_inter_2Dto1D):
		d = np.random.normal() * widthStimulus * N/2 * stretchGaussian
		d = int(np.sign(d)) * (abs(int(d))+1)
		pos = int( np.random.uniform() * N )
		neighborhood.append((d,pos))
	neighborhoods_inter_gauss_and_uniform_2Dto1D.append(neighborhood)


'''
define connection functions
'''

# using scipy.random.randint would be faster than np.random.randint

def connect_intra_1D(ne):
	neighbors_list = []
	target_list = []
	for i in range(N):
		neighborhood = neighborhoods1D[np.random.randint(0,N_neighborhoods)]
		n_i = ne[i]
		for d in neighborhood:
			iNeighbor = (i+d) % N
			neighbors_list.append(ne[ iNeighbor ])
			target_list.append(n_i)
	nest.Connect(neighbors_list, target_list, "one_to_one")
	

def connect_intra_2D(ne):
	neighbors_list = []
	target_list = []
	for f in range(N):
		for l in range(N):
			neighborhood = neighborhoods2D[np.random.randint(0,N_neighborhoods)]
			n_i = ne[f*N+l]
			for ds in neighborhood:
				f_neighbor = (f+ds[0]) % N
				l_neighbor = (l+ds[1]) % N
				iNeighbor = f_neighbor * N + l_neighbor
				neighbors_list.append(ne[ iNeighbor ])
				target_list.append(n_i)
	nest.Connect(neighbors_list, target_list, "one_to_one")

def connect_inter_f(nRe, nFe):
	neighbors_list = []
	target_list = []
	
	for f in range(N):
		for l in range(N):
			n_i = nRe[f*N+l]
			neighborhood = neighborhoods_inter_gauss_1Dto2D[np.random.randint(0,N_neighborhoods)]
			for d in neighborhood:
				iNeighbor = (f+d) % N
				neighbors_list.append(nFe[ iNeighbor ])
				target_list.append(n_i)
	
	for f in range(N):
		n_f = nFe[f]
		neighborhood = neighborhoods_inter_gauss_and_uniform_2Dto1D[np.random.randint(0,N_neighborhoods)]
		for neighbor in neighborhood:
			df = neighbor[0]
			f_neighbor = (f+df) % N
			l_neighbor = neighbor[1]
			iNeighbor = f_neighbor * N + l_neighbor
			neighbors_list.append(nRe[ iNeighbor ])
			target_list.append(n_f)
	
	nest.Connect(neighbors_list, target_list, "one_to_one")

def connect_inter_l(nRe, nLe):
	neighbors_list = []
	target_list = []
	
	for f in range(N):
		for l in range(N):
			n_i = nRe[f*N+l]
			neighborhood = neighborhoods_inter_gauss_1Dto2D[np.random.randint(0,N_neighborhoods)]
			for d in neighborhood:
				iNeighbor = (l+d) % N
				neighbors_list.append(nLe[ iNeighbor ])
				target_list.append(n_i)
	
	for l in range(N):
		n_l = nLe[l]
		neighborhood = neighborhoods_inter_gauss_and_uniform_2Dto1D[np.random.randint(0,N_neighborhoods)]
		for j in range(C_e_inter_2Dto1D / 2):
			neighbor = neighborhood[j]
			f_neighbor = neighbor[1]
			dl = neighbor[0]
			l_neighbor = (l+dl) % N
			iNeighbor = f_neighbor * N + l_neighbor
			neighbors_list.append(nRe[ iNeighbor ])
			target_list.append(n_l)
	
	nest.Connect(neighbors_list, target_list, "one_to_one")


'''
local connections
'''
t2connect = time.time()
c2 = time.clock()

connect_intra_2D(nR1e)
connect_intra_2D(nR2e)
connect_intra_1D(nF1e)
connect_intra_1D(nF2e)
connect_intra_1D(nLe)

connect_inter_f(nR1e, nF1e)
connect_inter_f(nR2e, nF2e)
connect_inter_l(nR1e, nLe)
connect_inter_l(nR2e, nLe)


t4connect = time.time()
c4 = time.clock()
print("time for making gaussian connections - create neighborhoods : "+str(t2connect-t1connect))
print("time for making gaussian connections - connect them actually - time : "+str(t4connect-t2connect))
print("time for making gaussian connections - connect them actually - clock: "+str(c4-c2))


nest.Connect(nR1e,nR1i, conn_spec=conn_exc_e_to_i)
nest.Connect(nR1i,nR1i, conn_spec=conn_inh, syn_spec=syn_inh)
nest.Connect(nR1i,nR1e, conn_spec=conn_inh, syn_spec=syn_inh)

nest.Connect(nR2e,nR2i, conn_spec=conn_exc_e_to_i)
nest.Connect(nR2i,nR2i, conn_spec=conn_inh, syn_spec=syn_inh)
nest.Connect(nR2i,nR2e, conn_spec=conn_inh, syn_spec=syn_inh)

nest.Connect(nF1e,nF1i, conn_spec=conn_exc_e_to_i)
nest.Connect(nF1i,nF1i, conn_spec=conn_inh, syn_spec=syn_inh)
nest.Connect(nF1i,nF1e, conn_spec=conn_inh, syn_spec=syn_inh)

nest.Connect(nF2e,nF2i, conn_spec=conn_exc_e_to_i)
nest.Connect(nF2i,nF2i, conn_spec=conn_inh, syn_spec=syn_inh)
nest.Connect(nF2i,nF2e, conn_spec=conn_inh, syn_spec=syn_inh)

nest.Connect(nLe,nLi, conn_spec=conn_exc_e_to_i)
nest.Connect(nLi,nLi, conn_spec=conn_inh, syn_spec=syn_inh)
nest.Connect(nLi,nLe, conn_spec=conn_inh, syn_spec=syn_inh)

'''
Connect neurons to spike detector
'''

nest.Connect(ns, spd)


# SET THE CONNECTION DELAY ONCE FOR ALL CONNECTIONS .....................................!!!!!!!!!!!!!!!!!!

#if idRun == 1:
#	nest.SetStatus(nest.GetConnections(), syn_spec, {"delay" : delay})


#sys.exit("stop it!!!!!!")

'''
Simulate
'''

print("status "+str(idRun)+": start simulation")

nest.SetStatus(poissonStimulus, "rate", 0.)
#nest.SetStatus(poissonAttention, "rate", 0.)
nest.SetStatus(poissonAttention, "rate", ratePoissonAttention)

nest.Simulate(tInit)

nest.SetStatus(poissonStimulus, "rate", ratePoissonStimulus)
nest.SetStatus(poissonAttention, "rate", ratePoissonAttention)

nest.Simulate(tStimulus)

nest.SetStatus(poissonStimulus, "rate", 0.)
nest.SetStatus(poissonAttention, "rate", 0.)

nest.Simulate(tFinal)

print("status "+str(idRun)+": simulation finished")

simulationTimeEnd = time.time()
simulationClockEnd = time.clock()



spikeEvents = nest.GetStatus(spd,"events")[0]
spikeSenders = np.array(spikeEvents["senders"])
spikeTimes = np.array(spikeEvents["times"])

def getPopRate(pop):
	popMin = min(pop)
	popMax = max(pop)
	return sum([(sender>=popMin)&(sender<=popMax) for sender in spikeSenders]) * 1000.0/tTotal / len(pop)
rateR1e = getPopRate(nR1e)
rateR1i = getPopRate(nR1i)
rateR2e = getPopRate(nR2e)
rateR2i = getPopRate(nR2i)
rateF1e = getPopRate(nF1e)
rateF1i = getPopRate(nF1i)
rateF2e = getPopRate(nF2e)
rateF2i = getPopRate(nF2i)
rateLe = getPopRate(nLe)
rateLi = getPopRate(nLi)

def getPopCV(pop):
	return np.mean([getCV(spikeTimes[spikeSenders==sender]) for sender in pop])
cvR1e = getPopCV(nR1e)
cvR2e = getPopCV(nR2e)
cvF1e = getPopCV(nF1e)
cvF2e = getPopCV(nF2e)
cvLe = getPopCV(nLe)


print("status "+str(idRun)+": plot rates")

directory = "out/run_"+str(idRun)+"/"
if not os.path.exists(directory):
	os.makedirs(directory)

def to2D(arr):
	return np.concatenate([arr,arr]).reshape(2,len(arr))

def visualizeInterval(iInterval):
	tIntervalEnd = ts[iInterval]
	tIntervalStart = tIntervalEnd - tInterval
	
	sendersInterval = spikeSenders[ (spikeTimes>=tIntervalStart) & (spikeTimes<tIntervalEnd) ]
	
	countSpikes = np.zeros(len(ns))
	
	for sender in sendersInterval:
		countSpikes[sender-1] += 1
	
	matR1e = np.mat(rates(countSpikes, nR1e))
	matR1e = matR1e.reshape(N,N).transpose()
	matR2e = np.mat(rates(countSpikes, nR2e))
	matR2e = matR2e.reshape(N,N).transpose()
	matF1e = to2D(np.array(rates(countSpikes, nF1e)))
	matF2e = to2D(np.array(rates(countSpikes, nF2e)))
	matL = to2D(np.array(rates(countSpikes, nLe)))
	matL = matL.transpose()

	minRate = 0
	maxRate = max(matR1e.max(), matR1e.max(), matR2e.max(), matR2e.max(), matF1e.max(), matF1e.max(), matF2e.max(), matF2e.max(), matL.max())
	
	
	f = plt.figure(figsize=(10,5))
	gs = gridspec.GridSpec(2,4,width_ratios=[10,1,10,1],height_ratios=[1,10])

	axR1 = plt.subplot(gs[4])
	axF1 = plt.subplot(gs[0], sharex=axR1)
	axR2 = plt.subplot(gs[6], sharey=axR1)
	axF2 = plt.subplot(gs[2], sharex=axR2)
	axL = plt.subplot(gs[5], sharey=axR1)
	axColorbar = plt.subplot(gs[7])

	imR1 = axR1.matshow(matR1e, vmin=minRate, vmax=maxRate)
	imR2 = axR2.matshow(matR2e, vmin=minRate, vmax=maxRate)
	imF1 = axF1.matshow(matF1e, vmin=minRate, vmax=maxRate, aspect="auto")
	imF2 = axF2.matshow(matF2e, vmin=minRate, vmax=maxRate, aspect="auto")
	imL = axL.matshow(matL, vmin=minRate, vmax=maxRate, aspect="auto")
	
	#axF1.xaxis.set_label_position("top")
	axF1.set_xlabel("feature 1 (color)")
	axF2.set_xlabel("feature 2 (shape)")
	axL.set_ylabel("location")
	axL.yaxis.set_label_position("right")
	
	plt.colorbar(imR1, cax=axColorbar)
	
	
	plt.suptitle("time = [ "+str(int(tIntervalEnd-tInterval))+" , "+str(int(tIntervalEnd))+" ]")
	
	axR1.get_xaxis().set_visible(False)
	axR1.get_yaxis().set_visible(False)
	axR2.get_xaxis().set_visible(False)
	axR2.get_yaxis().set_visible(False)
	axF1.get_yaxis().set_visible(False)
	axF2.get_yaxis().set_visible(False)
	axL.get_xaxis().set_visible(False)
	#axL.get_xaxis().tick_bottom()

	#f.show()
	
	plt.savefig(directory+'interval_'+str(iInterval)+'.png', bbox_inches='tight')
	plt.clf()
	plt.close()


for iInterval in range(len(ts)):
	visualizeInterval(iInterval)

'''
raster plot
'''
#raster_plot.from_device(spd, hist=True, hist_binwidth=5.0)


n_raster_plot = n_R1_other + n_R1_o1 + n_R1_o2 + n_R2_other + n_R2_o1 + n_R2_o2

indices_raster = np.array([(spikeSender in n_raster_plot) for spikeSender in spikeSenders])
spikeSenders_raster = spikeSenders[indices_raster]
spikeTimes_raster = spikeTimes[indices_raster]

spikeSenders_raster_normalized = [np.where(n_raster_plot==spikeSender)[0][0] for spikeSender in spikeSenders_raster]

marker = "," #"|" # ","
plt.plot(spikeTimes_raster, spikeSenders_raster_normalized, marker, color="black")

countPatch = len(n_R1_other)

color_o1 = "#ffaaaa"
color_o2 = "#aaffaa"
color_other = "#aaaaff"

plt.axhspan(0*countPatch, 1*countPatch, facecolor=color_other, edgecolor=color_other)
plt.axhspan(1*countPatch, 2*countPatch, facecolor=color_o1, edgecolor=color_o1)
plt.axhspan(2*countPatch, 3*countPatch, facecolor=color_o2, edgecolor=color_o2)
plt.axhspan(3*countPatch, 4*countPatch, facecolor=color_other, edgecolor=color_other)
plt.axhspan(4*countPatch, 5*countPatch, facecolor=color_o1, edgecolor=color_o1)
plt.axhspan(5*countPatch, 6*countPatch, facecolor=color_o2, edgecolor=color_o2)

color_stimulus_edge = "#ffa500"
color_stimulus_face = "#ffedcc"
plt.axvspan(tInit, tInit, facecolor=color_stimulus_face, edgecolor=color_stimulus_edge)
plt.axvspan(tInit+tStimulus, tInit+tStimulus, facecolor=color_stimulus_face, edgecolor=color_stimulus_edge)

plt.ylim( (0, 6*countPatch) )

plt.xlabel("time / ms")
plt.ylabel("neuron groups")

yticks_pos = np.array(range(6))*countPatch + countPatch/2
yticks_str = ["F1L map", "F1L obj T", "F1L obj D", "F2L map", "F2L obj T", "F2L obj D"]
plt.yticks(yticks_pos, yticks_str)

plt.title("Raster plot of selected neuron groups")

plt.savefig(directory+'1_raster_plot.png', bbox_inches='tight')
plt.clf()
plt.close()

'''
Retreive decision
'''


global getMeanCountSpikesPerMs, getSmoothenedRates, getWinner, convolutionArr

nF1e_arr = np.array(nF1e)
nF2e_arr = np.array(nF2e)

tMeasure = tInterval

convolutionArr = np.array( [1000/tMeasure] * tMeasure )


def getMeanCountSpikesPerMs(pop):
	countSpikes_pop = np.zeros(tTotal)
	spikeIndices_pop = np.array([(spikeSender==pop).any() for spikeSender in spikeSenders])
	spikeTimes_pop = spikeTimes[ spikeIndices_pop ]
	singleSpikeMean = 1. / len(pop)
	for spikeTime in spikeTimes_pop:
		countSpikes_pop[spikeTime] += singleSpikeMean
	return countSpikes_pop

def getSmoothenedRates(pop):
	return np.convolve(getMeanCountSpikesPerMs(pop), convolutionArr, mode="valid")

def getWinner(n_arrs):
	smoothenedRates_competitors = np.array([getSmoothenedRates(n_arr) for n_arr in n_arrs])
	winner = smoothenedRates_competitors.argmax(axis=0)
	return winner


plt.plot(getSmoothenedRates(nF2e_arr),label="map",c="blue")
plt.plot(getSmoothenedRates(n_F2_o1),label="obj T",c="red")
plt.plot(getSmoothenedRates(n_F2_o2),label="obj D",c="green")

color_stimulus_edge = "#ffa500"
color_stimulus_face = "#ffedcc"
plt.axvspan(tInit-tInterval, tInit+tStimulus-tInterval, facecolor=color_stimulus_face, edgecolor=color_stimulus_edge)

plt.xticks(ts-tInterval, ts)
plt.legend()
plt.xlabel("time / ms")
plt.ylabel("mean rate / Hz")
plt.title("Rates in F2")
plt.savefig(directory+'0_smoothenedrate.png', bbox_inches='tight')
plt.clf()
plt.close()

winner = getWinner([nF2e_arr, n_F2_o1, n_F2_o2])
plt.plot(winner)
plt.xticks(ts-tInterval, ts)
plt.savefig(directory+'2_winner.png', bbox_inches='tight')
plt.clf()




'''
Create output files
'''

executionTimeEnd = time.time()
executionClockEnd = time.clock()

pars = json.dumps([baseseed,ratePoissonStimulus,ratePoissonAttention])

strOut = "============= results, idRun: "+str(idRun)+" =================" +\
"\ntotal       time  elapsed: "+timeToStr(executionTimeEnd-executionTimeStart) +\
"\ntotal       clock elapsed: "+timeToStr(executionClockEnd-executionClockStart) +\
"\nsimulation  time  elapsed: "+timeToStr(simulationTimeEnd-executionTimeStart) +\
"\nsimulation  clock elapsed: "+timeToStr(simulationClockEnd-executionClockStart) +\
"\nspike count time  elapsed: "+timeToStr(executionTimeEnd-simulationTimeEnd) +\
"\nspike count clock elapsed: "+timeToStr(executionClockEnd-simulationClockEnd) +\
"\nrate R1e: "+str(rateR1e) +\
"\nrate R2e: "+str(rateR2e) +\
"\nrate F1e: "+str(rateF1e) +\
"\nrate F2e: "+str(rateF2e) +\
"\nrate Le : "+str(rateLe) +\
"\nrate R1i: "+str(rateR1i) +\
"\nrate R2i: "+str(rateR2i) +\
"\nrate F1i: "+str(rateF1i) +\
"\nrate F2i: "+str(rateF2i) +\
"\nrate Li : "+str(rateLi) +\
"\ncv R1e: "+str(cvR1e) +\
"\ncv R2e: "+str(cvR2e) +\
"\ncv F1e: "+str(cvF1e) +\
"\ncv F2e: "+str(cvF2e) +\
"\ncv Le : "+str(cvLe)


print(strOut)

f=open(directory+"parameters_and_results.json", "w")
f.write(pars)
f.write(strOut)
f.close()


print("status "+str(idRun)+": finished")
