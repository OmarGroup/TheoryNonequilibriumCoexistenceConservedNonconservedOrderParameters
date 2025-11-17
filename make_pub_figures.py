import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema
from scipy.integrate import trapz
import matplotlib

def get_coex(P, phi, spinodall, spinodalh, E_rho):
	if min(phi) > 0 and spinodall < 0:
		spinodall += 1.0
		spinodall /= 2.0
		spinodalh += 1.0
		spinodalh /= 2.0
	phinew = phi * 1
	min_P = P[np.argmin(np.abs(phinew - spinodalh))]
	max_P = P[np.argmin(np.abs(phinew - spinodall))]
	phi_P_cp = max(phi)
	phi1, phi2, P_coex = None, None, None
	best = None
	if len(phinew[((P - min_P) > 0) & (phinew < spinodall)]) > 0:
		min_phi = min(phi[((P - min_P) > 0) & (phi < spinodall)])
		if len(phi[((P - max_P) < 0) & (phi > spinodalh)]) == 0:
			spinodall -= 1e-4
			print()
			print(max_P)
			max_P = P[np.argmin(np.abs(phinew - spinodall))]
			print(max_P)
			print()
		max_phi = max(phi[((P - max_P) < 0) & (phi > spinodalh)])
		phi_init = phi[(phi > min_phi) & (phi < spinodall)]
		P_init = P[(phi > min_phi) & (phi < spinodall)]
		P_init = P_init[phi_init < phi_P_cp]
		phi_init = phi_init[phi_init < phi_P_cp]
		P_fin = P[(phi < max_phi) & (phi > spinodalh)]
		phi_fin = phi[(phi < max_phi) & (phi > spinodalh)][np.argmin(np.abs(np.subtract.outer(P_fin, P_init)), axis=0)]
		if len(P_fin) > 0 and len(P_init) > 0 and len(phi_fin) > 0:
			P_relevant = P[(phi < max_phi) & (phi > min_phi)]
			phi_relevant = phi[(phi < max_phi) & (phi > min_phi)]
			best = 100000000000000
			for i in range(len(phi_init)):
				if np.abs(phi_init[i] - phi_fin[i]) > 0.001:
					# print('trying')
					if len(E_rho) < len(phi):
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[1:] * np.diff(E_rho[(phi > phi_init[i]) & (phi < phi_fin[i])])[:]))
					else:
						integral = np.abs(np.sum((P_init[i] - P[(phi > phi_init[i]) & (phi < phi_fin[i])])[1:] * np.diff(E_rho[(phi > phi_init[i]) & (phi < phi_fin[i])])[:]))
					if integral < best:
						best = integral
						phi1 = phi_init[i]
						phi2 = phi_fin[i]
						P_coex = P_init[i]
	return phi1, phi2, P_coex, best


"""
# First let's store the relevant data
phiAfs_num_ambA1 = []
phiAss_num_ambA1 = []
phiBfs_num_ambA1 = []
phiBss_num_ambA1 = []
ambA1s = []
for file in os.listdir(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA/'):
	if file[:2] == '1d' in file:
		ambA = float(file.split('_m_0.t')[0].split('_')[-1])
		data = np.loadtxt(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA/' + file, delimiter=',')
		phiA = data[:, 1]
		phiB = data[:, 2]
		ambA1s.append(ambA)
		phiA1 = phiA[0]
		phiA2  = phiA[-1]
		phiB1 = phiB[0]
		phiB2 = phiB[-1]
		phiAs = [phiA1, phiA2]
		phiBs = [phiB1, phiB2]
		fi = np.argmin(phiAs)
		si = np.argmax(phiAs)
		phiAfs_num_ambA1.append(phiAs[fi])
		phiAss_num_ambA1.append(phiAs[si])
		phiBfs_num_ambA1.append(phiBs[fi])
		phiBss_num_ambA1.append(phiBs[si])
sort_ambA1 = np.argsort(ambA1s)
coex_data_ambA1 = np.hstack((np.asarray(ambA1s)[sort_ambA1].reshape(-1, 1), np.asarray(phiAfs_num_ambA1)[sort_ambA1].reshape(-1, 1), np.asarray(phiBfs_num_ambA1)[sort_ambA1].reshape(-1, 1), np.asarray(phiAss_num_ambA1)[sort_ambA1].reshape(-1, 1), np.asarray(phiBss_num_ambA1)[sort_ambA1].reshape(-1, 1)))
np.savetxt('coex_data_ambA1.txt', coex_data_ambA1)

phiAfs_num_ambA2 = []
phiAss_num_ambA2 = []
phiBfs_num_ambA2 = []
phiBss_num_ambA2 = []
ambA2s = []
for file in os.listdir(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA2/'):
	if file[:2] == '1d' in file:
		ambA = float(file.split('_m_0.t')[0].split('_')[-1])
		data = np.loadtxt(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA2/' + file, delimiter=',')
		phiA = data[:, 1]
		phiB = data[:, 2]
		ambA2s.append(ambA)
		phiA1 = phiA[0]
		phiA2  = phiA[-1]
		phiB1 = phiB[0]
		phiB2 = phiB[-1]
		phiAs = [phiA1, phiA2]
		phiBs = [phiB1, phiB2]
		fi = np.argmin(phiAs)
		si = np.argmax(phiAs)
		phiAfs_num_ambA2.append(phiAs[fi])
		phiAss_num_ambA2.append(phiAs[si])
		phiBfs_num_ambA2.append(phiBs[fi])
		phiBss_num_ambA2.append(phiBs[si])
sort_ambA2 = np.argsort(ambA2s)
coex_data_ambA2 = np.hstack((np.asarray(ambA2s)[sort_ambA2].reshape(-1, 1), np.asarray(phiAfs_num_ambA2)[sort_ambA2].reshape(-1, 1), np.asarray(phiBfs_num_ambA2)[sort_ambA2].reshape(-1, 1), np.asarray(phiAss_num_ambA2)[sort_ambA2].reshape(-1, 1), np.asarray(phiBss_num_ambA2)[sort_ambA2].reshape(-1, 1)))
np.savetxt('coex_data_ambA2.txt', coex_data_ambA2)

phiAfs_num_ambA3 = []
phiAss_num_ambA3 = []
phiBfs_num_ambA3 = []
phiBss_num_ambA3 = []
ambA3s = []
for file in os.listdir(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA3/'):
	if file[:2] == '1d' in file:
		ambA = float(file.split('_m_0.t')[0].split('_')[-1])
		data = np.loadtxt(f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA3/' + file, delimiter=',')
		phiA = data[:, 1]
		phiB = data[:, 2]
		ambA3s.append(ambA)
		phiA1 = phiA[0]
		phiA2  = phiA[-1]
		phiB1 = phiB[0]
		phiB2 = phiB[-1]
		phiAs = [phiA1, phiA2]
		phiBs = [phiB1, phiB2]
		fi = np.argmin(phiAs)
		si = np.argmax(phiAs)
		phiAfs_num_ambA3.append(phiAs[fi])
		phiAss_num_ambA3.append(phiAs[si])
		phiBfs_num_ambA3.append(phiBs[fi])
		phiBss_num_ambA3.append(phiBs[si])
sort_ambA3 = np.argsort(ambA3s)
coex_data_ambA3 = np.hstack((np.asarray(ambA3s)[sort_ambA3].reshape(-1, 1), np.asarray(phiAfs_num_ambA3)[sort_ambA3].reshape(-1, 1), np.asarray(phiBfs_num_ambA3)[sort_ambA3].reshape(-1, 1), np.asarray(phiAss_num_ambA3)[sort_ambA3].reshape(-1, 1), np.asarray(phiBss_num_ambA3)[sort_ambA3].reshape(-1, 1)))
np.savetxt('coex_data_ambA3.txt', coex_data_ambA3)

phiAfs_num_nr = []
phiAss_num_nr = []
phiBfs_num_nr = []
phiBss_num_nr = []
nrs = []
for file in os.listdir(f'amc_profs_chi05_nrm0_tuned_lams9_tunenr/'):
	if file[:2] == '1d' in file:
		nr = float(file.split('_m_0.t')[0].split('_')[-1])
		data = np.loadtxt(f'amc_profs_chi05_nrm0_tuned_lams9_tunenr/' + file, delimiter=',')
		phiA = data[:, 1]
		phiB = data[:, 2]
		nrs.append(nr)
		phiA1 = phiA[0]
		phiA2  = phiA[-1]
		phiB1 = phiB[0]
		phiB2 = phiB[-1]
		phiAs = [phiA1, phiA2]
		phiBs = [phiB1, phiB2]
		fi = np.argmin(phiAs)
		si = np.argmax(phiAs)
		phiAfs_num_nr.append(phiAs[fi])
		phiAss_num_nr.append(phiAs[si])
		phiBfs_num_nr.append(phiBs[fi])
		phiBss_num_nr.append(phiBs[si])
sort_nr = np.argsort(nrs)
coex_data_nr = np.hstack((np.asarray(nrs)[sort_nr].reshape(-1, 1), np.asarray(phiAfs_num_nr)[sort_nr].reshape(-1, 1), np.asarray(phiBfs_num_nr)[sort_nr].reshape(-1, 1), np.asarray(phiAss_num_nr)[sort_nr].reshape(-1, 1), np.asarray(phiBss_num_nr)[sort_nr].reshape(-1, 1)))
np.savetxt('coex_data_nr.txt', coex_data_nr)
"""

coex_data_nr = np.loadtxt('coex_data_nr.txt')
coex_data_ambA1 = np.loadtxt('coex_data_ambA1.txt')
coex_data_ambA2 = np.loadtxt('coex_data_ambA2.txt')
coex_data_ambA3 = np.loadtxt('coex_data_ambA3.txt')


# Now let's get the predicted binodals, using both the active criteria and the equilibrium maxwell construction
phiA = np.linspace(-1.5, 1.5, 20000)
phiAfs_theory_nr = []
phiAss_theory_nr = []
phiBfs_theory_nr = []
phiBss_theory_nr = []
nrs_theory = []
inds_nr = []
chi = 0.25
alpha = -1.2
for i in range(len(coex_data_nr)):
	nr = coex_data_nr[i, 0]
	phiB = -(chi-nr) * phiA
	muA = alpha * phiA + 4 * phiA**3 + (chi+nr)*phiB
	EA = phiA
	possible_min = argrelextrema(muA, np.less)[0]
	possible_max = argrelextrema(muA, np.greater)[0]
	if len(possible_max) > 0:
		spinodalli = possible_max[0]
		spinodalhi = possible_min[0]
		spinodall = phiA[spinodalli]
		spinodalh = phiA[spinodalhi]
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, EA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_nr.append(phiAf)
		phiAss_theory_nr.append(phiAs)
		phiBfs_theory_nr.append(phiBf)
		phiBss_theory_nr.append(phiBs)
		inds_nr.append(i)
		nrs_theory.append(nr)

coex_theory_nr = np.hstack((np.asarray(inds_nr).reshape(-1, 1), np.asarray(nrs_theory).reshape(-1, 1), np.asarray(phiAfs_theory_nr).reshape(-1, 1), np.asarray(phiBfs_theory_nr).reshape(-1, 1), np.asarray(phiAss_theory_nr).reshape(-1, 1), np.asarray(phiBss_theory_nr).reshape(-1, 1)))
np.savetxt('coex_theory_nr.txt', coex_theory_nr)

plt.figure()
plt.plot(coex_theory_nr[:, 2], (chi + coex_theory_nr[:, 1]) / (chi - coex_theory_nr[:, 1]), lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_nr[:, 4], (chi + coex_theory_nr[:, 1]) / (chi - coex_theory_nr[:, 1]), lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_nr[:, 3], (chi + coex_theory_nr[:, 1]) / (chi - coex_theory_nr[:, 1]), lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_nr[:, 5], (chi + coex_theory_nr[:, 1]) / (chi - coex_theory_nr[:, 1]), lw=2, color='tab:orange', zorder=-100)
plt.scatter(coex_data_nr[:, 1], (chi + coex_data_nr[:, 0]) / (chi - coex_data_nr[:, 0]), facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_nr[:, 3], (chi + coex_data_nr[:, 0]) / (chi - coex_data_nr[:, 0]), facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_nr[:, 2], (chi + coex_data_nr[:, 0]) / (chi - coex_data_nr[:, 0]), facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.scatter(coex_data_nr[:, 4], (chi + coex_data_nr[:, 0]) / (chi - coex_data_nr[:, 0]), facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.yscale('log')
plt.show()



phiA = np.linspace(-1.5, 1.5, 20000)
phiAfs_theory_act_ambA1 = []
phiAss_theory_act_ambA1 = []
phiBfs_theory_act_ambA1 = []
phiBss_theory_act_ambA1 = []
phiAfs_theory_eqm_ambA1 = []
phiAss_theory_eqm_ambA1 = []
phiBfs_theory_eqm_ambA1 = []
phiBss_theory_eqm_ambA1 = []
ambA1_theory = []
inds_ambA1 = []
chi = 0.25
alpha = -1.2
nr = -0.25
K_A = 0.01
for i in range(len(coex_data_ambA1)):
	ambA = coex_data_ambA1[i, 0]
	phiB = -(chi-nr) * phiA
	muA = alpha * phiA + 4 * phiA**3 + (chi+nr)*phiB
	EA = np.exp(2 * ambA * phiA / K_A) # + 2 * ambAAB * phiB / K_A)
	if np.abs(ambA) < 1e-10:
		EA = phiA
	possible_min = argrelextrema(muA, np.less)[0]
	possible_max = argrelextrema(muA, np.greater)[0]
	if len(possible_max) > 0:
		spinodalli = possible_max[0]
		spinodalhi = possible_min[0]
		spinodall = phiA[spinodalli]
		spinodalh = phiA[spinodalhi]
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, EA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_act_ambA1.append(phiAf)
		phiAss_theory_act_ambA1.append(phiAs)
		phiBfs_theory_act_ambA1.append(phiBf)
		phiBss_theory_act_ambA1.append(phiBs)
		inds_ambA1.append(i)
		ambA1_theory.append(ambA)
		# Now get equilibrium predictions
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, phiA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_eqm_ambA1.append(phiAf)
		phiAss_theory_eqm_ambA1.append(phiAs)
		phiBfs_theory_eqm_ambA1.append(phiBf)
		phiBss_theory_eqm_ambA1.append(phiBs)

coex_theory_act_ambA1 = np.hstack((np.asarray(inds_ambA1).reshape(-1, 1), np.asarray(ambA1_theory).reshape(-1, 1), np.asarray(phiAfs_theory_act_ambA1).reshape(-1, 1), np.asarray(phiBfs_theory_act_ambA1).reshape(-1, 1), np.asarray(phiAss_theory_act_ambA1).reshape(-1, 1), np.asarray(phiBss_theory_act_ambA1).reshape(-1, 1)))
np.savetxt('coex_theory_act_ambA1.txt', coex_theory_act_ambA1)
coex_theory_eqm_ambA1 = np.hstack((np.asarray(inds_ambA1).reshape(-1, 1), np.asarray(ambA1_theory).reshape(-1, 1), np.asarray(phiAfs_theory_eqm_ambA1).reshape(-1, 1), np.asarray(phiBfs_theory_eqm_ambA1).reshape(-1, 1), np.asarray(phiAss_theory_eqm_ambA1).reshape(-1, 1), np.asarray(phiBss_theory_eqm_ambA1).reshape(-1, 1)))
np.savetxt('coex_theory_eqm_ambA1.txt', coex_theory_eqm_ambA1)

plt.figure()
plt.plot(coex_theory_act_ambA1[:, 2], coex_theory_act_ambA1[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA1[:, 4], coex_theory_act_ambA1[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA1[:, 3], coex_theory_act_ambA1[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_act_ambA1[:, 5], coex_theory_act_ambA1[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_eqm_ambA1[:, 2], coex_theory_eqm_ambA1[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA1[:, 4], coex_theory_eqm_ambA1[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA1[:, 3], coex_theory_eqm_ambA1[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA1[:, 5], coex_theory_eqm_ambA1[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.scatter(coex_data_ambA1[:, 1], coex_data_ambA1[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA1[:, 3], coex_data_ambA1[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA1[:, 2], coex_data_ambA1[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.scatter(coex_data_ambA1[:, 4], coex_data_ambA1[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.yscale('log')
plt.show()



phiA = np.linspace(-1.5, 1.5, 20000)
phiAfs_theory_act_ambA2 = []
phiAss_theory_act_ambA2 = []
phiBfs_theory_act_ambA2 = []
phiBss_theory_act_ambA2 = []
phiAfs_theory_eqm_ambA2 = []
phiAss_theory_eqm_ambA2 = []
phiBfs_theory_eqm_ambA2 = []
phiBss_theory_eqm_ambA2 = []
ambA2_theory = []
inds_ambA2 = []
chi = 0.25
alpha = -1.2
nr = 0.0
K_A = 0.01
for i in range(len(coex_data_ambA2)):
	ambA = coex_data_ambA2[i, 0]
	phiB = -(chi-nr) * phiA
	muA = alpha * phiA + 4 * phiA**3 + (chi+nr)*phiB
	EA = np.exp(2 * ambA * phiA / K_A) # + 2 * ambAAB * phiB / K_A)
	if np.abs(ambA) < 1e-10:
		EA = phiA
	possible_min = argrelextrema(muA, np.less)[0]
	possible_max = argrelextrema(muA, np.greater)[0]
	if len(possible_max) > 0:
		spinodalli = possible_max[0]
		spinodalhi = possible_min[0]
		spinodall = phiA[spinodalli]
		spinodalh = phiA[spinodalhi]
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, EA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_act_ambA2.append(phiAf)
		phiAss_theory_act_ambA2.append(phiAs)
		phiBfs_theory_act_ambA2.append(phiBf)
		phiBss_theory_act_ambA2.append(phiBs)
		inds_ambA2.append(i)
		ambA2_theory.append(ambA)
		# Now get equilibrium predictions
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, phiA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_eqm_ambA2.append(phiAf)
		phiAss_theory_eqm_ambA2.append(phiAs)
		phiBfs_theory_eqm_ambA2.append(phiBf)
		phiBss_theory_eqm_ambA2.append(phiBs)

coex_theory_act_ambA2 = np.hstack((np.asarray(inds_ambA2).reshape(-1, 1), np.asarray(ambA2_theory).reshape(-1, 1), np.asarray(phiAfs_theory_act_ambA2).reshape(-1, 1), np.asarray(phiBfs_theory_act_ambA2).reshape(-1, 1), np.asarray(phiAss_theory_act_ambA2).reshape(-1, 1), np.asarray(phiBss_theory_act_ambA2).reshape(-1, 1)))
np.savetxt('coex_theory_act_ambA2.txt', coex_theory_act_ambA2)
coex_theory_eqm_ambA2 = np.hstack((np.asarray(inds_ambA2).reshape(-1, 1), np.asarray(ambA2_theory).reshape(-1, 1), np.asarray(phiAfs_theory_eqm_ambA2).reshape(-1, 1), np.asarray(phiBfs_theory_eqm_ambA2).reshape(-1, 1), np.asarray(phiAss_theory_eqm_ambA2).reshape(-1, 1), np.asarray(phiBss_theory_eqm_ambA2).reshape(-1, 1)))
np.savetxt('coex_theory_eqm_ambA2.txt', coex_theory_eqm_ambA2)

plt.figure()
plt.plot(coex_theory_act_ambA2[:, 2], coex_theory_act_ambA2[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA2[:, 4], coex_theory_act_ambA2[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA2[:, 3], coex_theory_act_ambA2[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_act_ambA2[:, 5], coex_theory_act_ambA2[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_eqm_ambA2[:, 2], coex_theory_eqm_ambA2[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA2[:, 4], coex_theory_eqm_ambA2[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA2[:, 3], coex_theory_eqm_ambA2[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA2[:, 5], coex_theory_eqm_ambA2[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.scatter(coex_data_ambA2[:, 1], coex_data_ambA2[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA2[:, 3], coex_data_ambA2[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA2[:, 2], coex_data_ambA2[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.scatter(coex_data_ambA2[:, 4], coex_data_ambA2[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.yscale('log')
plt.show()



phiA = np.linspace(-1.5, 1.5, 20000)
phiAfs_theory_act_ambA3 = []
phiAss_theory_act_ambA3 = []
phiBfs_theory_act_ambA3 = []
phiBss_theory_act_ambA3 = []
phiAfs_theory_eqm_ambA3 = []
phiAss_theory_eqm_ambA3 = []
phiBfs_theory_eqm_ambA3 = []
phiBss_theory_eqm_ambA3 = []
ambA3_theory = []
inds_ambA3 = []
chi = 0.25
alpha = -1.2
nr = 0.0
K_A = 0.01
ambAAB = 0.01
ambABB = 0.01
K_AB = 0.01 
# K_AB = K_AA ambABB / ambAAB
# ambAAB = ambABB
for i in range(len(coex_data_ambA3)):
	ambA = coex_data_ambA3[i, 0]
	phiB = -(chi-nr) * phiA
	muA = alpha * phiA + 4 * phiA**3 + (chi+nr)*phiB
	EA = np.exp(2 * ambA * phiA / K_A + 2 * ambAAB * phiB / K_A)
	# if np.abs(ambA) < 1e-10:
	# 	EA = phiA
	possible_min = argrelextrema(muA, np.less)[0]
	possible_max = argrelextrema(muA, np.greater)[0]
	if len(possible_max) > 0:
		spinodalli = possible_max[0]
		spinodalhi = possible_min[0]
		spinodall = phiA[spinodalli]
		spinodalh = phiA[spinodalhi]
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, EA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_act_ambA3.append(phiAf)
		phiAss_theory_act_ambA3.append(phiAs)
		phiBfs_theory_act_ambA3.append(phiBf)
		phiBss_theory_act_ambA3.append(phiBs)
		inds_ambA3.append(i)
		ambA3_theory.append(ambA)
		# Now get equilibrium predictions
		phiA1, phiA2, muA_coex, best = get_coex(muA, phiA, spinodall, spinodalh, phiA)
		phiAf = min([phiA1, phiA2])
		phiAs = max([phiA1, phiA2])
		phiBf = -(chi - nr) * phiAf
		phiBs = -(chi - nr) * phiAs
		phiAfs_theory_eqm_ambA3.append(phiAf)
		phiAss_theory_eqm_ambA3.append(phiAs)
		phiBfs_theory_eqm_ambA3.append(phiBf)
		phiBss_theory_eqm_ambA3.append(phiBs)

coex_theory_act_ambA3 = np.hstack((np.asarray(inds_ambA3).reshape(-1, 1), np.asarray(ambA3_theory).reshape(-1, 1), np.asarray(phiAfs_theory_act_ambA3).reshape(-1, 1), np.asarray(phiBfs_theory_act_ambA3).reshape(-1, 1), np.asarray(phiAss_theory_act_ambA3).reshape(-1, 1), np.asarray(phiBss_theory_act_ambA3).reshape(-1, 1)))
np.savetxt('coex_theory_act_ambA3.txt', coex_theory_act_ambA3)
coex_theory_eqm_ambA3 = np.hstack((np.asarray(inds_ambA3).reshape(-1, 1), np.asarray(ambA3_theory).reshape(-1, 1), np.asarray(phiAfs_theory_eqm_ambA3).reshape(-1, 1), np.asarray(phiBfs_theory_eqm_ambA3).reshape(-1, 1), np.asarray(phiAss_theory_eqm_ambA3).reshape(-1, 1), np.asarray(phiBss_theory_eqm_ambA3).reshape(-1, 1)))
np.savetxt('coex_theory_eqm_ambA3.txt', coex_theory_eqm_ambA3)

plt.figure()
plt.plot(coex_theory_act_ambA3[:, 2], coex_theory_act_ambA3[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA3[:, 4], coex_theory_act_ambA3[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100)
plt.plot(coex_theory_act_ambA3[:, 3], coex_theory_act_ambA3[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_act_ambA3[:, 5], coex_theory_act_ambA3[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100)
plt.plot(coex_theory_eqm_ambA3[:, 2], coex_theory_eqm_ambA3[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA3[:, 4], coex_theory_eqm_ambA3[:, 1] / K_A, lw=2, color='tab:blue', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA3[:, 3], coex_theory_eqm_ambA3[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.plot(coex_theory_eqm_ambA3[:, 5], coex_theory_eqm_ambA3[:, 1] / K_A, lw=2, color='tab:orange', zorder=-100, ls='--')
plt.scatter(coex_data_ambA3[:, 1], coex_data_ambA3[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA3[:, 3], coex_data_ambA3[:, 0] / K_A, facecolors='none', marker='s', edgecolors='tab:blue', lw=2)
plt.scatter(coex_data_ambA3[:, 2], coex_data_ambA3[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.scatter(coex_data_ambA3[:, 4], coex_data_ambA3[:, 0] / K_A, facecolors='none', marker='^', edgecolors='tab:orange', lw=2)
plt.yscale('log')
plt.show()

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = "Times New Roman"


def plot_pd(filename, figsize, data, theory_act, K_A_or_chi=1, tune_ambA=True, theory_eqm=None, lwm=2, lwl=2, show=False, legend=False):
	# theory_act = theory_act[1:, :]
	# data = data[1,:]
	# theory_act[theory_act[:, 1] > 0.0001]
	# data[data[:, 0] > 0.0001]
	if theory_eqm is not None:
		# theory_eqm = theory_eqm[1:, :]
		theory_eqm = theory_eqm[theory_eqm[:, 1] > 0.0001]
	if tune_ambA:
		plt.figure(figsize=figsize)
		p1=plt.plot(theory_act[:, 2], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100)[0]
		plt.plot(theory_act[:, 4], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100)
		p2=plt.plot(theory_act[:, 3], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100)[0]
		plt.plot(theory_act[:, 5], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100)
		if theory_eqm is not None:
			p3=plt.plot(theory_eqm[:, 2], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100, ls='--')[0]
			plt.plot(theory_eqm[:, 4], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100, ls='--')
			p4=plt.plot(theory_eqm[:, 3], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100, ls='--')[0]
			plt.plot(theory_eqm[:, 5], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100, ls='--')
		p5=plt.scatter(data[:, 1], data[:, 0] / K_A_or_chi, facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
		plt.scatter(data[:, 3], data[:, 0] / K_A_or_chi, facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
		p6=plt.scatter(data[:, 2], data[:, 0] / K_A_or_chi, facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
		plt.scatter(data[:, 4], data[:, 0] / K_A_or_chi, facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
		plt.yscale('log')
		if legend:
			if theory_eqm is not None:
				plt.legend([p5, p6, p1, p2, p3, p4], [r'$\rho$', r'$\psi$', 'theory rho', 'theory psi', 'eqm theory rho', 'eqm theory psi'])
			else:
				plt.legend([p5, p6, p1, p2], [r'$\rho$', r'$\psi$', 'theory rho', 'theory psi'])
		plt.ylabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}$', fontsize=14)
		ax = plt.gca()
		ax.tick_params(width=2, length=10, which='major', labelsize=14)
		ax.tick_params(width=2, length=6, which='minor', labelsize=14)
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(2.)
		plt.tight_layout()
		if show:
			plt.show()
		else:
			plt.savefig(filename, format='svg', dpi=1200)
			plt.close()
	else:
		theory_act = theory_act[1:-1, :]
		data = data[1:-1,:]
		print((K_A_or_chi + data[:, 0]) / (K_A_or_chi - data[:, 0]))
		# theory_act[theory_act[:, 1] > 0.0001]
		# data[data[:, 0] > 0.0001]
		plt.figure(figsize=figsize)
		p1=plt.plot(theory_act[:, 2], (K_A_or_chi + theory_act[:, 1]) / (K_A_or_chi - theory_act[:, 1]), lw=lwl, color='tab:blue', zorder=-100)[0]
		plt.plot(theory_act[:, 4], (K_A_or_chi + theory_act[:, 1]) / (K_A_or_chi - theory_act[:, 1]), lw=lwl, color='tab:blue', zorder=-100)
		p2=plt.plot(theory_act[:, 3], (K_A_or_chi + theory_act[:, 1]) / (K_A_or_chi - theory_act[:, 1]), lw=lwl, color='tab:orange', zorder=-100)[0]
		plt.plot(theory_act[:, 5], (K_A_or_chi + theory_act[:, 1]) / (K_A_or_chi - theory_act[:, 1]), lw=lwl, color='tab:orange', zorder=-100)
		p5=plt.scatter(data[:, 1], (K_A_or_chi + data[:, 0]) / (K_A_or_chi - data[:, 0]), facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
		plt.scatter(data[:, 3], (K_A_or_chi + data[:, 0]) / (K_A_or_chi - data[:, 0]), facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
		p6=plt.scatter(data[:, 2], (K_A_or_chi + data[:, 0]) / (K_A_or_chi - data[:, 0]), facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
		plt.scatter(data[:, 4], (K_A_or_chi + data[:, 0]) / (K_A_or_chi - data[:, 0]), facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
		plt.yscale('log')
		if legend:
			plt.legend([p5, p6, p1, p2], [r'$\rho$', r'$\psi$', 'theory rho', 'theory psi'])
		plt.ylabel(r'$\chi_{\rho \psi} / \chi_{\psi \rho}$', fontsize=14)
		ax = plt.gca()
		ax.tick_params(width=2, length=10, which='major', labelsize=14)
		ax.tick_params(width=2, length=6, which='minor', labelsize=14)
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(2.)
		plt.tight_layout()
		if show:
			plt.show()
		else:
			plt.savefig(filename, format='svg', dpi=1200)
			plt.close()

def ax_plot_pd(ax, data, theory_act, K_A_or_chi=1, theory_eqm=None, lwm=2, lwl=2):
	# theory_act[theory_act[:, 1] > 0.0001]
	# data[data[:, 0] > 0.0001]
	# if theory_eqm is not None:
	# 	# theory_eqm = theory_eqm[1:, :]
	# 	theory_eqm = theory_eqm[theory_eqm[:, 1] > 0.0001]
	ax.plot(theory_act[:, 2], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100)[0]
	ax.plot(theory_act[:, 4], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100)
	ax.plot(theory_act[:, 3], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100)[0]
	ax.plot(theory_act[:, 5], theory_act[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100)
	ax.plot(theory_eqm[:, 2], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100, ls='--')[0]
	ax.plot(theory_eqm[:, 4], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:blue', zorder=-100, ls='--')
	ax.plot(theory_eqm[:, 3], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100, ls='--')[0]
	ax.plot(theory_eqm[:, 5], theory_eqm[:, 1] / K_A_or_chi, lw=lwl, color='tab:orange', zorder=-100, ls='--')
	ax.scatter(data[:, 1], data[:, 0] / K_A_or_chi, facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
	ax.scatter(data[:, 3], data[:, 0] / K_A_or_chi, facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
	ax.scatter(data[:, 2], data[:, 0] / K_A_or_chi, facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
	ax.scatter(data[:, 4], data[:, 0] / K_A_or_chi, facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
	ax.tick_params(width=2, length=10, which='major', labelsize=14)
	ax.tick_params(width=2, length=6, which='minor', labelsize=14)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.)


def ax_plot_error(ax, error_act, error_eqm, K_A, color, lwm=2):
	# ax.scatter(error_act[:, 0] / K_A, error_act[:, 1], color='tab:blue', marker='s')
	# ax.scatter(error_act[:, 0] / K_A, error_act[:, 3], color='tab:blue', marker='s')
	# ax.scatter(error_act[:, 0] / K_A, error_act[:, 2], color='tab:orange', marker='^')
	# ax.scatter(error_act[:, 0] / K_A, error_act[:, 4], color='tab:orange', marker='^')
	ax.scatter(error_act[:, 0] / K_A, error_act[:, 1], facecolors='none', marker='s', edgecolors='tab:blue', lw=lwm)
	ax.scatter(error_act[:, 0] / K_A, error_act[:, 3], facecolors='none', marker='D', edgecolors='tab:blue', lw=lwm)
	ax.scatter(error_act[:, 0] / K_A, error_act[:, 2], facecolors='none', marker='^', edgecolors='tab:orange', lw=lwm)
	ax.scatter(error_act[:, 0] / K_A, error_act[:, 4], facecolors='none', marker='v', edgecolors='tab:orange', lw=lwm)
	ax.tick_params(width=2, length=10, which='major', labelsize=14)
	ax.tick_params(width=2, length=6, which='minor', labelsize=14)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.)




files = [f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA_const_ambAKAA/1d_coexistence_profiles_2param_ambA_0.005_m_0.txt',
		 f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA_const_ambAKAA/1d_coexistence_profiles_2param_ambA_0.010069_m_0.txt',
		 f'amc_profs_chi05_nrm0_tuned_lams9_tuneambA_const_ambAKAA/1d_coexistence_profiles_2param_ambA_0.02_m_0.txt']

plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.75))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(files)):
	file = files[i]
	data = np.loadtxt(file, delimiter=',')
	z = data[:, 0]
	phiA = data[:, 1]
	phiB = data[:, 2]
	plt.plot(z, phiA, lw=2, color=colors[i])
	plt.plot(z, phiB, lw=2, ls='--', color=colors[i])
plt.xlabel(r'$z$', fontsize=14)
ax = plt.gca()
ax.tick_params(width=2, length=10, which='major', labelsize=14)
ax.tick_params(width=2, length=6, which='minor', labelsize=14)
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.)
plt.tight_layout()
# plt.show()
plt.savefig('amc_prof_supplement.svg', format='svg', dpi=1200)
plt.close()
# assert False

coex_data_ambA1 = np.loadtxt('coex_data_ambA1.txt')
coex_theory_act_ambA1 = np.loadtxt('coex_theory_act_ambA1.txt')
coex_theory_eqm_ambA1 = np.loadtxt('coex_theory_eqm_ambA1.txt')
coex_data_ambA2 = np.loadtxt('coex_data_ambA2.txt')
coex_theory_act_ambA2 = np.loadtxt('coex_theory_act_ambA2.txt')
coex_theory_eqm_ambA2 = np.loadtxt('coex_theory_eqm_ambA2.txt')
coex_data_ambA3 = np.loadtxt('coex_data_ambA3.txt')
coex_theory_act_ambA3 = np.loadtxt('coex_theory_act_ambA3.txt')
coex_theory_eqm_ambA3 = np.loadtxt('coex_theory_eqm_ambA3.txt')
coex_data_nr = np.loadtxt('coex_data_nr.txt')
coex_theory_nr = np.loadtxt('coex_theory_nr.txt')

K_A = 0.01
chi = 0.25


nrs = []
Gbars = []
for file in os.listdir(f'tune_nr/amc_profs_chi05_nrm0_tuned_lams9_tunenr/'):
	if file[:2] == '1d' in file:
		nr = float(file.split('_m_0.t')[0].split('_')[-1])
		if nr != 0.25:
			print(nr)
			rat = (chi + nr) / (chi - nr)
			data = np.loadtxt(f'tune_nr/amc_profs_chi05_nrm0_tuned_lams9_tunenr/' + file, delimiter=',')
			z = data[:, 0]
			L = max(z) - min(z)
			phiA = data[:, 1]
			phiB = data[:, 2]
			nrs.append(nr)
			G1 = (rat - 2) / L * np.sum((phiB + (chi - nr) * phiA)[:-1] * np.diff(phiB))
			G2 = (rat - 2)**2 / L * np.sum((phiB + (chi - nr) * phiA)[:-1]**2 * np.diff(phiB)**2 / np.diff(z))
			Gbars.append(G1 / np.sqrt(G2 - G1**2))

nrs = np.asarray(nrs)

plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.75))
plt.scatter((chi + nrs) / (chi - nrs), Gbars, edgecolors='tab:blue', marker='s', facecolors='none', lw=2)
plt.xlabel(r'$\chi_{\rho \psi} / \chi_{\psi \rho}$', fontsize=14)
plt.ylabel(r'$\Delta^{\alpha \beta} \overline{G}^{\rm bulk}$', fontsize=14)
plt.ylim([-0.045, 0.04])
ax = plt.gca()
ax.tick_params(width=2, length=10, which='major', labelsize=14)
ax.tick_params(width=2, length=6, which='minor', labelsize=14)
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.)
plt.xscale('log')
plt.tight_layout()
plt.savefig('nrpd_supplement.svg', format='svg', dpi=1200)
plt.close()
# plt.show()


# assert False
		


coex_data_ambA1 = coex_data_ambA1[coex_data_ambA1[:, 0] <= 0.05]
coex_theory_act_ambA1 = coex_theory_act_ambA1[coex_theory_act_ambA1[:, 1] <= 0.05]
coex_theory_eqm_ambA1 = coex_theory_eqm_ambA1[coex_theory_eqm_ambA1[:, 1] <= 0.05]

coex_data_ambA2 = coex_data_ambA2[coex_data_ambA2[:, 0] <= 0.05]
coex_theory_act_ambA2 = coex_theory_act_ambA2[coex_theory_act_ambA2[:, 1] <= 0.05]
coex_theory_eqm_ambA2 = coex_theory_eqm_ambA2[coex_theory_eqm_ambA2[:, 1] <= 0.05]

coex_data_ambA3 = coex_data_ambA3[coex_data_ambA3[:, 0] <= 0.05]
coex_theory_act_ambA3 = coex_theory_act_ambA3[coex_theory_act_ambA3[:, 1] <= 0.05]
coex_theory_eqm_ambA3 = coex_theory_eqm_ambA3[coex_theory_eqm_ambA3[:, 1] <= 0.05]


# # plot_pd('ambA1pd.svg', (4, 4.8), coex_data_ambA1, coex_theory_act_ambA1, K_A, True, coex_theory_eqm_ambA1, 1.5, 1.5, True, False)
# # plot_pd('ambA2pd.svg', (4, 4.8), coex_data_ambA2, coex_theory_act_ambA2, K_A, True, coex_theory_eqm_ambA2, 1.5, 1.5, True, False)
# # plot_pd('ambA3pd.svg', (4, 4.8), coex_data_ambA3, coex_theory_act_ambA3, K_A, True, coex_theory_eqm_ambA3, 1.5, 1.5, True, False)
# plot_pd('nrpd2.svg', (4.8 * 1, 3.6 * 1), coex_data_nr, coex_theory_nr, chi, False, None, 1.5, 1.5, True, False)

# fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9, 2.7))
# ax_plot_pd(axs[0], coex_data_ambA1, coex_theory_act_ambA1, K_A, coex_theory_eqm_ambA1, 1.5, 1.5)
# ax_plot_pd(axs[1], coex_data_ambA2, coex_theory_act_ambA2, K_A, coex_theory_eqm_ambA2, 1.5, 1.5)
# ax_plot_pd(axs[2], coex_data_ambA3, coex_theory_act_ambA3, K_A, coex_theory_eqm_ambA3, 1.5, 1.5)
# axs[0].set_ylabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}$', fontsize=14)
# # plt.ylabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}}$', fontsize=14)
# plt.tight_layout()
# plt.show()
# # plt.savefig('ambA_pd_subplot2.svg', format='svg', dpi=1200)
# # plt.close()

pct_error_act1 = np.hstack((coex_data_ambA1[:, 0].reshape(-1, 1), 100 * (coex_theory_act_ambA1[:, 2:] - coex_data_ambA1[:, 1:]) / coex_data_ambA1[:, 1:]))
pct_error_act2 = np.hstack((coex_data_ambA2[:, 0].reshape(-1, 1), 100 * (coex_theory_act_ambA2[:, 2:] - coex_data_ambA2[:, 1:]) / coex_data_ambA2[:, 1:]))
pct_error_act3 = np.hstack((coex_data_ambA3[:, 0].reshape(-1, 1), 100 * (coex_theory_act_ambA3[:, 2:] - coex_data_ambA3[:, 1:]) / coex_data_ambA3[:, 1:]))
pct_error_eqm1 = np.hstack((coex_data_ambA1[:, 0].reshape(-1, 1), 100 * (coex_theory_eqm_ambA1[:, 2:] - coex_data_ambA1[:, 1:]) / coex_data_ambA1[:, 1:]))
pct_error_eqm2 = np.hstack((coex_data_ambA2[:, 0].reshape(-1, 1), 100 * (coex_theory_eqm_ambA2[:, 2:] - coex_data_ambA2[:, 1:]) / coex_data_ambA2[:, 1:]))
pct_error_eqm3 = np.hstack((coex_data_ambA3[:, 0].reshape(-1, 1), 100 * (coex_theory_eqm_ambA3[:, 2:] - coex_data_ambA3[:, 1:]) / coex_data_ambA3[:, 1:]))

# print()

# print(coex_data_ambA1[:, 0])
# print(coex_data_ambA1[:, 1])
# print()
# print(coex_data_ambA1[0, :])
# print(coex_theory_act_ambA1[0, :])
# print(coex_theory_act_ambA1[:, 2:])
# print(coex_theory_act_ambA1[:, 2:])

# assert False

fig, axs = plt.subplots(2, 3, sharex=True, figsize=(10, 4), sharey='row')
ax_plot_error(axs[0, 0], pct_error_eqm1, pct_error_eqm1, K_A, 1.5)
ax_plot_error(axs[1, 0], pct_error_act1, pct_error_eqm1, K_A, 1.5)
ax_plot_error(axs[0, 1], pct_error_eqm2, pct_error_eqm2, K_A, 1.5)
ax_plot_error(axs[1, 1], pct_error_act2, pct_error_eqm2, K_A, 1.5)
ax_plot_error(axs[0, 2], pct_error_eqm3, pct_error_eqm3, K_A, 1.5)
ax_plot_error(axs[1, 2], pct_error_act3, pct_error_eqm3, K_A, 1.5)
axs[1, 0].set_yticks([0, 1, 2])
# axs[1, 0].set_xticks([0, 1, 2, 3, 4, 5])
# axs[1, 0].set_xlabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}}$', fontsize=14)
# axs[1, 1].set_xlabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}}$', fontsize=14)
# axs[1, 2].set_xlabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}}$', fontsize=14)
# axs[0].scatter(coex_data_ambA1[:, 0], 100 * (coex_theory_act_ambA1[:, 2:] - coex_data_ambA1[:, 1:]) / coex_data_ambA1[:, 1:])
# axs[1].scatter(coex_data_ambA2[:, 0], 100 * (coex_theory_act_ambA2[:, 2:] - coex_data_ambA2[:, 1:]) / coex_data_ambA2[:, 1:])
# axs[2].scatter(coex_data_ambA3[:, 0], 100 * (coex_theory_act_ambA3[:, 2:] - coex_data_ambA3[:, 1:]) / coex_data_ambA3[:, 1:])
# plt.ylabel(r'$\lambda_{\rho \rho \rho} / K_{\rho \rho}}$', fontsize=14)
plt.tight_layout()
# plt.show()
plt.savefig('ambA_error_subplot.svg', format='svg', dpi=1200)
plt.close()



chiBA = 0.25
alpha = -1.2

# ambAs = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
ps = []
ambAs = []
# plt.figure()
g_diffs = []
rmsds = []
for file in os.listdir(f'tune_ambA/amc_profs_chi05_nrm0_tuned_lams9_tuneambA2/'):
	if file[:2] == '1d' in file:
		ambA = float(file.split('_m_0.t')[0].split('_')[-1])
		# nr = float(file.split('_m_0.t')[0].split('_')[-1])

		data = np.loadtxt(f'tune_ambA/amc_profs_chi05_nrm0_tuned_lams9_tuneambA2/' + file, delimiter=',')
		# ambA = ambAs[i]
		# data = np.loadtxt(f'tune_ambA/amc_profs_chi05_nrm0_tuned_lams9_tuneambA2/1d_coexistence_profiles_2param_ambA_{str(ambA)}_m_0.txt', delimiter=',')
		phiA = data[:, 1]
		phiB = data[:, 2]
		ambA_over_KAA = ambA / 0.01
		if ambA < 1e-10:
			EA = phiA
		else:
			EA = np.exp(2 * ambA_over_KAA * phiA)
		muA = alpha * phiA + 4 * phiA**3 + chiBA * phiB
		muA_approx = alpha * phiA + 4 * phiA**3 - chiBA**2 * phiA
		g = np.hstack((0, np.cumsum(EA[1:] * np.diff(muA))))
		g_approx = np.hstack((0, np.cumsum(EA[1:] * np.diff(muA_approx))))
		avg_max = 0.5 * (np.max(np.abs(g))*2 + 0 * np.max(np.abs(g_approx)))
		rmsds.append(np.sqrt(np.mean((g - g_approx)**2)) / avg_max)
		# print(np.abs(g_approx[-1] - g[-1]) / avg_max)
		# print()
		g_diffs.append(np.abs(g_approx[-1] - g[-1]) / avg_max)
		ambAs.append(ambA)
		# plt.figure()
		# plt.plot(phiA, g, ls='-.')
		# plt.plot(phiA, g_approx, ls='--')
		# plt.show()
	# 	ps.append(plt.plot(phiA[1:], g, ls='-.', color=colors[i])[0])
	# 	plt.plot(phiA[1:], g_approx, ls='--', color=colors[i])
	# plt.legend(ps, ambAs)
	# plt.show()

ambAs = np.asarray(ambAs)
g_diffs = np.asarray(g_diffs)
rmsds = np.asarray(rmsds)

sort = np.argsort(ambAs)

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = "Times New Roman"

plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.75))
plt.scatter(ambAs[sort][1:] / 0.01, 100 * g_diffs[sort][1:], edgecolors='tab:blue', marker='s', facecolors='none', lw=2)
plt.scatter(ambAs[sort][1:] / 0.01, 100 * rmsds[sort][1:], edgecolors='tab:orange', marker='^', facecolors='none', lw=2)
plt.xlabel(r'$\lambda_{AAA} / K_{AA}$', fontsize=14, fontname='Times New Roman')
plt.ylabel('Error (%)', fontsize=14)
plt.legend([r'$\sqrt{{\rm SD}_{n_z}}$', r'$RMSD$'])
ax = plt.gca()
ax.tick_params(width=2, length=10, which='major', labelsize=14)
ax.tick_params(width=2, length=6, which='minor', labelsize=14)
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(2.)

# fig, axs = plt.subplots(2, 1, figsize=(4.8, 6.4))
# axs[0].scatter(ambAs[sort][1:] / 0.01, 100 * g_diffs[sort][1:], edgecolors='tab:blue', facecolors='none', lw=2)
# # axs[0].set_xlabel(r'$\lambda_{AAA} / K_{AA}$', fontsize=18, fontname='Times New Roman')
# axs[0].set_ylabel(r'${\rm Relative} \ \sqrt{{\rm SD}_{n_z}} \ (\%)$', fontsize=18, fontname='Times New Roman')
# axs[0].tick_params(width=2, length=10, which='major', labelsize=14)
# axs[0].tick_params(width=2, length=6, which='minor', labelsize=14)
# for axis in ['top','bottom','left','right']:
# 	axs[0].spines[axis].set_linewidth(2.)
# axs[1].scatter(ambAs[sort][1:] / 0.01, 100 * rmsds[sort][1:], edgecolors='tab:blue', facecolors='none', lw=2)
# axs[1].set_xlabel(r'$\lambda_{AAA} / K_{AA}$', fontsize=18, fontname='Times New Roman')
# axs[1].set_ylabel(r'${\rm Relative \ RMSD} \ (\%)$', fontsize=18, fontname='Times New Roman')
# axs[1].tick_params(width=2, length=10, which='major', labelsize=14)
# axs[1].tick_params(width=2, length=6, which='minor', labelsize=14)
# for axis in ['top','bottom','left','right']:
# 	axs[1].spines[axis].set_linewidth(2.)
plt.tight_layout()
# plt.show()
plt.savefig('g_int_subplot.svg', format='svg', dpi=1200)
plt.close()


