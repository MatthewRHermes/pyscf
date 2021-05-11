#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import fci
from pyscf import mcscf

b = 1.4
mol = gto.M(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': '631g', },
)
m = scf.RHF(mol)
m.conv_tol = 1e-10
m.scf()
mc0 = mcscf.CASSCF(m, 4, 4).run()

molsym = gto.M(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': '631g', },
symmetry = True
)
msym = scf.RHF(molsym)
msym.conv_tol = 1e-10
msym.scf()

def tearDownModule():
    global mol, molsym, m, msym, mc0
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym, mc0


class KnownValues(unittest.TestCase):
    def test_with_x2c_scanner(self):
        mc1 = mcscf.CASSCF(m, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1 = mcscf.CASSCF(m, 4, 4).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1('N 0 0 0; N 0 0 1.1')
        self.assertAlmostEqual(mc1.e_tot, -109.02535605303684, 7)

    def test_mc1step_symm_with_x2c_scanner(self):
        mc1 = mcscf.CASSCF(msym, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1 = mcscf.CASSCF(msym, 4, 4).x2c().as_scanner().as_scanner()
        mc1(molsym)
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1('N 0 0 0; N 0 0 1.1')
        self.assertAlmostEqual(mc1.e_tot, -109.02535605303684, 7)

    def test_0core_0virtual(self):
        mol = gto.M(atom='He', basis='321g')
        mf = scf.RHF(mol).run()
        mc1 = mcscf.CASSCF(mf, 2, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.CASSCF(mf, 1, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 1, 0).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 2, 2)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.CASSCF(mf, 1, 2)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 1, 0)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

    def test_cas_natorb(self):
        mc1 = mcscf.CASSCF(msym, 4, 4, ncore=5)
        mo = mc1.sort_mo([4,5,10,13])
        mc1.sorting_mo_energy = True
        mc1.kernel(mo)
        mo0 = mc1.mo_coeff
        ci0 = mc1.ci
        self.assertAlmostEqual(mc1.e_tot, -108.7288793597413, 8)
        casdm1 = mc1.fcisolver.make_rdm1(mc1.ci, 4, 4)
        mc1.ci = None  # Force cas_natorb_ to recompute CI coefficients

        mc1.cas_natorb_(casdm1=casdm1, eris=mc1.ao2mo())
        mo1 = mc1.mo_coeff
        ci1 = mc1.ci
        s = numpy.einsum('pi,pq,qj->ij', mo0[:,5:9], msym.get_ovlp(), mo1[:,5:9])
        self.assertAlmostEqual(fci.addons.overlap(ci0, ci1, 4, 4, s), 1, 9)

    def test_get_h2eff(self):
        mc1 = mcscf.CASSCF(m, 4, 4).approx_hessian()
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertAlmostEqual(abs(eri1-eri2).max(), 0, 12)

        mc1 = mcscf.density_fit(mcscf.CASSCF(m, 4, 4))
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertTrue(abs(eri1-eri2).max() > 1e-5)

    def test_get_veff(self):
        mf = m.view(dft.rks.RKS)
        mc1 = mcscf.CASSCF(mf, 4, 4)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        veff1 = mc1.get_veff(mol, dm)
        veff2 = m.get_veff(mol, dm)
        self.assertAlmostEqual(abs(veff1-veff2).max(), 0, 12)

    def test_state_average(self):
        mc1 = mcscf.CASSCF(m, 4, 4).state_average_((0.5,0.5))
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(numpy.dot(mc1.e_states, [.5,.5]), -108.80445340617777, 8)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(lib.finger(mo_occ), 1.8748844779923917, 4)
        dm1 = mc1.analyze()
        self.assertAlmostEqual(lib.finger(dm1[0]), 2.6993157521103779, 4)
        self.assertAlmostEqual(lib.finger(dm1[1]), 2.6993157521103779, 4)

    def test_natorb(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.natorb = True
        mc1.conv_tol = 1e-10
        mc1.kernel(mo)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(mc1.e_tot, -105.83025103050596, 9)
        self.assertAlmostEqual(lib.finger(mo_occ), 2.4188178285392317, 4)

        mc1.mc2step(mo)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(mc1.e_tot, -105.83025103050596, 9)
        self.assertAlmostEqual(lib.finger(mo_occ), 2.418822007439851, 4)

    def test_dep4(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.with_dep4 = True
        mc1.max_cycle = 1
        mc1.max_cycle_micro = 6
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.8292690292608, 8)

    def test_dep4_df(self):
        mc1 = mcscf.CASSCF(msym, 4, 4).density_fit()
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.with_dep4 = True
        mc1.max_cycle = 1
        mc1.max_cycle_micro = 6
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82923271851176, 8)

    # FIXME: How to test ci_response_space? The test below seems numerical instable
    #def test_ci_response_space(self):
    #    mc1 = mcscf.CASSCF(m, 4, 4)
    #    mc1.ci_response_space = 9
    #    mc1.max_cycle = 1
    #    mc1.max_cycle_micro = 2
    #    mc1.kernel()
    #    self.assertAlmostEqual(mc1.e_tot, -108.85920100433893, 8)

    #    mc1 = mcscf.CASSCF(m, 4, 4)
    #    mc1.ci_response_space = 1
    #    mc1.max_cycle = 1
    #    mc1.max_cycle_micro = 2
    #    mc1.kernel()
    #    self.assertAlmostEqual(mc1.e_tot, -108.85920400781617, 8)

    def test_chk(self):
        mc2 = mcscf.CASSCF(m, 4, 4)
        mc2.update(mc0.chkfile)
        mc2.max_cycle = 0
        mc2.kernel()
        self.assertAlmostEqual(mc0.e_tot, mc2.e_tot, 8)

    def test_grad(self):
        self.assertAlmostEqual(abs(mc0.get_grad()).max(), 0, 4)

    def test_external_fcisolver(self):
        fcisolver1 = fci.direct_spin1.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_spin1.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
        mc1 = mcscf.CASSCF(m, 4, 4)
        mc1.fcisolver = FCI_as_DMRG(mol)
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot, -108.85974001740854, 8)
        dm1 = mc1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(lib.finger(dm1[0]), 5.33303, 4)

    def test_casci_in_casscf(self):
        mc1 = mcscf.CASSCF(m, 4, 4)
        e_tot, e_ci, fcivec = mc1.casci(mc1.mo_coeff)
        self.assertAlmostEqual(e_tot, -108.83741684447352, 9)

    def test_scanner(self):
        mc_scan = mcscf.CASSCF(scf.RHF(mol), 4, 4).as_scanner().as_scanner()
        mc_scan(mol)
        self.assertAlmostEqual(mc_scan.e_tot, -108.85974001740854, 8)

    def test_trust_region(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mc1.max_stepsize = 0.1
        mo = mc1.sort_mo_by_irrep({'A1u':3, 'A1g':1})
        mc1.ah_grad_trust_region = 0.3
        mc1.conv_tol = 1e-7
        tot_jk = []
        def count_jk(envs):
            tot_jk.append(envs.get('njk', 0))
        mc1.callback = count_jk
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82941031838349, 8)
        self.assertEqual(tot_jk, [3,6,6,4,4,3,6,6,3,6,6,3,4,4,3,3,3,3,4,4])

    def test_with_ci_init_guess(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        ci0 = numpy.zeros((6,6))
        ci0[0,1] = 1
        mc1.kernel(ci0=ci0)

        mc2 = mcscf.CASSCF(msym, 4, 4)
        mc2.wfnsym = 'A1u'
        mc2.kernel()
        self.assertAlmostEqual(mc1.e_tot, mc2.e_tot, 8)

    def test_dump_chk(self):
        mcdic = lib.chkfile.load(mc0.chkfile, 'mcscf')
        mcscf.chkfile.dump_mcscf(mc0, **mcdic)

    def test_state_average1(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.state_average_([0.5, 0.25, 0.25])
        mc.fcisolver.spin = 2
        mc.run()
        self.assertAlmostEqual(mc.e_states[0], -108.7513784239438, 6)
        self.assertAlmostEqual(mc.e_states[1], -108.6919327057737, 6)
        self.assertAlmostEqual(mc.e_states[2], -108.6919327057737, 6)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc = mcscf.CASCI(m, 4, 4)
        mc.state_average_([0.5, 0.25, 0.25])
        mc.fcisolver.spin = 2
        mc.kernel(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mc.e_states[0], -108.7513784239438, 6)
        self.assertAlmostEqual(mc.e_states[1], -108.6919327057737, 6)
        self.assertAlmostEqual(mc.e_states[2], -108.6919327057737, 6)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 7)
        # Second and third root are degenerated
        #self.assertAlmostEqual(abs((civec[1]*mc.ci[1]).sum()), 1, 7)

    def test_state_average_mix(self):
        mc = mcscf.CASSCF(m, 4, 4)
        cis1 = copy.copy(mc.fcisolver)
        cis1.spin = 2
        mc = mcscf.addons.state_average_mix(mc, [cis1, mc.fcisolver], [.5, .5])
        mc.run()
        self.assertAlmostEqual(mc.e_states[0], -108.7506795311190, 5)
        self.assertAlmostEqual(mc.e_states[1], -108.8582272809495, 5)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc.kernel(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mc.e_states[0], -108.7506795311190, 5)
        self.assertAlmostEqual(mc.e_states[1], -108.8582272809495, 5)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 7)
        self.assertAlmostEqual(abs((civec[1]*mc.ci[1]).sum()), 1, 7)

    def test_gen_g_hop (self):
        # Show that:
        #   1) relative error [(analytic - numerical) / numerical]
        #      of both gradient and hessian-vector product is
        #      asymptotically proportional to numerical step size (x**1)
        #   2) g_update asymptotically agrees with full recalculation
        #      of g quadratically (err ~x**2)
        #   3) diagonal hessian agrees with hessian-vector product
        # Move off of optimized orbitals so that terms in the Hessian that are
        # proportional to the gradient are double-checked.
        mc = mcscf.CASSCF(m, 4, 4).run ()
        mo = mc.mo_coeff = m.mo_coeff
        nmo, ncore, ncas, nelecas = mo.shape[1], mc.ncore, mc.ncas, mc.nelecas
        ci, nocc = mc.ci, ncore+ncas
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (ci, ncas, nelecas)
        def _get_energy (u=1):
            mo1 = numpy.dot (mo, u)
            eris = mc.ao2mo (mo1)
            fcasci = mcscf.mc1step._fake_h_for_fast_casci (mc, mo1, eris)
            h1, h0 = fcasci.get_h1eff ()
            h2 = fcasci.get_h2eff ()
            return (h0 + numpy.tensordot (h1, casdm1, axes=2)
                    + 0.5 * numpy.tensordot (h2, casdm2, axes=4))
        eris = mc.ao2mo (mo)
        e0 = _get_energy () 
        g0, g_update, h_op, h_diag = mc.gen_g_hop (mo, 1, casdm1, casdm2, eris)
        nvar = len (g0)
        numpy.random.seed (1)
        x0 = ((2 * numpy.random.random (nvar)) - 1)
        u0 = mc.update_rotate_matrix (x0)
        def _true_g_update (u):
            mo1 = numpy.dot (mo, u)
            g1 = mc.gen_g_hop (mo1, 1, casdm1, casdm2, mc.ao2mo (mo1))[0]
            return g1
        def _get_err (p):
            x1 = x0 / 2**p
            x1_norm = numpy.linalg.norm (x1)
            # Analytic
            gx = numpy.dot (g0, x1) 
            hx = h_op (x1)
            xhx = numpy.dot (hx, x1) 
            hx_norm = numpy.linalg.norm (hx)
            # Numerical
            u1 = mc.update_rotate_matrix (x1/2)
            e1 = _get_energy (u1)
            # Correction to g1 evaluated at different orbitals
            k1, g1 = (mc.unpack_uniq_var (v) for v in (x1, g0))
            g1_corr = numpy.dot (k1, g1)
            g1_corr = mc.pack_uniq_var (g1_corr-g1_corr.T)/2
            # TODO: rationalize conventions?
            # mc1step "x" = newton_casscf "x" * 2 
            # Therefore, I have to divide by 2 above ^, but not in
            # the newton_casscf version of this test. That's annoying
            u1 = mc.update_rotate_matrix (x1)
            g1 = _true_g_update (u1) + g1_corr
            de = (e1-e0)
            dg = (g1-g0)
            dg_norm = numpy.linalg.norm (dg)
            # Approximate update
            g1_approx = g_update (u1, ci) + g1_corr
            g1_norm = numpy.linalg.norm (g1)
            # Relative error
            g_err = (gx - de) / de
            h_err = numpy.linalg.norm (hx - dg) / dg_norm
            up_err = numpy.linalg.norm (g1_approx - g1) / g1_norm
            return x1_norm, g_err, h_err, up_err
        # To examine the actual convergence, uncomment the below
        #tab = numpy.array ([_get_err (p) for p in range (21)])
        #frac_tab = tab[1:,:] / tab[:-1,:]
        #print ("")
        #for row, frac_row in zip (tab[1:], frac_tab):
        #    print (row, frac_row)
        x2, g2, h2, u2 = _get_err (20)
        x1, g1, h1, u1 = _get_err (19)
        with self.subTest ('gradient'): 
            self.assertAlmostEqual (g2/g1, .5, 2)
        with self.subTest ('hessian-vector'):
            self.assertAlmostEqual (h2/h1, .5, 2)
        with self.subTest ('g_update'):
            self.assertAlmostEqual (u2/u1, .25, 2)
        h_diag_ref = numpy.zeros_like (h_diag)
        for p in range (nvar):
            x0[:] = 0.0
            x0[p] = 1.0
            h_diag_ref[p] = h_op (x0)[p]
        with self.subTest ('hessian diagonal'):
            self.assertAlmostEqual (lib.finger (h_diag), lib.finger (h_diag_ref), 9)

if __name__ == "__main__":
    print("Full Tests for mc1step")
    unittest.main()

