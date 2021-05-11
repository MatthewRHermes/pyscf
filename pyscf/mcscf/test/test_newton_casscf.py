#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
from functools import reduce
import numpy
from pyscf import gto, scf, lib, fci
from pyscf.mcscf import newton_casscf, CASSCF, addons, mc1step

mol = gto.Mole()
mol.verbose = lib.logger.DEBUG
mol.output = '/dev/null'
mol.atom = [
    ['H', ( 5.,-1.    , 1.   )],
    ['H', ( 0.,-5.    ,-2.   )],
    ['H', ( 4.,-0.5   ,-3.   )],
    ['H', ( 0.,-4.5   ,-1.   )],
    ['H', ( 3.,-0.5   ,-0.   )],
    ['H', ( 0.,-3.    ,-1.   )],
    ['H', ( 2.,-2.5   , 0.   )],
    ['H', ( 1., 1.    , 3.   )],
]
mol.basis = 'sto-3g'
mol.build()

b = 1.4
mol_N2 = gto.Mole()
mol_N2.build(
verbose = lib.logger.DEBUG,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = 1
)
mf_N2 = scf.RHF (mol_N2).run ()
solver1 = fci.FCI(mol_N2)
solver1.spin = 0
solver1.nroots = 2
solver2 = fci.FCI(mol_N2, singlet=False)
solver2.spin = 2
mc_N2 = CASSCF(mf_N2, 4, 4)
mc_N2 = addons.state_average_mix_(mc_N2, [solver1, solver2],
                                     (0.25,0.25,0.5)).newton ()
mc_N2.kernel()
mf = scf.RHF(mol)
mf.max_cycle = 3
mf.kernel()
mc = newton_casscf.CASSCF(mf, 4, 4)
mc.fcisolver = fci.direct_spin1.FCI(mol)
mc.kernel()
sa = CASSCF(mf, 4, 4)
sa.fcisolver = fci.direct_spin1.FCI (mol)
sa = sa.state_average ([1.0/2.0,]*2).newton ()
sa.fix_spin_(ss=0)
sa.conv_tol = 1e-10
# MRH 05/07/2021: fix spin for consistent solution
sa.kernel()
mo = sa.mo_coeff.copy ()
ci = [c.copy () for c in sa.ci]
# TODO: fix_spin_ compatibility of this test
# Is the hessian-vector product still wrong for fix_spin_?
sa = CASSCF(mf, 4, 4)
sa.fcisolver = fci.direct_spin1.FCI (mol)
sa = sa.state_average ([1.0/2.0,]*2).newton ()
sa.kernel(mo, ci)

def _get_energy (my_mc, mo1, ci1, eris1):
    dm1, dm2 = my_mc.fcisolver.make_rdm12 (ci1, my_mc.ncas, my_mc.nelecas)
    fcasci = mc1step._fake_h_for_fast_casci (my_mc, mo1, eris1)
    h1, h0 = fcasci.get_h1eff ()
    h2 = fcasci.get_h2eff ()
    return (h0 + numpy.tensordot (h1, dm1, axes=2)
            + 0.5 * numpy.tensordot (h2, dm2, axes=4))

def _get_redundant_gradient (my_mc, mo1, ci1, eris1):
    ''' For correcting the seminumerical hessian-vector calcn'''
    ncore, ncas = my_mc.ncore, my_mc.ncas
    nocc, nmo = ncore + ncas, mo1.shape[1]
    g0 = numpy.zeros ((nmo,nmo))
    g0_cas = g0[ncore:nocc,ncore:nocc]
    g0_core = g0[:ncore,:ncore]
    mo1_occ = mo1[:,:nocc]
    mo1_core = mo1[:,:ncore]
    mo1_cas = mo1_occ[:,ncore:]
    dm1, dm2 = my_mc.fcisolver.make_rdm12 (ci1, my_mc.ncas, my_mc.nelecas)
    fcasci = mc1step._fake_h_for_fast_casci (my_mc, mo1, eris1)
    h1, h0 = fcasci.get_h1eff ()
    h2 = fcasci.get_h2eff ()
    g0_cas[:,:] = numpy.dot (h1, dm1) + numpy.einsum ('pabc,qabc->pq', h2, dm2)
    h1_core = reduce (numpy.dot, (mo1_core.T, my_mc.get_hcore (), mo1_core))
    h1_core += eris1.vhf_c[:ncore,:ncore]
    for i in range (ncore):
        jbuf = eris1.ppaa[i][:ncore,:,:]
        kbuf = eris1.papa[i][:,:ncore,:]
        h1_core[:,i] += (numpy.einsum('quv,uv->q', jbuf, dm1) -
                         numpy.einsum('uqv,uv->q', kbuf, dm1) * .5)
    g0_core[:,:] = h1_core*2
    return (g0-g0.T)*2


def _test_vs_numerical (my_mc, mo0, ci0, eris0, g0, g1op, h0op, x0):
    ''' Returns a table which reports the fractional error of gradient
        and hessian-vector against finite-difference calculation
        as step size repeatedly halves '''
    e0 = _get_energy (my_mc, mo0, ci0, eris0)
    _get_redundant_gradient (my_mc, mo0, ci0, eris0)
    ci0_arr = numpy.asarray (ci0).reshape (-1,36)
    tab = numpy.zeros ((21,3))
    nmo = mo0.shape[1]
    ncore, ncas, frozen = my_mc.ncore, my_mc.ncas, my_mc.frozen
    nvar_orb = numpy.count_nonzero (my_mc.uniq_var_indices (nmo,
        ncore,ncas,frozen))
    g0_orb = (_get_redundant_gradient (my_mc, mo0, ci0, eris0)
              + my_mc.unpack_uniq_var (g0[:nvar_orb]))
    def _get_err (p):
        x1 = x0 / (2**p)
        x1_norm = numpy.linalg.norm (x1)
        u, ci1 = newton_casscf.extract_rotation (my_mc, x1, 1, ci0)
        mo1 = numpy.dot (mo0, u)
        eris1 = my_mc.ao2mo (mo1)
        # Numerical
        e1 = _get_energy (my_mc, mo1, ci1, eris1)
        k1 = my_mc.unpack_uniq_var (x1[:nvar_orb])
        g1_corr = numpy.dot (k1, g0_orb)
        g1_corr = my_mc.pack_uniq_var (g1_corr-g1_corr.T)/2
        g1 = newton_casscf.gen_g_hop (my_mc, mo1, ci1, eris1)[0]
        g1[:nvar_orb] += g1_corr
        # g_update
        g1_approx = g1op (u, ci1) 
        g1_approx[:nvar_orb] += g1_corr
        g1_norm = numpy.linalg.norm (g1)
        de_ref = e1 - e0
        dg_ref = g1 - g0
        # Project undefined components out of dg_ref
        gci = dg_ref[20:].reshape (-1,36)
        gc = numpy.einsum ('pr,pr->p', gci, ci0_arr)
        dg_ref[20:] -= (gc[:,None] * ci0_arr).ravel ()
        dg_ref_norm = numpy.linalg.norm (dg_ref)
        # Analytic
        hx = h0op (x1)
        gx = numpy.dot (g0, x1)
        xhx = numpy.dot (hx, x1)
        de_test = gx 
        dg_test = hx
        # Relative error
        de_err = (de_test-de_ref)/de_ref
        dg_err = numpy.linalg.norm (dg_test-dg_ref) / dg_ref_norm
        du_err = numpy.linalg.norm (g1_approx-g1) / g1_norm
        return x1_norm, de_err, dg_err, du_err
    h_diag_ref = numpy.empty_like (g0)
    x1 = numpy.empty_like (x0)
    for p in range (h_diag_ref.size):
        x1[:] = 0.0
        x1[p] = 1.0
        h_diag_ref[p] = h0op (x1)[p]
    return _get_err, h_diag_ref

def tearDownModule():
    global mol, mf, mc, sa, mol_N2, mf_N2, mc_N2
    del mol, mf, mc, sa, mol_N2, mf_N2, mc_N2


class KnownValues(unittest.TestCase):
    def test_gen_g_hop(self):
        numpy.random.seed(1)
        mo = numpy.random.random(mf.mo_coeff.shape)
        ci0 = numpy.random.random((6,6))
        ci0/= numpy.linalg.norm(ci0)
        eris0 = mc.ao2mo (mo)
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(mc, mo, ci0, eris0)
        x = numpy.random.random(gall.size)
        # More rigorous test: taking the limit
        # plot vs p on log-log axes to see the actual convergence
        _get_err, hdiag_ref = _test_vs_numerical (mc, mo, ci0, eris0, gall, gop, hop, x)
        self.assertAlmostEqual (lib.finger (hdiag), lib.finger (hdiag_ref), 8) 
        x1, g1, h1, u1 = _get_err (20)
        x2, g2, h2, u2 = _get_err (19)
        self.assertAlmostEqual (g1/g2, .5, 2)
        self.assertAlmostEqual (h1/h2, .5, 2)
        self.assertAlmostEqual (u1/u2, .25, 2)
        # Less rigorous fixed tests
        u, ci1 = newton_casscf.extract_rotation(mc, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gall), 21.288022525148595, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -15.618395788969822, 8)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -412.9441873541524, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 24.045699256609716, 8)

    def test_get_grad(self):
        self.assertAlmostEqual(mc.e_tot, -3.6268060853430573, 8)
        self.assertAlmostEqual(abs(mc.get_grad()).max(), 0, 5)

    def test_sa_gen_g_hop(self):
        numpy.random.seed(1)
        mo = numpy.random.random(mf.mo_coeff.shape)
        ci0 = numpy.random.random((2,36))
        ci0/= numpy.linalg.norm(ci0, axis=1)[:,None]
        ci0 = list (ci0.reshape ((2,6,6)))
        eris0 = sa.ao2mo (mo)
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(sa, mo, ci0, eris0)
        x = numpy.random.random(gall.size)
        # More rigorous test: taking the limit
        # plot vs p on log-log axes to see the actual convergence
        _get_err, hdiag_ref = _test_vs_numerical (sa, mo, ci0, eris0, gall, gop, hop, x)
        self.assertAlmostEqual (lib.finger (hdiag), lib.finger (hdiag_ref), 8) 
        x1, g1, h1, u1 = _get_err (20)
        x2, g2, h2, u2 = _get_err (19)
        self.assertAlmostEqual (g1/g2, .5, 2)
        self.assertAlmostEqual (h1/h2, .5, 2)
        self.assertAlmostEqual (u1/u2, .25, 2)
        # Less rigorous fixed tests
        u, ci1 = newton_casscf.extract_rotation(sa, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gall), 32.46973284682048, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -70.61862254321514, 8)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -49.017079186126, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 136.20779886241564, 8)

    def test_sa_get_grad(self):
        self.assertAlmostEqual(sa.e_tot, -3.626586177820634, 7)
        self.assertAlmostEqual(abs(sa.get_grad()).max(), 0, 5)

    def test_sa_mix(self):
        e = mc_N2.e_states
        self.assertAlmostEqual(mc_N2.e_tot, -108.80340952016508, 7)
        self.assertAlmostEqual(mc_N2.e_average, -108.80340952016508, 7)
        self.assertAlmostEqual(numpy.dot(e,[.25,.25,.5]), -108.80340952016508, 7)
        dm1 = mc_N2.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.52172669549357464, 4)
        self.assertAlmostEqual(lib.fp(dm1[1]), 0.53366776017869022, 4)
        self.assertAlmostEqual(lib.fp(dm1[0]+dm1[1]), 1.0553944556722636, 4)
        mc_N2.cas_natorb()



if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()


