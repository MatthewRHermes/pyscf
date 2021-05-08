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

def _test_vs_numerical (my_mc, mo0, ci0, eris0, g0, h0op, x0):
    ''' Returns a table which reports the fractional error of gradient
        and hessian-vector against finite-difference calculation
        as step size repeatedly halves '''
    e0 = _get_energy (my_mc, mo0, ci0, eris0)
    ci0_arr = numpy.asarray (ci0).reshape (-1,36)
    tab = numpy.zeros ((21,3))
    for p in range (21):
        x1 = x0 / (2**p)
        x1_norm = numpy.linalg.norm (x1)
        u, ci1 = newton_casscf.extract_rotation (my_mc, x1, 1, ci0)
        mo1 = numpy.dot (mo0, u)
        eris1 = my_mc.ao2mo (mo1)
        # Numerical
        e1 = _get_energy (my_mc, mo1, ci1, eris1)
        g1 = newton_casscf.gen_g_hop (my_mc, mo1, ci1, eris1)[0]
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
        tab[p,:] = [x1_norm, de_err, dg_err]
    #print ("\ntab:")
    #for row in tab:
    #    print ("{:.5e} {:.5e} {:.5e}".format (*row))
    return tab

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
        # plot the table on log-log axes to see the actual convergence
        err_tab = _test_vs_numerical (mc, mo, ci0, eris0, gall, hop, x)
        x_fac, g_err_fac, h_err_fac = err_tab[-1,:] / err_tab[-2,:]
        self.assertAlmostEqual (x_fac, g_err_fac, 2)
        self.assertAlmostEqual (x_fac, h_err_fac, 2)
        # Less rigorous fixed tests
        u, ci1 = newton_casscf.extract_rotation(mc, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gall), 21.288022525148595, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -4.6864640132374618, 8)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -412.9441873541524, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 8.152498748614988, 8)

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
        # plot the table on log-log axes to see the actual convergence
        err_tab = _test_vs_numerical (sa, mo, ci0, eris0, gall, hop, x)
        x_fac, g_err_fac, h_err_fac = err_tab[-1,:] / err_tab[-2,:]
        self.assertAlmostEqual (x_fac, g_err_fac, 2)
        self.assertAlmostEqual (x_fac, h_err_fac, 2)
        # Less rigorous fixed tests
        u, ci1 = newton_casscf.extract_rotation(sa, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gall), 32.46973284682048, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -63.6527761153809, 8)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -49.017079186126, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 176.2175779856562, 8)

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


