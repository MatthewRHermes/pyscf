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
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import grad
from pyscf.qmmm import itrf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = ''' H                 -0.00000000   -0.000    0.
 H                 -0.00000000   -0.000    1.
 H                 -0.00000000   -0.82    0.
 H                 -0.91000000   -0.020    0.''',
    basis = 'cc-pvdz')

class KnowValues(unittest.TestCase):
    def test_energy(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges)
        self.assertAlmostEqual(mf.kernel(), 2.0042702433049024, 9)

    def test_grad(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges).run()
        hfg = itrf.mm_charge_grad(grad.RHF(mf), coords, charges).run()
        self.assertAlmostEqual(numpy.linalg.norm(hfg.de), 30.316453059873059, 9)


if __name__ == "__main__":
    print("Full Tests for qmmm")
    unittest.main()
