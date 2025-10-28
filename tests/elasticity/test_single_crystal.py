"""Unit tests for single crystal elasticity"""
import numpy as np
import pytest

from polycrystal.elasticity.single_crystal import SingleCrystal
from polycrystal.elasticity.moduli_tools.cubic import Cubic
from polycrystal.orientations.crystalsymmetry import get_symmetries


TOL = 1e-14


def maxdiff(a, b):
    return np.max(np.abs(a - b))



class TestSingleCrystal:

    @pytest.fixture
    def ID_6X6(cls):
        return np.identity(6)

    @pytest.fixture
    def eps0(cls):
        """Test strain data"""
        return np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ])

    @pytest.fixture
    def rot_90(cls):
        """90 degree rotations about coordinate axes"""
        return np.array([
            [[ 1.,  0.,  0.],
             [ 0.,  0., -1.],
             [ 0.,  1.,  0.]],
            [[ 0.,  0.,  1.],
             [ 0.,  1.,  0.],
             [-1.,  0.,  0.]],
            [[ 0., -1.,  0.],
             [ 1.,  0.,  0.],
             [ 0.,  0.,  1.]]
        ])

    def test_identity(self, ID_6X6):

        # From K, G
        sx = SingleCrystal.from_K_G(1/3, 1/2)
        sx.system = "MANDEL"
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # From E, nu
        sx = SingleCrystal.from_E_nu(1, 0)
        sx.system = "MANDEL"
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Isotropic: c11, c12
        sx = SingleCrystal(
            'isotropic', [1.0,  0.0]
        )
        sx.system = "MANDEL"
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Cubic
        sx = SingleCrystal(
            'cubic', [1.0,  0.0,  0.5]
        )
        sx.system = "MANDEL"
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Hexagonal
        sx = SingleCrystal(
            'hexagonal', [1.0,  0.0,  0.0,  1.0,  0.5]
        )
        sx.system = "MANDEL"
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Triclinic
        cij = [
            1.0, 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,
            0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  1.0
        ]
        sx = SingleCrystal('triclinic', cij, system="MANDEL")
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

    def test_cij(self):
        """Test cij"""

        # Isotropic.
        sx = SingleCrystal(
            'isotropic', [2.3, 4.9],
            system = "VOIGT_GAMMA",
        )
        assert np.allclose([2.3, 4.9], sx.cij)
        sx.system = "MANDEL"
        assert np.allclose([2.3, 4.9], sx.cij)

        # Cubic.
        sx = SingleCrystal(
            'cubic', [2.3, 4.5, 7.8],
            system = "VOIGT_GAMMA",
        )
        assert np.allclose([2.3, 4.5, 7.8], sx.cij)
        sx.system = "VOIGT_EPSILON"
        assert np.allclose([2.3, 4.5, 2 * 7.8], sx.cij)

        # Hexagonal
        test_cij_eps = [1.0,  2.3, 3.4, 5.6, 2 * 3.3]
        test_cij_gam = [1.0,  2.3, 3.4, 5.6, 3.3]
        sx = SingleCrystal(
            'hexagonal', test_cij_gam,
        )
        # sx.system = "VOIGT_GAMMA"
        assert np.allclose(test_cij_gam, sx.cij)

        # Triclinic
        test_cij = np.arange(21)
        sx = SingleCrystal(
            'triclinic', test_cij,
            system="MANDEL",
        )
        sx.system = "VOIGT_GAMMA"
        top_left = np.array((0, 1, 2, 6, 7, 11))
        assert np.allclose(sx.cij[top_left], test_cij[top_left])
        top_right = np.array((3, 4, 5, 8, 9, 10, 12, 13, 14,))
        assert np.allclose(sx.cij[top_right], 1/np.sqrt(2) * test_cij[top_right])
        # Bottom Left.
        assert np.allclose(sx.cij[15:], 0.5 * test_cij[15:])

    def test_units(self):

        sx = SingleCrystal('isotropic', [1.0, 0.0], units= "GPa")
        assert sx.stiffness[0, 0] == 1.0

        sx.units = "MPa"
        assert sx.stiffness[0, 0] == 1.0e3
        assert sx.units == sx.moduli.units

    def test_apply(self, rot_90, eps0):
        """Test apply_stiffness and apply_compliance methods"""
        sx = SingleCrystal.from_K_G(2/3, 1.0, system="VOIGT_GAMMA")
        sx = SingleCrystal.from_K_G(2/3, 1.0, system="VOIGT_EPSILON")
        sig0 = sx.apply_stiffness(eps0)
        assert np.allclose(sig0, 2 * eps0)

        # Add rotation matrices.
        eps3 = np.tile(eps0, (3, 1, 1))
        sig3 = sx.apply_stiffness(eps3, rot_90)
        assert np.allclose(sig3, 2 * eps3)

        # Apply compliance to get original back.
        eps3_a = sx.apply_compliance(sig3, rot_90)
        assert np.allclose(eps3, eps3_a)

    def test_apply_cubic(self, eps0):
        """Test cubic stiffness/compliance under rotation"""
        sym = 'cubic'
        system = Cubic.SYSTEMS.MANDEL
        cij = Cubic.cij_from_K_Gd_Gs(3., 2, 5., system)
        sx = SingleCrystal('cubic', cij, system=system)

        cubsym = get_symmetries(sym)
        rmats = cubsym.rmats
        nsym = len(rmats)

        eps3 = np.tile(eps0, (nsym, 1, 1))
        sig3 = sx.apply_stiffness(eps3, rmats)

        # Stresses should be the same since the crystal is invariant under it's
        # symmetry group.
        assert np.allclose(sig3, np.tile(sig3[0], (nsym, 1, 1)))

        # Verify compliance.
        eps3_a = sx.apply_compliance(sig3, rmats)
        assert np.allclose(eps3, eps3_a)

    def test_apply_hexagonal(self, eps0):
        """Test hexagonal stiffness/compliance under rotation"""
        # Note that you have to choose moduli to ensure the stiffness is nonsingular.
        # The choice: [5, 4, 3, 2, 1] apparently led to a singular matrix.

        sym = 'hexagonal'
        mod = [5.0, 6.2, 3.2, 1.9, 4.2]
        sx = SingleCrystal(sym, mod)
        sx.system = "MANDEL"

        hexsym = get_symmetries(sym)
        rmats = hexsym.rmats
        nsym = len(rmats)

        eps3 = np.tile(eps0, (nsym, 1, 1))
        sig3 = sx.apply_stiffness(eps3, rmats)

        # Stresses should be the same since the crystal is invariant under it's
        # symmetry group.

        assert np.allclose(sig3, np.tile(sig3[0], (nsym, 1, 1)))

        # Verify compliance.
        eps3_a = sx.apply_compliance(sig3, rmats)
        assert np.allclose(eps3, eps3_a)


    def test_change_of_basis(self, eps0, rot_90):
        """Test change of basis"""
        cob = SingleCrystal._change_basis(eps0, None)
        assert np.allclose(eps0, cob)

        eps3 = np.tile(eps0, (3, 1, 1))
        cob3 = SingleCrystal._change_basis(eps0, rot_90)

        for i in (0, 1, 2):
            rot_i = rot_90[i]
            cob_i = rot_i @ eps0 @ rot_i.T
            assert np.allclose(cob_i, cob3[i])


class TestThermalExpansion:

    def test_cte_none(self):

        # From float.
        sx = SingleCrystal.from_E_nu(1, 0)
        assert not hasattr(sx, "cte")

    def test_cte_float(self):

        # From float.
        cte = 1.2e-3
        sx = SingleCrystal.from_E_nu(1, 0, cte=cte)
        assert np.allclose(sx.cte, np.diag(3 * (cte,)))

    def test_cte_array33(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
            [3.1, 41, 5.9],
        ])
        sx = SingleCrystal.from_E_nu(1, 0, cte=arr)
        assert np.allclose(sx.cte, arr)

    def test_cte_array_shape(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
        ])
        with pytest.raises(RuntimeError):
            sx = SingleCrystal.from_E_nu(1, 0, cte=arr)
