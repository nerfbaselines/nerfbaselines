import math
import numpy as np
from scipy import linalg
from operator import mul
from functools import reduce


def quaternion_multiply(q1, q2):
    """Multiply two sets of quaternions."""
    a = q1[..., 0]*q2[...,0] - q1[...,1]*q2[...,1] - q1[...,2]*q2[...,2] - q1[...,3]*q2[...,3]
    b = q1[..., 0]*q2[...,1] + q1[...,1]*q2[...,0] + q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2]
    c = q1[..., 0]*q2[...,2] - q1[...,1]*q2[...,3] + q1[...,2]*q2[...,0] + q1[...,3]*q2[...,1]
    d = q1[..., 0]*q2[...,3] + q1[...,1]*q2[...,2] - q1[...,2]*q2[...,1] + q1[...,3]*q2[...,0]
    return np.stack([a, b, c, d], axis=-1)


def quaternion_conjugate(q):
    """Return quaternion-conjugate of quaternion q̄"""
    return np.stack([+q[...,0], -q[...,1], -q[..., 2], -q[...,3]], -1)

def quaternion_to_rotation_matrix(r):
    norm = np.sqrt((r**2).sum(-1))

    q = r / norm[..., None]

    R = np.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype)

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotation_matrix_to_quaternion(R):
    """Convert input 3x3 rotation matrix to unit quaternion.

    Assuming an orthogonal 3x3 matrix ℛ rotates a vector v such that

        v' = ℛ * v,

    we can also express this rotation in terms of a unit quaternion R such that

        v' = R * v * R⁻¹,

    where v and v' are now considered pure-vector quaternions.  This function
    returns that quaternion.  If `rot` is not orthogonal, the "closest" orthogonal
    matrix is used; see Notes below.

    Parameters
    ----------
    R : (...Nx3x3) float array
        Each 3x3 matrix represents a rotation by multiplying (from the left)
        a column vector to produce a rotated column vector.  Note that this
        input may actually have ndims>3; it is just assumed that the last
        two dimensions have size 3, representing the matrix.

    Returns
    -------
    q : array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-2].

    Raises
    ------
    LinAlgError
        If any of the eigenvalue solutions does not converge

    Notes
    -----
    This function uses Bar-Itzhack's algorithm to allow for
    non-orthogonal matrices.  [J. Guidance, Vol. 23, No. 6, p. 1085
    <http://dx.doi.org/10.2514/2.4654>]  This will almost certainly be quite a bit
    slower than simpler versions, though it will be more robust to numerical errors
    in the rotation matrix.  Also note that the Bar-Itzhack paper uses some pretty
    weird conventions.  The last component of the quaternion appears to represent
    the scalar, and the quaternion itself is conjugated relative to the convention
    used throughout the quaternionic module.
    """

    rot = np.array(R, copy=False)
    shape = rot.shape[:-2]


    K3 = np.empty(shape+(4, 4), dtype=rot.dtype)
    K3[..., 0, 0] = (rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2])/3
    K3[..., 0, 1] = (rot[..., 1, 0] + rot[..., 0, 1])/3
    K3[..., 0, 2] = (rot[..., 2, 0] + rot[..., 0, 2])/3
    K3[..., 0, 3] = (rot[..., 1, 2] - rot[..., 2, 1])/3
    K3[..., 1, 0] = K3[..., 0, 1]
    K3[..., 1, 1] = (rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2])/3
    K3[..., 1, 2] = (rot[..., 2, 1] + rot[..., 1, 2])/3
    K3[..., 1, 3] = (rot[..., 2, 0] - rot[..., 0, 2])/3
    K3[..., 2, 0] = K3[..., 0, 2]
    K3[..., 2, 1] = K3[..., 1, 2]
    K3[..., 2, 2] = (rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1])/3
    K3[..., 2, 3] = (rot[..., 0, 1] - rot[..., 1, 0])/3
    K3[..., 3, 0] = K3[..., 0, 3]
    K3[..., 3, 1] = K3[..., 1, 3]
    K3[..., 3, 2] = K3[..., 2, 3]
    K3[..., 3, 3] = (rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])/3

    if not shape:
        q = np.empty((4,), dtype=rot.dtype)
        eigvals, eigvecs = linalg.eigh(K3.T, subset_by_index=(3, 3))
        q[0] = eigvecs[-1].item()
        q[1:] = -eigvecs[:-1].flatten()
        return q
    else:
        q = np.empty(shape+(4,), dtype=rot.dtype)
        for flat_index in range(reduce(mul, shape)):
            multi_index = np.unravel_index(flat_index, shape)
            eigvals, eigvecs = linalg.eigh(K3[multi_index], subset_by_index=(3, 3))
            q[multi_index+(0,)] = eigvecs[-1]
            q[multi_index+(slice(1,None),)] = -eigvecs[:-1].flatten()
        return q


def wigner_D_matrix(R, ell_max: int):
    """
    Build a Wigner matrix from a rotation matrix.
    """
    """
This code was taken from https://github.com/moble/spherica
It is hosted under the following license:

The MIT License (MIT)

Copyright (c) 2023 Mike Boyle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    ell_min = 0
    mp_max=np.iinfo(np.int64).max

    def _complex_powers(zravel, M, zpowers):
        """Helper function for complex_powers(z, M)"""
        for i in range(zravel.size):
            zpowers[i, 0] = 1.0 + 0.0j
            if M > 0:
                z = zravel[i]
                θ = 1
                while z.real<0 or z.imag<0:
                    θ *= 1j
                    z /= 1j
                zpowers[i, 1] = z
                clock = θ
                dc = -2 * np.sqrt(z).imag ** 2
                t = 2 * dc
                dz = dc * (1 + 2 * zpowers[i, 1]) + 1j * np.sqrt(-dc * (2 + dc))
                for m in range(2, M+1):
                    zpowers[i, m] = zpowers[i, m-1] + dz
                    dz += t * zpowers[i, m]
                    zpowers[i, m-1] *= clock
                    clock *= θ
                zpowers[i, M] *= clock
    
    def WignerHsize(mp_max, ell_max=-2):
        if ell_max == -2:
            ell_max = mp_max
        elif ell_max < 0:
            return 0
        if mp_max is None or mp_max >= ell_max:
            return (ell_max+1) * (ell_max+2) * (2*ell_max+3) // 6
        else:
            return ((ell_max+1) * (ell_max+2) * (2*ell_max+3) - 2*(ell_max-mp_max)*(ell_max-mp_max+1)*(ell_max-mp_max+2)) // 6

    def WignerDsize(ell_min, mp_max, ell_max=-1):
        if ell_max < 0:
            ell_max = mp_max
        if mp_max >= ell_max:
            return (
                ell_max * (ell_max * (4 * ell_max + 12) + 11)
                + ell_min * (1 - 4 * ell_min**2)
                + 3
            ) // 3
        if mp_max > ell_min:
            return (
                3 * ell_max * (ell_max + 2)
                + ell_min * (1 - 4 * ell_min**2)
                + mp_max * (
                    3 * ell_max * (2 * ell_max + 4)
                    + mp_max * (-2 * mp_max - 3) + 5
                )
                + 3
            ) // 3
        else:
            return (ell_max * (ell_max + 2) - ell_min**2) * (1 + 2 * mp_max) + 2 * mp_max + 1
    
    def _WignerHindex(ell, mp, m, mp_max):
        mp_max = min(mp_max, ell)
        i = WignerHsize(mp_max, ell-1)  # total size of everything with smaller ell
        if mp<1:
            i += (mp_max + mp) * (2*ell - mp_max + mp + 1) // 2  # size of wedge to the left of m'
        else:
            i += (mp_max + 1) * (2*ell - mp_max + 2) // 2  # size of entire left half of wedge
            i += (mp - 1) * (2*ell - mp + 2) // 2  # size of right half of wedge to the left of m'
        i += m - abs(mp)  # size of column in wedge between m and |m'|
        return i

    def WignerHindex(ell, mp, m, mp_max=None):
        if ell == 0:
            return 0
        mpmax = ell
        if mp_max is not None:
            mpmax = min(mp_max, mpmax)
        if m < -mp:
            if m < mp:
                return _WignerHindex(ell, -mp, -m, mpmax)
            else:
                return _WignerHindex(ell, -m, -mp, mpmax)
        elif m < mp:
            return _WignerHindex(ell, m, mp, mpmax)
        else:
            return _WignerHindex(ell, mp, m, mpmax)

    def WignerDindex(ell, mp, m, ell_min=0, mp_max=-1):
        if mp_max < 0:
            mp_max = ell
        i = (mp + min(mp_max, ell)) * (2 * ell + 1) + m + ell
        if ell > ell_min:
            i += WignerDsize(ell_min, mp_max, ell-1)
        return i

    def nm_index(n, m):
        return m + n * (n + 1)
    
    def nabsm_index(n, absm):
        return absm + (n * (n + 1)) // 2

    def _step_1(Hwedge):
        """If n=0 set H_{0}^{0,0}=1."""
        Hwedge[0] = 1.0


    def _step_2(g, h, n_max, mp_max, Hwedge, Hextra, Hv, expiβ):
        """Compute values H^{0,m}_{n}(β)for m=0,...,n and H^{0,m}_{n+1}(β) for m=0,...,n+1 using Eq. (32):

            H^{0,m}_{n}(β) = (-1)^m √((n-|m|)! / (n+|m|)!) P^{|m|}_{n}(cos β)
                        = (-1)^m (sin β)^m P̂^{|m|}_{n}(cos β) / √(k (2n+1))

        This function computes the associated Legendre functions directly by recursion
        as explained by Holmes and Featherstone (2002), doi:10.1007/s00190-002-0216-2.
        Note that I had to adjust certain steps for consistency with the notation
        assumed by arxiv:1403.7698 -- mostly involving factors of (-1)**m.

        NOTE: Though not specified in arxiv:1403.7698, there is not enough information
        for step 4 unless we also use symmetry to set H^{1,0}_{n} here.  Similarly,
        step 5 needs additional information, which depends on setting H^{0, -1}_{n}
        from its symmetric equivalent H^{0, 1}_{n} in this step.

        """
        cosβ = expiβ.real
        sinβ = expiβ.imag
        if n_max > 0:
            # n = 1
            n0n_index = WignerHindex(1, 0, 1, mp_max)
            nn_index = nm_index(1, 1)
            Hwedge[n0n_index] = np.sqrt(3)  # Un-normalized
            Hwedge[n0n_index-1] = (g[nn_index-1] * cosβ) * 1.0 / np.sqrt(2)  # Normalized
            # n = 2, ..., n_max+1
            for n in range(2, n_max+2):
                if n <= n_max:
                    n0n_index = WignerHindex(n, 0, n, mp_max)
                    H = Hwedge
                else:
                    n0n_index = n
                    H = Hextra
                nm10nm1_index = WignerHindex(n-1, 0, n-1, mp_max)
                nn_index = nm_index(n, n)
                const = np.sqrt(1.0 + 0.5/n)
                gi = g[nn_index-1]
                # m = n
                H[n0n_index] = const * Hwedge[nm10nm1_index]
                # m = n-1
                H[n0n_index-1] = gi * cosβ * H[n0n_index]
                # m = n-2, ..., 1
                for i in range(2, n):
                    gi = g[nn_index-i]
                    hi = h[nn_index-i]
                    H[n0n_index-i] = gi * cosβ * H[n0n_index-i+1] - hi * sinβ**2 * H[n0n_index-i+2]
                # m = 0, with normalization
                const = 1.0 / np.sqrt(4*n+2)
                gi = g[nn_index-n]
                hi = h[nn_index-n]
                H[n0n_index-n] = (gi * cosβ * H[n0n_index-n+1] - hi * sinβ**2 * H[n0n_index-n+2]) * const
                # Now, loop back through, correcting the normalization for this row, except for n=n element
                prefactor = const
                for i in range(1, n):
                    prefactor *= sinβ
                    H[n0n_index-n+i] *= prefactor
                # Supply extra edge cases as noted in docstring
                if n <= n_max:
                    Hv[nm_index(n, 1)] = Hwedge[WignerHindex(n, 0, 1, mp_max)]
                    Hv[nm_index(n, 0)] = Hwedge[WignerHindex(n, 0, 1, mp_max)]
            # Correct normalization of m=n elements
            prefactor = 1.0
            for n in range(1, n_max+1):
                prefactor *= sinβ
                Hwedge[WignerHindex(n, 0, n, mp_max)] *= prefactor / np.sqrt(4*n+2)
            for n in [n_max+1]:
                prefactor *= sinβ
                Hextra[n] *= prefactor / np.sqrt(4*n+2)
            # Supply extra edge cases as noted in docstring
            Hv[nm_index(1, 1)] = Hwedge[WignerHindex(1, 0, 1, mp_max)]
            Hv[nm_index(1, 0)] = Hwedge[WignerHindex(1, 0, 1, mp_max)]

    def _step_3(a, b, n_max, mp_max, Hwedge, Hextra, expiβ):
        """Use relation (41) to compute H^{1,m}_{n}(β) for m=1,...,n.  Using symmetry and shift
        of the indices this relation can be written as

            b^{0}_{n+1} H^{1, m}_{n} =   (b^{−m−1}_{n+1} (1−cosβ))/2 H^{0, m+1}_{n+1}
                                    − (b^{ m−1}_{n+1} (1+cosβ))/2 H^{0, m−1}_{n+1}
                                    − a^{m}_{n} sinβ H^{0, m}_{n+1}

        """
        cosβ = expiβ.real
        sinβ = expiβ.imag
        if n_max > 0 and mp_max > 0:
            for n in range(1, n_max+1):
                # m = 1, ..., n
                i1 = WignerHindex(n, 1, 1, mp_max)
                if n+1 <= n_max:
                    i2 = WignerHindex(n+1, 0, 0, mp_max)
                    H2 = Hwedge
                else:
                    i2 = 0
                    H2 = Hextra
                i3 = nm_index(n+1, 0)
                i4 = nabsm_index(n, 1)
                inverse_b5 = 1.0 / b[i3]
                for i in range(n):
                    b6 = b[-i+i3-2]
                    b7 = b[i+i3]
                    a8 = a[i+i4]
                    Hwedge[i+i1] = inverse_b5 * (
                        0.5 * (
                            b6 * (1-cosβ) * H2[i+i2+2]
                            - b7 * (1+cosβ) * H2[i+i2]
                        )
                        - a8 * sinβ * H2[i+i2+1]
                    )

    def _step_4(d, n_max, mp_max, Hwedge, Hv):
        """Recursively compute H^{m'+1, m}_{n}(β) for m'=1,...,n−1, m=m',...,n using relation (50) resolved
        with respect to H^{m'+1, m}_{n}:

        d^{m'}_{n} H^{m'+1, m}_{n} =   d^{m'−1}_{n} H^{m'−1, m}_{n}
                                    − d^{m−1}_{n} H^{m', m−1}_{n}
                                    + d^{m}_{n} H^{m', m+1}_{n}

        (where the last term drops out for m=n).

        """
        if n_max > 0 and mp_max > 0:
            for n in range(2, n_max+1):
                for mp in range(1, min(n, mp_max)):
                    # m = m', ..., n-1
                    # i1 = WignerHindex(n, mp+1, mp, mp_max)
                    i1 = WignerHindex(n, mp+1, mp+1, mp_max) - 1
                    i2 = WignerHindex(n, mp-1, mp, mp_max)
                    # i3 = WignerHindex(n, mp, mp-1, mp_max)
                    i3 = WignerHindex(n, mp, mp, mp_max) - 1
                    i4 = WignerHindex(n, mp, mp+1, mp_max)
                    i5 = nm_index(n, mp)
                    i6 = nm_index(n, mp-1)
                    inverse_d5 = 1.0 / d[i5]
                    d6 = d[i6]
                    for i in [0]:
                        d7 = d[i+i6]
                        d8 = d[i+i5]
                        Hv[i+nm_index(n, mp+1)] = inverse_d5 * (
                            d6 * Hwedge[i+i2]
                            - d7 * Hv[i+nm_index(n, mp)]
                            + d8 * Hwedge[i+i4]
                        )
                    for i in range(1, n-mp):
                        d7 = d[i+i6]
                        d8 = d[i+i5]
                        Hwedge[i+i1] = inverse_d5 * (
                            d6 * Hwedge[i+i2]
                            - d7 * Hwedge[i+i3]
                            + d8 * Hwedge[i+i4]
                        )
                    # m = n
                    for i in [n-mp]:
                        Hwedge[i+i1] = inverse_d5 * (
                            d6 * Hwedge[i+i2]
                            - d[i+i6] * Hwedge[i+i3]
                        )

    def _step_5(d, n_max, mp_max, Hwedge, Hv):
        """Recursively compute H^{m'−1, m}_{n}(β) for m'=−1,...,−n+1, m=−m',...,n using relation (50)
        resolved with respect to H^{m'−1, m}_{n}:

        d^{m'−1}_{n} H^{m'−1, m}_{n} = d^{m'}_{n} H^{m'+1, m}_{n}
                                        + d^{m−1}_{n} H^{m', m−1}_{n}
                                        − d^{m}_{n} H^{m', m+1}_{n}

        (where the last term drops out for m=n).

        NOTE: Although arxiv:1403.7698 specifies the loop over mp to start at -1, I
        find it necessary to start at 0, or there will be missing information.  This
        also requires setting the (m',m)=(0,-1) components before beginning this loop.

        """
        if n_max > 0 and mp_max > 0:
            for n in range(0, n_max+1):
                for mp in range(0, -min(n, mp_max), -1):
                    # m = -m', ..., n-1
                    # i1 = WignerHindex(n, mp-1, -mp, mp_max)
                    i1 = WignerHindex(n, mp-1, -mp+1, mp_max) - 1
                    # i2 = WignerHindex(n, mp+1, -mp, mp_max)
                    i2 = WignerHindex(n, mp+1, -mp+1, mp_max) - 1
                    # i3 = WignerHindex(n, mp, -mp-1, mp_max)
                    i3 = WignerHindex(n, mp, -mp, mp_max) - 1
                    i4 = WignerHindex(n, mp, -mp+1, mp_max)
                    i5 = nm_index(n, mp-1)
                    i6 = nm_index(n, mp)
                    i7 = nm_index(n, -mp-1)
                    i8 = nm_index(n, -mp)
                    inverse_d5 = 1.0 / d[i5]
                    d6 = d[i6]
                    for i in [0]:
                        d7 = d[i+i7]
                        d8 = d[i+i8]
                        if mp == 0:
                            Hv[i+nm_index(n, mp-1)] = inverse_d5 * (
                                d6 * Hv[i+nm_index(n, mp+1)]
                                + d7 * Hv[i+nm_index(n, mp)]
                                - d8 * Hwedge[i+i4]
                            )
                        else:
                            Hv[i+nm_index(n, mp-1)] = inverse_d5 * (
                                d6 * Hwedge[i+i2]
                                + d7 * Hv[i+nm_index(n, mp)]
                                - d8 * Hwedge[i+i4]
                            )
                    for i in range(1, n+mp):
                        d7 = d[i+i7]
                        d8 = d[i+i8]
                        Hwedge[i+i1] = inverse_d5 * (
                            d6 * Hwedge[i+i2]
                            + d7 * Hwedge[i+i3]
                            - d8 * Hwedge[i+i4]
                        )
                    # m = n
                    i = n+mp
                    Hwedge[i+i1] = inverse_d5 * (
                        d6 * Hwedge[i+i2]
                        + d[i+i7] * Hwedge[i+i3]
                    )
    
    def ϵ(m):
        if m <= 0:
            return 1
        elif m%2:
            return -1
        else:
            return 1

    def _fill_wigner_D(ell_min, ell_max, mp_max, 𝔇, Hwedge, zₐpowers, zᵧpowers):
        """Helper function for Wigner.D"""
        # 𝔇ˡₘₚ,ₘ(R) = dˡₘₚ,ₘ(R) exp[iϕₐ(m-mp)+iϕₛ(m+mp)] = dˡₘₚ,ₘ(R) exp[i(ϕₛ+ϕₐ)m+i(ϕₛ-ϕₐ)mp]
        # exp[iϕₛ] = R̂ₛ = hat(R[0] + 1j * R[3]) = zp
        # exp[iϕₐ] = R̂ₐ = hat(R[2] + 1j * R[1]) = zm.conjugate()
        # exp[i(ϕₛ+ϕₐ)] = zp * zm.conjugate() = z[2] = zᵧ
        # exp[i(ϕₛ-ϕₐ)] = zp * zm = z[0] = zₐ
        for ell in range(ell_min, ell_max+1):
            for mp in range(-ell, 0):
                i_D = WignerDindex(ell, mp, -ell, ell_min)
                for m in range(-ell, 0):
                    i_H = WignerHindex(ell, mp, m, mp_max)
                    𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[-mp].conjugate()
                    i_D += 1
                for m in range(0, ell+1):
                    i_H = WignerHindex(ell, mp, m, mp_max)
                    𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[-mp].conjugate()
                    i_D += 1
            for mp in range(0, ell+1):
                i_D = WignerDindex(ell, mp, -ell, ell_min)
                for m in range(-ell, 0):
                    i_H = WignerHindex(ell, mp, m, mp_max)
                    𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[mp]
                    i_D += 1
                for m in range(0, ell+1):
                    i_H = WignerHindex(ell, mp, m, mp_max)
                    𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[mp]
                    i_D += 1
    
    def _to_euler_phases(R, z):
        """Helper function for `to_euler_phases`"""
        a = R[0]**2 + R[3]**2
        b = R[1]**2 + R[2]**2
        sqrta = np.sqrt(a)
        sqrtb = np.sqrt(b)
        z[1] = ((a - b) + 2j * sqrta * sqrtb) / (a + b)  # exp[iβ]
        if sqrta > 0.0:
            zp = (R[0] + 1j * R[3]) / sqrta  # exp[i(α+γ)/2]
        else:
            zp = 1.0 + 0.0j
        if abs(sqrtb) > 0.0:
            zm = (R[2] - 1j * R[1]) / sqrtb  # exp[i(α-γ)/2]
        else:
            zm = 1.0 +0.0j
        z[0] = zp * zm
        z[2] = zp * zm.conjugate()

    # quaternions = quaternionic.array(R).ndarray.reshape((-1, 4))
    quaternions = rotation_matrix_to_quaternion(
        np.swapaxes(R, -1, -2)
    ).reshape((-1, 4))
    Dsize = WignerDsize(ell_min, mp_max, ell_max)
    Hsize = WignerHsize(mp_max, ell_max)
    function_values = np.zeros(quaternions.shape[:-1] + (Dsize,), dtype=complex)

    n = np.array([n for n in range(ell_max+2) for m in range(-n, n+1)])
    m = np.array([m for n in range(ell_max+2) for m in range(-n, n+1)])
    absn = np.array([n for n in range(ell_max+2) for m in range(n+1)])
    absm = np.array([m for n in range(ell_max+2) for m in range(n+1)])
    _a = np.sqrt((absn+1+absm) * (absn+1-absm) / ((2*absn+1)*(2*absn+3)))
    _b = np.sqrt((n-m-1) * (n-m) / ((2*n-1)*(2*n+1)))
    _b[m<0] *= -1
    _d = 0.5 * np.sqrt((n-m) * (n+m+1))
    _d[m<0] *= -1
    with np.errstate(divide='ignore', invalid='ignore'):
        _g = 2*(m+1) / np.sqrt((n-m)*(n+m+1))
        _h = np.sqrt((n+m+2)*(n-m-1) / ((n-m)*(n+m+1)))
    if not (
        np.all(np.isfinite(_a)) and
        np.all(np.isfinite(_b)) and
        np.all(np.isfinite(_d))
    ):
        raise ValueError("Found a non-finite value inside this object")

    # Loop over all input quaternions
    for i_R in range(quaternions.shape[0]):
        # Init
        Hwedge = np.zeros((Hsize,), dtype=float)
        Hv = np.zeros(((ell_max+1)**2,), dtype=float)
        Hextra = np.zeros((ell_max+2,), dtype=float)
        zₐpowers = np.zeros((ell_max+1), dtype=complex)[np.newaxis]
        zᵧpowers = np.zeros((ell_max+1), dtype=complex)[np.newaxis]
        z = np.zeros((3,), dtype=complex)
        
        _to_euler_phases(quaternions[i_R], z)

        # Compute a quarter of the H matrix
        _step_1(Hwedge)
        _step_2(_g, _h, ell_max, mp_max, Hwedge, Hextra, Hv, z[1])
        _step_3(_a, _b, ell_max, mp_max, Hwedge, Hextra, z[1])
        _step_4(_d, ell_max, mp_max, Hwedge, Hv)
        _step_5(_d, ell_max, mp_max, Hwedge, Hv)

        D = function_values[i_R]
        _complex_powers(z[0:1], ell_max, zₐpowers)
        _complex_powers(z[2:3], ell_max, zᵧpowers)
        _fill_wigner_D(ell_min, ell_max, mp_max, D, Hwedge, zₐpowers[0], zᵧpowers[0])
    return function_values.reshape(R.shape[:-2] + (Dsize,))


def winger_D_multiply_spherical_harmonics(D, y):
    """
    Multiply a Wigner D matrix by a spherical harmonic coefficients.
    """
    output = np.zeros_like(y)
    offset, ls, i = 0, 0, 0
    for i in range(int(math.sqrt(y.shape[-1]))):
        ls = 2*i+1
        offset = ((2*i-1)*i*(4*i+2))//6
        d_part = D[..., offset:offset+ls**2].reshape(D.shape[:-1] + (ls, ls)).T
        offset_sh = i**2
        y_part = y[..., offset_sh:offset_sh+ls]
        output[..., offset_sh:offset_sh+ls] = np.matmul(d_part, y_part[..., None])[..., 0].real
    if (offset+ls**2) != D.shape[-1]:
        raise ValueError(f"The D matrix shape {D.shape[-1]} does not match the expected shape {offset} for the spherical harmonics with rank {i-1}")
    return output


def rotate_spherical_harmonics(R, y):
    """
    Rotate spherical harmonics coefficients by a rotation matrix R.
    """
    D = wigner_D_matrix(R, int(math.sqrt(y.shape[-1]))-1)
    return winger_D_multiply_spherical_harmonics(D, y)
