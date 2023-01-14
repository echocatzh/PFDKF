# Copyright 2022 Shimin Zhang <shmzhang@npu-aslp.org>
# Version: 1.0
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
#
# reference paper: "STATE-SPACE ARCHITECTURE OF THE PARTITIONED-BLOCK-BASED
# ACOUSTIC ECHO CONTROLLER"
# =============================================================================

""" Partitioned-Block-Based Frequency Domain Kalman Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class PFDKF:
    def __init__(self, N, M, A=0.999, P_initial=1, keep_m_gate=0.5, res=True):
        """Initial state of partitioned block based frequency domain kalman filter

        Args:
            N (int): Num of blocks.
            M (int): Filter length in one block.
            A (float, optional): Diag coeff, if more nonlinear components,
            can be set to 0.99. Defaults to 0.999.
            P_initial (int, optional): About the begining covergence. 
            Defaults to 10.
            keep_m_gate (float, optional): When more linear,
            can be set to 0.2 or less. Defaults to 0.5.
        """
        # M = 2*V
        self.N = N
        self.M = M
        self.A = A
        self.A2 = A**2
        self.m_smooth_factor = keep_m_gate
        self.res = res

        self.x = np.zeros(shape=(2*self.M), dtype=np.float32)
        self.m = np.zeros(shape=(self.M + 1), dtype=np.float32)
        self.P = np.full((self.N, self.M + 1), P_initial)
        self.X = np.zeros((self.N, self.M + 1), dtype=complex)
        self.H = np.zeros((self.N, self.M + 1), dtype=complex)
        self.mu = np.zeros((self.N, self.M + 1), dtype=complex)
        self.half_window = np.concatenate(([1]*self.M, [0]*self.M))

    def filt(self, x, d):
        assert(len(x) == self.M)
        self.x = np.concatenate([self.x[self.M:], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        Y = np.sum(self.H*self.X, axis=0)
        y = ifft(Y).real[self.M:]
        e = d-y

        e_fft = np.concatenate(
            (np.zeros(shape=(self.M,), dtype=np.float32), e))
        self.E = fft(e_fft)
        self.m = self.m_smooth_factor * self.m + \
            (1-self.m_smooth_factor) * np.abs(self.E)**2
        R = np.sum(self.X*self.P*self.X.conj(), 0) + 2*self.m/self.N
        self.mu = self.P / (R + 1e-10)
        if self.res:
            W = 1 - np.sum(self.mu*np.abs(self.X)**2, 0)
            E_res = W*self.E
            e = ifft(E_res).real[self.M:].real
            y = d-e
        return e, y

    def update(self):
        G = self.mu*self.X.conj()
        self.P = self.A2*(1 - 0.5*G*self.X)*self.P + \
            (1-self.A2)*np.abs(self.H)**2
        self.H = self.A*(self.H + fft(self.half_window*(ifft(self.E*G).real)))


def pfdkf(x, d, N=10, M=256, A=0.999, P_initial=1, keep_m_gate=0.1):
    ft = PFDKF(N, M, A, P_initial, keep_m_gate)
    num_block = min(len(x), len(d)) // M

    e = np.zeros(num_block*M)
    y = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()
        e[n*M:(n+1)*M] = e_n
        y[n*M:(n+1)*M] = y_n
    return e, y


if __name__ == "__main__":
    import soundfile as sf
    mic, sr = sf.read("d.wav")
    ref, sr = sf.read("u.wav")
    # for linear senario:
    # error, echo = pfdkf(ref, mic, A=0.999, keep_m_gate=0.2)
    # for non-linear senario:
    error, echo = pfdkf(ref, mic, A=0.999, keep_m_gate=0.5)
    sf.write("out_kalman.wav", error, sr)
    sf.write("out_echo.wav", echo, sr)
