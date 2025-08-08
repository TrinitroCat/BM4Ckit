""" Molecular Dynamics via Verlet algo. """
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseMD.py
#  Environment: Python 3.12

import sys
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import logging

import torch as th
from torch import nn

import numpy as np

from BM4Ckit.utils._Element_info import MASS, N_MASS, ATOMIC_NUMBER
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BM4Ckit.utils.scatter_reduce import scatter_reduce


class _BaseMD:
    """ Base BatchMD """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        """
        Parameters: 
            time_step: float, time per step (ps).
            max_step: maximum steps.
            T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: device that program run on.
            verbose: control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
        """
        self.time_step = time_step
        assert (max_step > 0) and isinstance(max_step, int), f'max_step must be a positive integer, but occurred {max_step}.'
        self.max_step = max_step
        self.T_init = T_init
        self.output_file = output_file
        self.output_structures_per_step = output_structures_per_step
        self.device = device
        self.verbose = verbose

        self.EK_TARGET = None  # target kinetic energy under set temperature.
        self.Ekt_vir = None    # virtual kinetic energy for _CSVR_ thermostat.
        self.Ek = None         # kinetic energy at each timestep.
        self.p_iota = None       # thermostat var. for Nose-Hoover.

        self.batch_tensor = None  # tensor form of `batch_indices` if it was given.
        self.batch_scatter = None # tensor indices form of `batch_indices` if it was given
                                  # e.g., batch_indices = (3, 2, 1), thus self.scatter = tensor([0, 0, 0, 1, 1, 2])

        # logging
        self.logger = logging.getLogger('Main.MD')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            if (output_file is None) or (output_file == ''):
                log_handler = logging.StreamHandler(sys.stdout)
                log_handler.setLevel(logging.INFO)
                log_handler.setFormatter(formatter)
            else:
                log_handler = logging.FileHandler(output_file, 'w')
                log_handler.setLevel(logging.INFO)
                log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    def _reduce_Ek_T(self, batch_indices, masses, V, n_atom, n_dim):
        if batch_indices is not None:
            Ek = th.sum(
                0.5 * scatter_reduce(
                    masses * V ** 2,
                    self.batch_scatter,
                    1
                ) * 103.642696562621738,
                dim=-1
            )  # (n_batch, ), eV/atom. Faraday constant F = 96485.3321233100184.

            temperature = (2 * Ek) / (n_dim * (self.batch_tensor - 1 + 1e-20) * 8.617333262145e-5)
        else:
            Ek = 0.5 * th.sum(
                masses * V ** 2,
                dim=(-2, -1)
            ) * 103.642696562621738  # (n_batch, ), eV/atom. Faraday constant F = 96485.3321233100184.
            temperature = (2 * Ek) / (
                    n_dim * (n_atom - 1 + 1e-20) * 8.617333262145e-5
            )  # Boltzmann constant kB = 8.617333262145e-5 eV/K

        return Ek, temperature

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            is_fix_mass_center: bool = False
    ) -> None:
        """

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func. If a 2D X was given, the first dimension would be set to 1.
            Element_list: List[List[str | int]], the atomic type (element) corresponding to each row of each batch in X.
            V_init: the initial velocities of each atom. If None, a random velocity generated by Boltzmann distribution would be set.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X i.e., grad = grad_func(X, y, ...), else grad = grad_func(X, ...)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            batch_indices: the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            fixed_atom_tensor: the indices of X that fixed.
            is_fix_mass_center: whether transition coordinates and velocities of atoms into the mass center.

        Returns:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.

        """
        t_main = time.perf_counter()
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        elif len(X.shape) != 3:
            raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
        n_batch, n_atom, n_dim = X.shape
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        # Check batch indices
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                self.batch_tensor = batch_indices
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)  # the tensor version of batch_indices which is a List.
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.

        # Manage Atomic Type & Masses
        masses = list()
        for _Elem in Element_list:
            masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
        masses = th.tensor(masses, dtype=th.float32, device=self.device)
        masses = masses.unsqueeze(-1).expand_as(X)  # (n_batch, n_atom, n_dim)
        # grad_func
        if grad_func is None:
            is_grad_func_contain_y = True
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
        # Selective dynamics  TODO: fixed atom is not compatible with freedom degree. FIXME !!!
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        # target kinetic energy for NVT|NPT ensembles
        if batch_indices:
            self.EK_TARGET = th.tensor(
                [((n_dim / 2.) * (_n_atom - 1) * 8.617333262145e-5 * self.T_init) for _n_atom in batch_indices],
                dtype=X.dtype,
                device=self.device
            )
        else:
            self.EK_TARGET = (n_dim / 2.) * (n_atom - 1) * 8.617333262145e-5 * self.T_init  # Unit: eV/atom. Boltzmann constant kB = 8.6173332621e-5 eV/K
        # other check
        if (not isinstance(self.max_step, int)) or (self.max_step <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.max_step}. It would be an integer greater than 0.')

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.eval()
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        # Generate initial Velocities
        if V_init is not None:
            if V_init.shape != X.shape:
                raise ValueError(f'V_init and X must have the same shape, but got {V_init.shape} and {X.shape}')
            if self.verbose > 0: self.logger.info('Initial velocities was given, T_init will be ignored.')
            V_init = V_init.to(self.device)
            V = V_init.detach() * atom_masks
            V = V.to(self.device)
        else:
            V = th.normal(
                0.,
                th.sqrt(self.T_init * 8314.46261815324 / masses) * 1.e-5,  # Unit: Ang/fs, R = kB * L = 8.31446261815324 J/(mol * K)
            ) * atom_masks
        # split by batch_indices
        if batch_indices is not None:
            masses_tup = th.split(masses, batch_indices, dim=1)
            V_tup = th.split(V, batch_indices, dim=1)
            masses_sum = th.zeros((1, len(batch_indices), n_dim), device=self.device).scatter_reduce_(
                1,
                self.batch_scatter.view(1, -1, 1).expand_as(masses),
                masses,
                'sum'
            )[:, self.batch_scatter, :]  # (1, n_batch*n_atom, n_dim)
        else:
            masses_tup = (masses,)
            V_tup = (V,)
            masses_sum = th.sum(masses, dim=1, keepdim=True)  # (n_batch, 1, n_dim)

        # initialize thermostat parameters
        # (n_batch, ), eV/atom. The initial virtual Ek_t for CSVR.
        self.Ekt_vir = th.cat(
            [0.5 * th.sum(_m * V_tup[_] ** 2, dim=(-2, -1)) * 103.642696562621738 for _, _m in enumerate(masses_tup)]
        )
        # The initial iota for Nose-Hoover
        if batch_indices is not None:
            self.p_iota = th.zeros(1, len(batch_indices), device=self.device, dtype=th.float32)
        else:
            self.p_iota = 0.
        # whether grad needs autograd
        self.require_grad = require_grad

        # print Atoms Information
        if self.verbose > 0:
            elem_list = list()
            _element_list = list()
            if batch_indices is not None:
                indx_old = 0
                for indx in batch_indices:
                    _element_list.append(Element_list[0][indx_old: indx_old + indx])
                    indx_old += indx
            else:
                _element_list = Element_list
            for elements in _element_list:
                __element_now = ''
                __elem = ''
                elem_info = ''
                __elem_count = ''
                for i, elem in enumerate(elements, 1):
                    # get element symbol
                    if isinstance(elem, int):
                        __elem = ATOMIC_NUMBER[elem]
                    else:
                        __elem = elem
                    # count element number
                    if __elem == __element_now:
                        __elem_count += 1
                    else:
                        elem_info = elem_info + str(__elem_count) + '  '
                        elem_info = elem_info + __elem + ': '
                        __elem_count = 1
                        __element_now = __elem
                elem_info = elem_info + str(__elem_count)
                elem_list.append(elem_info)
            # log out
            for i, ee in enumerate(elem_list):
                self.logger.info(f'Structure {i:>5d}: {ee}')

        # MAIN Loop
        with th.no_grad():
            t_in = time.perf_counter()
            with th.set_grad_enabled(require_grad):
                X.requires_grad_(require_grad)
                Energy = func(X, *func_args, **func_kwargs)  # Note: func must return th.Tensor(n_batch, )
                if batch_indices is not None:
                    if len(Energy) != len(batch_indices):  # check batch number of output
                        raise RuntimeError(
                            f'batch number of `func` output ({len(Energy)}) does not match the input `batch_indices` ({len(batch_indices)}'
                        )

                if is_grad_func_contain_y:
                    Forces = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Forces = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
            temperature_ = th.empty(len(V_tup), 1, device=self.device, dtype=Energy.dtype)
            Ek_ = th.empty(len(V_tup), 1, device=self.device, dtype=th.float32)

            #ptlist = list()  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            for i in range(self.max_step):
                # Print & Output file
                if batch_indices is not None:
                    X_tup = th.split(X, batch_indices, dim=1)
                    masses_tup = th.split(masses, batch_indices, dim=1)
                    V_tup = th.split(V, batch_indices, dim=1)
                else:
                    X_tup = (X,)
                    masses_tup = (masses,)
                    V_tup = (V,)
                # update Ek and temperature  TODO: optimize by scatter ops.
                for _i, __x in enumerate(V_tup):
                    Ek_[_i] = 0.5 * th.sum(
                        masses_tup[_i] * __x ** 2,
                        dim=(-2, -1)
                    ) * 103.642696562621738  # (n_batch, ), eV/atom. Faraday constant F = 96485.3321233100184.
                    temperature_[_i] = (2 * Ek_[_i]) / (
                            n_dim * (__x.shape[1] - 1 + 1e-20) * 8.617333262145e-5
                    )  # Boltzmann constant kB = 8.617333262145e-5 eV/K
                Ek, temperature = self._reduce_Ek_T(batch_indices, masses, V, n_atom, n_dim)
                self.Ek = Ek.squeeze()  # th.sum(Ek, dim=0)  # saving the real kinetic energy for VR & CSVR to avoid double counting.
                # if self.verbose > 0:
                if i % self.output_structures_per_step == 0:
                    if self.verbose > 1:
                        np.set_printoptions(
                            precision=8, floatmode='fixed', suppress=True, formatter={'float': '{:> 5.10f}'.format}, threshold=3000000
                        )
                        self.logger.info('_' * 100)
                        self.logger.info(f'Configuration {i}:')
                        for __x in X_tup:
                            X_str = np.array2string(
                                __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                            ).replace("[", " ").replace("]", " ")
                            self.logger.info(f'{X_str}\n')
                        del X_str, X_tup
                        if self.verbose > 2:
                            self.logger.info(f'Velocities {i}:')
                            for __x in V_tup:
                                V_str = np.array2string(
                                    __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                                ).replace("[", " ").replace("]", " ")
                                self.logger.info(f'{V_str}\n')
                            del V_str
                        self.logger.info('_' * 100)
                    #ptlist.append(X.numpy(force=True))  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # Calc. Kinetic Energy, Temperature, etc.
                    if self.verbose > 0:
                        np.set_printoptions(
                            precision = 8,
                            linewidth = 1024,
                            floatmode = 'fixed',
                            suppress = True,
                            formatter = {'float': '{:> .2f}'.format},
                            threshold = 2000
                        )
                        self.logger.info(
                            f'Step: {i:>12d}\n\t'
                            f'T     = {temperature.squeeze().numpy(force=True)}\n\t'
                            f'E_tol = {np.array2string((Ek.squeeze() + Energy.squeeze()).numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                            f'Ek    = {np.array2string(Ek.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                            f'Ep    = {np.array2string(Energy.squeeze().squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                            f'Time: {time.perf_counter() - t_in:>5.4f}'
                        )
                        t_in = time.perf_counter()

                # Update X, V
                X, V, Energy, Forces = self._updateXV(
                    X, V, Forces,
                    func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                    masses, atom_masks, is_grad_func_contain_y, batch_indices
                )
                # Correct barycentric transition  TODO: solve the batch_indices problem
                if is_fix_mass_center:
                    if batch_indices is None:
                        X = X - th.sum(masses * X, dim=1, keepdim=True)/masses_sum  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)
                        V = V - th.sum(masses * V, dim=1, keepdim=True)/masses_sum
                    else:                                                           # (1, nb * n_atom, n_dim) - (1, nb * n_atom, n_dim)
                        X = X - th.zeros(1, len(batch_indices), n_dim, device=self.device).scatter_reduce_(
                            1,
                            self.batch_scatter.view(1, -1, 1).expand_as(masses),
                            (masses * X),
                            'sum'
                        )[:, self.batch_scatter, :]/masses_sum
                        V = V - th.zeros(1, len(batch_indices), n_dim, device=self.device).scatter_reduce_(
                            1,
                            self.batch_scatter.view(1, -1, 1).expand_as(masses),
                            (masses * V),
                            'sum'
                        )[:, self.batch_scatter, :]/masses_sum
                        #raise NotImplementedError

            # Finale step
            if self.verbose > 0:
                # if self.verbose > 0:
                if batch_indices is not None:
                    X_tup = th.split(X, batch_indices, dim=1)
                    masses_tup = th.split(masses, batch_indices, dim=1)
                    V_tup = th.split(V, batch_indices, dim=1)
                else:
                    X_tup = (X,)
                    masses_tup = (masses,)
                    V_tup = (V,)
                if self.verbose > 1:
                    np.set_printoptions(
                        precision=8, linewidth=120, floatmode='fixed', suppress=True, formatter={'float': '{:> 5.10f}'.format}, threshold=3000000
                    )
                    self.logger.info('_' * 100)
                    self.logger.info(f'Finale Configuration:')
                    for __x in X_tup:
                        X_str = np.array2string(__x.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        self.logger.info(f'{X_str}\n')
                    del X_str, X_tup
                    if self.verbose > 2:
                        self.logger.info(f'Finale Velocities:')
                        for __x in V_tup:
                            V_str = np.array2string(__x.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                            self.logger.info(f'{V_str}\n')
                        del V_str
                    self.logger.info('_' * 100)
                th.cuda.synchronize()
                Ek, temperature = self._reduce_Ek_T(batch_indices, masses, V, n_atom, n_dim)
                self.Ek = Ek #th.sum(Ek, dim=0)  # saving the real kinetic energy for VR & CSVR to avoid double counting.
                self.logger.info(
                    f'Step: {i:>12d}\n\t'
                    f'T     = {temperature.numpy(force=True)}\n\t'
                    f'E_tol = {np.array2string((Ek.squeeze() + Energy.squeeze()).numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                    f'Ek    = {np.array2string(Ek.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                    f'Ep    = {np.array2string(Energy.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                    f'Time: {time.perf_counter() - t_in:>5.4f}'
                )
                self.logger.info('_' * 100 + '\n' + f'Main Loop Done. Total Time: {time.perf_counter() - t_main:>.4f}')

        del self.Ekt_vir
        #return ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # OVERRIDE THIS METHOD TO IMPLEMENT BatchMD UNDER VARIOUS ENSEMBLES.
    def _updateXV(
            self,
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
            batch_indices
    ) -> (th.Tensor, th.Tensor, th.Tensor):
        """ Update X, V, Force, and return X, V, Energy, Force. """
        raise NotImplementedError
        # return X, V, Energy, Force
