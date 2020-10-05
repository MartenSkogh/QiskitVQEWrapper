import sys
import numpy as np
import scipy as sp
import re
from copy import deepcopy
from pprint import pprint
from timeit import default_timer as timer
from enum import Enum

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.algorithms.minimum_eigen_solvers import VQE
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.components.optimizers import SLSQP, L_BFGS_B, COBYLA, SPSA
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, GaussianDriver, UnitsType, HFMethodType
from qiskit.chemistry.components.variational_forms import UCCSD 
from qiskit.chemistry.components.initial_states import HartreeFock

class DriverType(Enum):
    """ DriverType Enum """
    PYSCF = 'PySCF'
    GAUSSIAN = 'Gaussian'

class VQEWrapper():
    
    def __init__(self):
        # These things need to be set before running
        self.molecule_string = None
        # You can make a pretty educated guess for these two
        self.spin = None
        self.charge = None

        self.qmolecule = None

        self.length_unit = UnitsType.ANGSTROM
        #Basis has to be in a format accepted by Gaussian (sto-3g, 6-31g)
        self.basis = 'sto-3g'
        self.hf_method = HFMethodType.UHF

        self.chem_driver = DriverType.GAUSSIAN
        self.driver = None
        self.core = None

        self.transformation = TransformationType.FULL
        self.qubit_mapping = QubitMappingType.JORDAN_WIGNER
        self.two_qubit_reduction = False
        self.freeze_core = True
        self.orbital_reduction = []

        #Parallelization threshold in number of qubits (14 if not specified)
        self.par_thres = 14

        self.qubit_op = None
        self.aux_ops = None
        self.initial_point = None

        self.optimizer = SLSQP(maxiter=5000)

        # Choose the backend (use Aer instead of BasicAer) 
        self.backend = Aer.get_backend('statevector_simulator') 
        self.quantum_instance = QuantumInstance(backend=self.backend,
                                                backend_options = {'statevector_parallel_threshold':self.par_thres})

        self.vqe_algo = None

        self.vqe_callback = None
        self.vqe_time = None

    def opt_str(self):
        match = re.search(r'optimizers.[A-z]+.(.+) object', str(self.optimizer))
        opt_str = match.group(1)
        return opt_str

    def gaussian_config(self):
        #Formats properties to a string fitting the gaussian input format
        gaussian_config = f'# {self.hf_method.value}/{self.basis} scf(conventional)\n\nMolecule\n\n{self.charge} {self.spin+1}\n'
        gaussian_config = gaussian_config + self.molecule_string.replace('; ','\n') + '\n\n'
        return gaussian_config

    def initiate(self):
        #print("Setting up system:")
        #print(f"  Molecule: {self.molecule_string}")
        #print(f"  Charge: {self.charge}")
        #print(f"  Spin: {self.spin}")
        if self.chem_driver.value == 'PySCF':
            self.driver = PySCFDriver(atom=self.molecule_string, 
                                      unit=self.length_unit, 
                                      charge=self.charge,
                                      spin=self.spin,
                                      hf_method=self.hf_method,
                                      basis=self.basis)

        elif self.chem_driver.value == 'Gaussian':
            self.driver = GaussianDriver(config=self.gaussian_config())

        self.qmolecule = self.driver.run()

        self.core = Hamiltonian(transformation=self.transformation, 
                           qubit_mapping=self.qubit_mapping, 
                           two_qubit_reduction=self.two_qubit_reduction, 
                           freeze_core=self.freeze_core, 
                           orbital_reduction=self.orbital_reduction)

        self.qubit_op, self.aux_ops = self.core.run(self.qmolecule)

        # initial state
        self.init_state = HartreeFock(num_orbitals=self.core._molecule_info['num_orbitals'], 
                                      qubit_mapping=self.core._qubit_mapping,
                                      two_qubit_reduction=self.core._two_qubit_reduction, 
                                      num_particles=self.core._molecule_info['num_particles'])

        # UCCSD Ansatz
        self.var_form = UCCSD(num_orbitals=self.core._molecule_info['num_orbitals'], 
                              num_particles=self.core._molecule_info['num_particles'], 
                              initial_state=self.init_state, 
                              qubit_mapping=self.core._qubit_mapping, 
                              two_qubit_reduction=self.core._two_qubit_reduction, 
                              num_time_slices=1, 
                              shallow_circuit_concat=False)
        
        self.init_vqe()

    #set up VQE
    def init_vqe(self):
        self.vqe_algo = VQE(self.qubit_op, 
                            self.var_form, 
                            self.optimizer, 
                            initial_point=self.initial_point, 
                            callback=self.vqe_callback)


    def run_vqe(self):
        # run the algorithm
        vqe_start = timer()
        self.vqe_result = self.vqe_algo.run(self.quantum_instance)
        self.vqe_time = timer() - vqe_start

        # get the results
        result = self.core.process_algorithm_result(self.vqe_result) 

        return result
