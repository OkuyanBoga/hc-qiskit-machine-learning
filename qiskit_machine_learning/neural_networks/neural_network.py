# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Neural Network abstract class for all (quantum) neural networks within Qiskit
Machine Learning module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.transpiler.passmanager import BasePassManager

import qiskit_machine_learning.optionals as _optionals
from ..exceptions import QiskitMachineLearningError

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


class NeuralNetwork(ABC):
    """Abstract Neural Network class providing forward and backward pass and handling
    batched inputs. This is to be implemented by other (quantum) neural networks.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        num_inputs: int,
        num_weights: int,
        sparse: bool,
        output_shape: int | tuple[int, ...],
        input_gradients: bool = False,
        pass_manager: BasePassManager | None = None,
    ) -> None:
        """
        Args:
            num_inputs: The number of input features.
            num_weights: The number of trainable weights.
            sparse: Determines whether the output is a sparse array or not.
            output_shape: The shape of the output.
            input_gradients: Determines whether to compute gradients with respect to input data.
            pass_manager: The pass manager to transpile the circuits, if necessary.
                Defaults to ``None``, as some primitives do not need transpiled circuits.
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        if num_inputs < 0:
            raise QiskitMachineLearningError(f"Number of inputs cannot be negative: {num_inputs}!")
        self._num_inputs = num_inputs
        if num_weights < 0:
            raise QiskitMachineLearningError(
                f"Number of weights cannot be negative: {num_weights}!"
            )
        self._num_weights = num_weights

        self._sparse = sparse
        self.pass_manager = pass_manager

        # output shape may be derived later, so check it only if it is not None
        if output_shape is not None:
            self._output_shape = self._validate_output_shape(output_shape)

        self._input_gradients = input_gradients

    @property
    def num_inputs(self) -> int:
        """Returns the number of input features."""
        return self._num_inputs

    @property
    def num_weights(self) -> int:
        """Returns the number of trainable weights."""
        return self._num_weights

    @property
    def sparse(self) -> bool:
        """Returns whether the output is sparse or not."""
        return self._sparse

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Returns the output shape."""
        return self._output_shape

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

    def _validate_output_shape(self, output_shape):
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if not np.all([s > 0 for s in output_shape]):
            raise QiskitMachineLearningError(
                f"Invalid output shape, all components must be > 0, but got: {output_shape}."
            )
        return output_shape

    def _validate_input(
        self,
        input_data: float | QuantumCircuit | list[QuantumCircuit] | list[float] | np.ndarray | None,
        input_params: float | list[float] | np.ndarray | None = None,
    ) -> tuple[
        np.ndarray | QuantumCircuit | list[QuantumCircuit] | None,
        tuple[int, ...] | None,
        np.ndarray | None,
    ]:
        if input_data is None:
            return None, None, None
        elif isinstance(input_data, QuantumCircuit):
            if input_params is None:
                return input_data, (len(input_data),), input_params
            else:
                input_ = np.array(input_params)
                input_ = input_.reshape((1, 1))
                return [input_data], input_.shape, input_
        if type(input_data) is list:
            if isinstance(input_data[0], QuantumCircuit):
                if input_params is None:
                    return input_data, (len(input_data),), input_params
                else:
                    input_ = np.array(input_params)
            else:
                input_ = np.array(input_data)
        else:
            input_ = np.array(input_data)
        shape = input_.shape
        if len(shape) == 0:
            # there's a single value in the input.
            input_ = input_.reshape((1, 1))
            return input_, shape, input_params

        if shape[-1] != self._num_inputs:
            raise QiskitMachineLearningError(
                f"Input data has incorrect shape, last dimension "
                f"is not equal to the number of inputs: "
                f"{self._num_inputs}, but got: {shape[-1]}."
            )

        if len(shape) == 1:
            # add an empty dimension for samples (batch dimension)
            input_ = input_.reshape((1, -1))
        elif len(shape) > 2:
            # flatten lower dimensions, keep num_inputs as a last dimension
            input_ = input_.reshape((np.prod(input_.shape[:-1]), -1))

        if isinstance(input_data[0], QuantumCircuit):
            return input_data, shape, input_

        return input_, shape, input_params

    def _compose_circs(
        self, input_circuit: QuantumCircuit, ansatz: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose input circuits with given ansatz."""
        if hasattr(input_circuit.layout, "_input_qubit_count"):
            if hasattr(ansatz.layout, "_input_qubit_count"):
                if ansatz.num_qubits != input_circuit.num_qubits:
                    raise QiskitMachineLearningError(
                        "Transpiled circuits need to have same number of qubits."
                    )
                circuit_ = input_circuit.compose(ansatz)
            else:
                if self.pass_manager is not None:
                    _isa_ansatz = self.pass_manager.run(ansatz)
                    circuit_ = input_circuit.compose(_isa_ansatz)
                else:
                    raise QiskitMachineLearningError(
                        "Both input circuits needs to be same type or "
                        "please provide a pass manager."
                    )
        else:
            if hasattr(ansatz.layout, "_input_qubit_count"):
                if self.pass_manager is not None:
                    _isa_fm = self.pass_manager.run(input_circuit)
                    if ansatz.num_qubits != _isa_fm.num_qubits:
                        raise QiskitMachineLearningError(
                            "Transpiled circuits need to have same number of qubits."
                            "Please ensure if you are using the same pass manager."
                        )
                    circuit_ = _isa_fm.compose(ansatz)
                else:
                    raise QiskitMachineLearningError(
                        "Both input circuits needs to be same type or "
                        "please provide a pass manager."
                    )
            else:
                circuit_ = QuantumCircuit(ansatz.num_qubits)
                circuit_.compose(input_circuit, inplace=True)
                circuit_.compose(ansatz, inplace=True)
        return circuit_

    def _preprocess_input(
        self,
        input_data: np.ndarray | list[QuantumCircuit] | QuantumCircuit | None,
        weights: np.ndarray | None,
        input_params: np.ndarray | None,
        ansatz: QuantumCircuit | None,
        output_shape: int = 1,
    ) -> tuple[list[QuantumCircuit] | None, np.ndarray | None, int | None, bool]:
        """
        Pre-processing input data of the network for the primitive-based networks.
        """
        is_circ_input = False
        if input_data is None:
            parameter_values, num_samples = self._preprocess_forward(input_data, weights)
            _circuits = [ansatz] * output_shape * num_samples
        else:
            if isinstance(input_data, QuantumCircuit):
                num_samples = 1
                _circuits = [self._compose_circs(input_data, ansatz)] * output_shape
                parameter_values, _ = self._preprocess_forward(input_params, weights)
                is_circ_input = True
            elif isinstance(input_data[0], QuantumCircuit):
                num_samples = len(input_data)
                _circuits = [self._compose_circs(x, ansatz) for x in input_data] * output_shape
                parameter_values, _ = self._preprocess_forward(input_params, weights)
                is_circ_input = True
            else:
                parameter_values, num_samples = self._preprocess_forward(input_data, weights)
                _circuits = [ansatz] * output_shape * num_samples
        return _circuits, parameter_values, num_samples, is_circ_input

    def _preprocess_forward(
        self,
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int | None]:
        """
        Pre-processing during forward pass of the network for the primitive-based networks.
        """
        if input_data is not None:
            num_samples = input_data.shape[0]
            if weights is not None:
                weights = np.broadcast_to(weights, (num_samples, len(weights)))
                parameters = np.concatenate((input_data, weights), axis=1)
            else:
                parameters = input_data
        else:
            if weights is not None:
                num_samples = 1
                parameters = np.broadcast_to(weights, (num_samples, len(weights)))
            else:
                # no input, no weights, just execute circuit once
                num_samples = 1
                parameters = np.asarray([])
        return parameters, num_samples

    def _validate_weights(
        self, weights: float | list[float] | np.ndarray | None
    ) -> np.ndarray | None:
        if weights is None:
            return None
        weights_ = np.array(weights)
        return weights_.reshape(self._num_weights)

    def _validate_forward_output(
        self, output_data: np.ndarray, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        if original_shape and len(original_shape) >= 2:
            output_data = output_data.reshape((*original_shape[:-1], *self._output_shape))

        return output_data

    def _validate_backward_output(
        self,
        input_grad: np.ndarray,
        weight_grad: np.ndarray,
        original_shape: tuple[int, ...],
    ) -> tuple[np.ndarray | SparseArray, np.ndarray | SparseArray]:
        if input_grad is not None and np.prod(input_grad.shape) == 0:
            input_grad = None
        if input_grad is not None and original_shape and len(original_shape) >= 2:
            input_grad = input_grad.reshape(
                (*original_shape[:-1], *self._output_shape, self._num_inputs)
            )
        if weight_grad is not None and np.prod(weight_grad.shape) == 0:
            weight_grad = None
        if weight_grad is not None and original_shape and len(original_shape) >= 2:
            weight_grad = weight_grad.reshape(
                (*original_shape[:-1], *self._output_shape, self._num_weights)
            )

        return input_grad, weight_grad

    def forward(
        self,
        input_data: float | list[float] | np.ndarray | QuantumCircuit | list[QuantumCircuit] | None,
        weights: float | list[float] | np.ndarray | None,
        input_params: float | list[float] | np.ndarray | None = None,
    ) -> np.ndarray | SparseArray:
        """Forward pass of the network.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array. If the data is set of circuits
                either bind the input parameters to circuits or use input_params.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
            input_params: Input parameters for each circuit given as data. Only use when input is circuit.
                Defaults to `None`.
        Returns:
            The result of the neural network of the shape (output_shape).
        """
        input_, shape, input_params_ = self._validate_input(input_data, input_params)
        weights_ = self._validate_weights(weights)
        output_data = self._forward(input_, weights_, input_params_)
        return self._validate_forward_output(output_data, shape)

    @abstractmethod
    def _forward(
        self,
        input_data: np.ndarray | QuantumCircuit | list[QuantumCircuit] | None,
        weights: np.ndarray | None,
        input_params: np.ndarray | None = None,
    ) -> np.ndarray | SparseArray:
        raise NotImplementedError

    def backward(
        self,
        input_data: float | list[float] | np.ndarray | QuantumCircuit | list[QuantumCircuit] | None,
        weights: float | list[float] | np.ndarray | None,
        input_params: float | list[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray | None]:
        """Backward pass of the network.

        Args:
            input_data: input data of the shape (num_inputs). In case of a
                single scalar input it is directly cast to and interpreted like a one-element array.
                If the data is set of circuits either bind the input parameters to circuits
                or use input_params.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
            it is directly cast to and interpreted like a one-element array.
            input_params: Input parameters for each circuit given as data. Only use when input is circuit.
                Defaults to `None`.
        Returns:
            The result of the neural network of the backward pass, i.e., a tuple with the gradients
            for input and weights of shape (output_shape, num_input) and
            (output_shape, num_weights), respectively.
        """
        input_, shape, input_params_ = self._validate_input(input_data, input_params)
        weights_ = self._validate_weights(weights)
        input_grad, weight_grad = self._backward(input_, weights_, input_params_)

        input_grad_reshaped, weight_grad_reshaped = self._validate_backward_output(
            input_grad, weight_grad, shape
        )

        return input_grad_reshaped, weight_grad_reshaped

    @abstractmethod
    def _backward(
        self,
        input_data: np.ndarray | QuantumCircuit | list[QuantumCircuit] | None,
        weights: np.ndarray | None,
        input_params: np.ndarray | None = None,
    ) -> tuple[np.ndarray | SparseArray | None, np.ndarray | SparseArray | None]:
        raise NotImplementedError

    def _reparameterize_circuit(
        self,
        circuit: QuantumCircuit,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
    ) -> QuantumCircuit:
        # As the data (parameter values) for the primitive is ordered as inputs followed by weights
        # we need to ensure that the parameters are ordered like this naturally too so the rewrites
        # parameters to ensure this. "inputs" as a name comes before "weights" and within they are
        # numerically ordered.
        if input_params and self.num_inputs != len(input_params):
            raise ValueError(
                f"input_params length {len(input_params)}"
                f" mismatch with num_inputs (self.num_inputs)"
            )
        if weight_params and self.num_weights != len(weight_params):
            raise ValueError(
                f"weight_params length {len(weight_params)}"
                f" mismatch with num_weights (self.num_weights)"
            )

        parameters = circuit.parameters

        if len(parameters) != (self.num_inputs + self.num_weights):
            raise ValueError(
                f"Number of circuit parameters ({len(parameters)})"
                f" does not match the sum of number of inputs  and weights"
                f" ({self.num_inputs + self.num_weights})."
            )

        new_input_params = ParameterVector("inputs", self.num_inputs)
        new_weight_params = ParameterVector("weights", self.num_weights)

        new_parameters = {}
        if input_params:
            for i, param in enumerate(input_params):
                if param not in parameters:
                    raise ValueError(f"Input param `{param.name}` not present in circuit")
                new_parameters[param] = new_input_params[i]

        if weight_params:
            for i, param in enumerate(weight_params):
                if param not in parameters:
                    raise ValueError(f"Weight param {param.name} `not present in circuit")
                new_parameters[param] = new_weight_params[i]

        if new_parameters:
            circuit = circuit.assign_parameters(new_parameters)

        return circuit
