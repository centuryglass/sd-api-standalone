"""Abstract interface for all ComfyUI workflow nodes."""
from copy import deepcopy
from typing import TypeAlias, Any
from typing_extensions import TypedDict

NodeId: TypeAlias = str  # Should be an integer string
SlotIndex: TypeAlias = int  # connection index within the connected node, not the current one.
NodeConnection: TypeAlias = tuple[NodeId, SlotIndex]


class NodeDict(TypedDict):
    """Generic node data format."""
    class_type: str
    inputs: dict[str, Any]


class ComfyNode:
    """Abstract Node class."""

    def __init__(self, class_type: str, input_data: dict[str, Any], node_input_keys: set[str],
                 output_count: int) -> None:
        self._class_type = class_type
        self._inputs = input_data
        self._node_input_keys = node_input_keys
        self._output_count = output_count

    def __deepcopy__(self, memo: dict[int, Any]) -> 'ComfyNode':
        # Reconstruct as the concrete subclass without re-invoking __init__ (subclass constructors have varied
        # signatures). Copying every instance attribute preserves subclass-specific state and, critically, the subclass
        # type itself, so overrides like UltimateUpscaleNode.add_input survive a graph deepcopy.
        cls = self.__class__
        node_copy = cls.__new__(cls)
        memo[id(self)] = node_copy
        for key, value in self.__dict__.items():
            setattr(node_copy, key, deepcopy(value, memo))
        return node_copy

    @property
    def node_name(self) -> str:
        """Returns the node's type name used in the API."""
        return self._class_type

    def clear_connections(self) -> None:
        """Removes all inputs from other nodes."""
        for input_key in self._node_input_keys:
            if input_key in self._inputs:
                del self._inputs[input_key]

    @property
    def output_count(self) -> int:
        """Returns the number of outputs the node provides."""
        return self._output_count

    def add_input(self, connected_node: str, output_slot_index: int, input_key: str):
        """Connect one of this node's inputs to another node's output.

        This will check the validity of the input key, but doesn't do anything to validate that the output is correct.
        """
        if input_key not in self._node_input_keys:
            raise ValueError(f'Unexpected input key {input_key}')
        self._inputs[input_key] = (connected_node, output_slot_index)

    def get_dict(self) -> NodeDict:
        """Returns the node API dict."""
        return {
            'class_type': self._class_type,
            'inputs': deepcopy(self._inputs)
        }
