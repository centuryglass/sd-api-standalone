"""Base model for ComfyUI workflow nodes.

Each `ComfyNode` subclass is a pydantic model for one ComfyUI node type. Its fields are the node's inputs under their
ComfyUI names, and its `Output` attributes name the node's output slots. Connection inputs hold the `NodeOutput` they
read from. `ComfyNodeGraph.get_workflow_dict` assigns node keys and turns those into `(node_key, slot)` pairs.
"""
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeAlias, overload

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

NodeId: TypeAlias = str  # Should be an integer string
SlotIndex: TypeAlias = int  # connection index within the connected node, not the current one.
NodeConnection: TypeAlias = tuple[NodeId, SlotIndex]


class NodeDict(TypedDict):
    """Generic node data format."""
    class_type: str
    inputs: dict[str, Any]


class NodeOutput:
    """One output slot of one node: the value a connection input holds."""

    def __init__(self, node: 'ComfyNode', slot: SlotIndex) -> None:
        self.node = node
        self.slot = slot

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NodeOutput) and other.node is self.node and other.slot == self.slot

    def __hash__(self) -> int:
        return hash((id(self.node), self.slot))

    def __repr__(self) -> str:
        return f'NodeOutput({self.node.class_type}, {self.slot})'


Connection: TypeAlias = Optional[NodeOutput]
"""Type of a node's connection inputs. None leaves the input unconnected, and the workflow omits it."""


class Output:
    """Declares a node output slot. Reading it from a node gives that node's `NodeOutput` for the slot."""

    def __init__(self, slot: SlotIndex) -> None:
        self.slot = slot

    @overload
    def __get__(self, instance: None, owner: type) -> 'Output': ...

    @overload
    def __get__(self, instance: 'ComfyNode', owner: type) -> NodeOutput: ...

    def __get__(self, instance: Optional['ComfyNode'], owner: type) -> 'Output | NodeOutput':
        if instance is None:
            return self
        return NodeOutput(instance, self.slot)


class ComfyNode(BaseModel):
    """Abstract node model. Subclasses set `CLASS_TYPE` and declare inputs as fields and outputs as `Output`s.

    Fields marked `Field(exclude=True)` configure the node without being sent as inputs. Nodes compare and hash by
    identity, since two nodes with the same inputs are still separate graph vertices.
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True, arbitrary_types_allowed=True,
                              ignored_types=(Output,), protected_namespaces=())

    CLASS_TYPE: ClassVar[str]

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @property
    def class_type(self) -> str:
        """Returns the node's type name used in the API."""
        return self.CLASS_TYPE

    def inputs(self) -> dict[str, Any]:
        """Returns the node's set inputs by ComfyUI input name, with connections as `NodeOutput`s."""
        return {name: getattr(self, name) for name, field in type(self).model_fields.items()
                if not field.exclude and getattr(self, name) is not None}

    def connections(self) -> list[NodeOutput]:
        """Returns the outputs of other nodes that this node's inputs are connected to."""
        return [value for value in self.inputs().values() if isinstance(value, NodeOutput)]

    def clear_connections(self) -> None:
        """Disconnects all inputs from other nodes."""
        for name in type(self).model_fields.keys():
            if isinstance(getattr(self, name), NodeOutput):
                setattr(self, name, None)

    def get_dict(self, node_keys: Mapping['ComfyNode', NodeId]) -> NodeDict:
        """Returns the node API dict, using `node_keys` to identify connected nodes."""
        inputs: dict[str, Any] = {}
        for name, value in self.inputs().items():
            if isinstance(value, NodeOutput):
                connection: NodeConnection = (node_keys[value.node], value.slot)
                inputs[name] = connection
            else:
                inputs[name] = deepcopy(value)
        return {'class_type': self.class_type, 'inputs': inputs}
