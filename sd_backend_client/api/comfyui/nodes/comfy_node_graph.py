"""Collects connected ComfyUI nodes into a workflow and assigns their keys."""
from sd_backend_client.api.comfyui.nodes.comfy_node import ComfyNode, NodeId, NodeDict

FIRST_NODE_KEY = 3


class ComfyNodeGraph:
    """A ComfyUI workflow: the added nodes, plus every node they read input from.

    Node keys are assigned when the workflow dict is built, upstream nodes first, counting up from `FIRST_NODE_KEY`.
    """

    def __init__(self) -> None:
        self._nodes: list[ComfyNode] = []

    def add_node(self, node: ComfyNode) -> None:
        """Adds a node to the workflow. Nodes it reads from through its connections are included automatically."""
        if node not in self._nodes:
            self._nodes.append(node)

    def get_workflow_dict(self) -> dict[NodeId, NodeDict]:
        """Gets the dict defining the entire workflow.

        Raises
        ------
        ValueError
            If the node connections form a cycle.
        """
        node_keys: dict[ComfyNode, NodeId] = {}
        visiting: set[ComfyNode] = set()

        def _assign_keys(node: ComfyNode) -> None:
            if node in node_keys:
                return
            if node in visiting:
                raise ValueError(f'Workflow has a connection cycle through a {node.class_type} node')
            visiting.add(node)
            for connection in node.connections():
                _assign_keys(connection.node)
            visiting.remove(node)
            node_keys[node] = str(FIRST_NODE_KEY + len(node_keys))

        for graph_node in self._nodes:
            _assign_keys(graph_node)
        return {key: node.get_dict(node_keys) for node, key in node_keys.items()}
