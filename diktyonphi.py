import enum
import subprocess
from typing import Dict, Hashable, Any, Optional, Iterator, Tuple


class GraphType(enum.Enum):
    """Graph orientation type: directed or undirected."""
    DIRECTED = 0
    UNDIRECTED = 1


class Edge:
    """Representation of an edge between two nodes with associated attributes."""

    def __init__(self, src: 'Node', dest: 'Node', attrs: Dict[str, Any]):
        """
        Initialize an edge from src_id to dest_id with given attributes.

        :param src: Source node identifier.
        :param dest: Destination node identifier.
        :param attrs: Dictionary of edge attributes.
        """
        self.src = src
        self.dest = dest
        self._attrs = attrs

    def __getitem__(self, key: str) -> Any:
        """Access edge attribute by key."""
        return self._attrs[key]

    def __setitem__(self, key: str, val: Any) -> None:
        """Set edge attribute by key."""
        self._attrs[key] = val

    def __repr__(self):
        return f"Edge({self.src.id}→{self.dest.id}, {self._attrs})"


class Node:
    """Representation of a graph node with attributes and outgoing edges."""

    def __init__(self, graph: 'Graph', node_id: Hashable, attrs: Dict[str, Any]):
        """
        Initialize a node with a given identifier and attributes.

        :param node_id: Unique identifier of the node.
        :param attrs: Dictionary of node attributes.
        """
        self.id = node_id
        self.graph = graph
        self._attrs = attrs
        self._neighbors: Dict[Hashable, Dict[str, Any]] = {}

    def __getitem__(self, item: str) -> Any:
        """Access node attribute by key."""
        return self._attrs[item]

    def __setitem__(self, item: str, val: Any) -> None:
        """Set node attribute by key."""
        self._attrs[item] = val

    def to(self, dest: Hashable | 'Node') -> Edge:
        """
        Get the edge from this node to the specified destination node.

        :param dest_id: ID of the target node.
        :return: Edge instance representing the connection.
        :raises ValueError: If no such edge exists.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        if dest_id not in self._neighbors:
            raise ValueError(f"No edge from {self.id} to {dest_id}")
        return Edge(self, self.graph.node(dest_id), self._neighbors[dest_id])

    def connect_to(self,  dest: Hashable | 'Node', attrs: Optional[Dict[str, Any]] = None):
        dest = dest if isinstance(dest, Node) else self.graph.node(dest)
        assert dest.graph == self.graph, f"Destination node {dest.id} is not in the same graph"
        assert dest.id in self.graph, f"Destination node {dest.id} is not in graph"
        self.graph.add_edge(self.id, dest.id, attrs if attrs is not None else {})

    def is_edge_to(self, dest: Hashable | 'Node') -> bool:
        """
        Check if this node has an edge to the given node.

        :param dest_id: ID of the target node.
        :return: True if edge exists, False otherwise.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        return dest_id in self._neighbors

    @property
    def neighbor_ids(self) -> Iterator[Hashable]:
        """Return an iterator over IDs of neighboring nodes."""
        return iter(self._neighbors)

    @property
    def neighbor_nodes(self) -> Iterator['Node']:
        for id in self.neighbor_ids:
            yield self.graph.node(id)

    @property
    def out_degree(self) -> int:
        """Return the number of outgoing edges."""
        return len(self._neighbors)

    def __repr__(self):
        return f"Node({self.id}, {self._attrs})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Graph:
    """Graph data structure supporting directed and undirected graphs."""

    def __init__(self, type: GraphType):
        """
        Initialize a graph with the given type.

        :param type: GraphType.DIRECTED or GraphType.UNDIRECTED
        """
        self.type = type
        self._nodes: Dict[Hashable, Node] = {}

    def add_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """
        Add a new node to the graph.

        :param node_id: Unique node identifier.
        :param attrs: Optional dictionary of attributes.
        :raises ValueError: If the node already exists.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        return self._create_node(node_id, attrs if attrs is not None else {})

    def add_edge(self, src_id: Hashable, dst_id: Hashable,
                 attrs: Optional[Dict[str, Any]] = None) -> Tuple[Node, Node]:
        """
        Add a new edge to the graph. Nodes are created automatically if missing.

        :param src_id: Source node ID.
        :param dst_id: Destination node ID.
        :param attrs: Optional dictionary of edge attributes.
        :raises ValueError: If the edge already exists.
        """
        attrs = attrs if attrs is not None else {}
        if src_id not in self._nodes:
            self._create_node(src_id, {})
        if dst_id not in self._nodes:
            self._create_node(dst_id, {})
        self._set_edge(src_id, dst_id, attrs)
        if self.type == GraphType.UNDIRECTED:
            self._set_edge(dst_id, src_id, attrs)
        return (self._nodes[src_id], self._nodes[dst_id])

    def __contains__(self, node_id: Hashable) -> bool:
        """Check whether a node exists in the graph."""
        return node_id in self._nodes

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over node IDs in the graph."""
        return iter(self._nodes.values())

    def node_ids(self) -> Iterator[Hashable]:
        return iter(self._nodes.keys())

    def node(self, node_id: Hashable) -> Node:
        """
        Get the Node instance with the given ID.

        :param node_id: The ID of the node.
        :return: Node instance.
        :raises KeyError: If the node does not exist.
        """
        return self._nodes[node_id]

    def _create_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """Internal method to create a node."""
        node = Node(self, node_id, attrs)
        self._nodes[node_id] = node
        return node

    def _set_edge(self, src_id: Hashable, target_id: Hashable, attrs: Dict[str, Any]) -> None:
        """Internal method to create a directed edge."""
        if target_id in self._nodes[src_id]._neighbors:
            raise ValueError(f"Edge {src_id}→{target_id} already exists")
        self._nodes[src_id]._neighbors[target_id] = attrs

    def __repr__(self):
        edges = sum(node.out_degree for node in self._nodes.values())
        if self.type == GraphType.UNDIRECTED:
            edges //= 2
        return f"Graph({self.type}, nodes: {len(self._nodes)}, edges: {edges})"

    def to_dot(self, label_attr:str ="label", weight_attr:str = "weight") -> str:
        """
        Generate a simple Graphviz (DOT) representation of the graph. Generated by ChatGPT.

        :return: String in DOT language.
        """
        lines = []
        name = "G"
        connector = "->" if self.type == GraphType.DIRECTED else "--"

        lines.append(f'digraph {name} {{' if self.type == GraphType.DIRECTED else f'graph {name} {{')

        # Nodes
        for node_id in self.node_ids():
            node = self.node(node_id)
            label = node[label_attr] if label_attr in node._attrs else str(node_id)
            lines.append(f'    "{node_id}" [label="{label}"];')

        # Edges
        seen = set()
        for node_id in self.node_ids():
            node = self.node(node_id)
            for dst_id in node.neighbor_ids:
                if self.type == GraphType.UNDIRECTED and (dst_id, node_id) in seen:
                    continue
                seen.add((node_id, dst_id))
                edge = node.to(dst_id)
                label = edge[weight_attr] if weight_attr in edge._attrs else ""
                lines.append(f'    "{node_id}" {connector} "{dst_id}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)


    def export_to_png(self, filename: str = None) -> None:
        """
        Export the graph to a PNG file using Graphviz (dot). Graphviz (https://graphviz.org/)
         must be installed.

        :param filename: Output PNG filename.
        :raises RuntimeError: If Graphviz 'dot' command fails.
        """
        dot_data = self.to_dot()
        try:
            subprocess.run(
                ["dot", "-Tpng", "-o", filename],
                input=dot_data,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz 'dot' command failed: {e}") from e

    def _repr_svg_(self):
        """
          Return SVG representation of the graph for Jupyter notebook (implementation
          of protocol of IPython).
        """
        return self.to_image().data

    def to_image(self):
        """
            Return graph as SVG (usable in IPython notebook).
        """
        from IPython.display import SVG
        dot_data = self.to_dot()
        try:
            process = subprocess.run(
                ['dot', '-Tsvg'],
                input=dot_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return SVG(data=process.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz 'dot' command failed: {e} with stderr: {e.stderr.decode('utf-8')}") from e

if __name__ == "__main__":
    # Create a directed graph
    g = Graph(GraphType.DIRECTED)

    # Add nodes with attributes
    g.add_node("A", {"label": "Start", "color": "green"})
    g.add_node("B", {"label": "Middle", "color": "yellow"})
    g.add_node("C", {"label": "End", "color": "red"})
    g.add_node("D", {"label": "Optional", "color": "blue"})

    # Add edges with attributes
    g.add_edge("A", "B", {"weight": 1.0, "type": "normal"})
    g.add_edge("B", "C", {"weight": 2.5, "type": "critical"})
    g.add_edge("A", "D", {"weight": 0.8, "type": "optional"})
    g.add_edge("D", "C", {"weight": 1.7, "type": "fallback"})

    # Access and update node attribute
    print("Node A color:", g.node("A")["color"])
    g.node("A")["color"] = "darkgreen"

    # Access edge and modify its weight
    edge = g.node("A").to("B")
    print("Edge A→B weight:", edge["weight"])
    edge["weight"] = 1.1

    # Iterate through the graph
    print("\nGraph structure:")
    for node_id in g.node_ids():
        node = g.node(node_id)
        print(f"Node {node.id}: label={node['label']}, out_degree={node.out_degree}")
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            print(f"  → {neighbor_id} (weight={edge['weight']}, type={edge['type']})")

    print("-----------------")
    print(g.to_image())









if __name__ == "__main__":
    
          # ===============  TASKS 1–3  (drop-in для вашего Graph API)  ===============

 from collections import defaultdict, deque
import itertools

def _iter_edges(g: "Graph"):
    """
    Итератор по рёбрам графа как по парам (u, v).
    Для UNDIRECTED возвращаем каждое ребро один раз (u <= v).
    """
    if g.type == GraphType.DIRECTED:
        for u in g.node_ids():
            for v in g.node(u).neighbor_ids:
                yield (u, v)
    else:
        seen = set()
        for u in g.node_ids():
            for v in g.node(u).neighbor_ids:
                a = (u, v) if u <= v else (v, u)
                if a not in seen:
                    seen.add(a)
                    yield a

# ---------- Задание 1: построить кружницу и проверить, что граф — ровно один цикл ----------

def build_cycle_graph(n: int) -> "Graph":
    if n < 3:
        raise ValueError("Кружница требует n >= 3")
    g = Graph(GraphType.UNDIRECTED)
    for i in range(1, n):
        g.add_edge(i, i+1)
    g.add_edge(n, 1)
    return g

def is_cycle_graph(g: "Graph") -> bool:
    if g.type != GraphType.UNDIRECTED:
        return False

    V = set(g.node_ids())
    if len(V) < 3:
        return False

    E = list(_iter_edges(g)) 
    if len(E) != len(V):
        return False

    deg = defaultdict(int)
    adj = defaultdict(set)
    for u, v in E:
        deg[u] += 1; deg[v] += 1
        adj[u].add(v); adj[v].add(u)

    if any(deg[u] != 2 for u in V):
        return False

    start = next(iter(V))
    seen = {start}
    q = deque([start])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); q.append(y)
    return len(seen) == len(V)

# ---------- Задание 2: для ориентированного графа Σ outdegree = Σ indegree ----------

def check_in_out_sum_equal(dg: "Graph"):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Ожидается ориентированный граф")

    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    m = 0
    for u in dg.node_ids():
        for v in dg.node(u).neighbor_ids:
            outdeg[u] += 1
            indeg[v]  += 1
            m += 1

    sum_out = sum(outdeg.values())
    sum_in  = sum(indeg.values())
    return (sum_in == sum_out == m, sum_in, sum_out, m)

# ---------- Задание 3: линейный граф L(G) для НЕориентированного графа ----------

def line_graph(g: "Graph") -> "Graph":
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Ожидается неориентированный граф")

    base_edges = list(_iter_edges(g))  # уже без дублей (u<=v)
    L = Graph(GraphType.UNDIRECTED)

    # Чтобы изолированные вершины тоже существовали в L(G)
    for e in base_edges:
        L.add_node(e, {})  # кортеж годится как ID

    for e1, e2 in itertools.combinations(base_edges, 2):
        if set(e1) & set(e2):
            L.add_edge(e1, e2)  # add_edge сам не продублирует в UNDIRECTED
    return L


if __name__ == "__main__":
    # 1) Кружница
    g1 = build_cycle_graph(4)
    print("1) Кружница? ->", is_cycle_graph(g1))  # True

    # 2) Σ in == Σ out
    dg = Graph(GraphType.DIRECTED)
    dg.add_edge(1, 2); dg.add_edge(1, 3); dg.add_edge(2, 3)
    ok, si, so, m = check_in_out_sum_equal(dg)
    print(f"2) Σin={si}, Σout={so}, m={m}, ok={ok}")  # ok=True

    # 3) Линейный граф
    g3 = Graph(GraphType.UNDIRECTED)
    g3.add_edge(1,2); g3.add_edge(2,3); g3.add_edge(3,4); g3.add_edge(3,5)
    L = line_graph(g3)
    print("3) L(G):", L)  # краткая сводка (число вершин/рёбер)
    # Если установлен Graphviz, можно вывести PNG:
    # L.export_to_png("line_graph.png")



#Ověření „věty o podání rukou“ (neorientovaný graf)
from collections import defaultdict
def _edges_undirected_once(g):
    seen = set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            a = (u, v) if u <= v else (v, u)
            if a not in seen:
                seen.add(a)
                yield a

def over_handshaking(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Očekáván neorientovaný graf.")
    deg = defaultdict(int)
    E = list(_edges_undirected_once(g))
    for u, v in E:
        deg[u] += 1
        deg[v] += 1

    sum_deg = sum(deg.values())
    twoE = 2 * len(E)
    return (sum_deg == twoE, sum_deg, twoE, len(E))

if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    g.add_edge(1,2); g.add_edge(2,3); g.add_edge(3,4); g.add_edge(4,1)
    ok, sdeg, twoE, m = over_handshaking(g)
    print(f"Handshaking: ok={ok}, Σdeg={sdeg}, 2|E|={twoE}, |E|={m}")



from collections import deque  # tady se nepoužije, ale ať nevadí

def over_handshaking(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Čekám neorientovaný graf.")

    # Σdeg(v) = součet počtů sousedů
    sum_deg = 0
    for u in g.node_ids():
        sum_deg += len(g.node(u).neighbor_ids)

    # |E| spočítám tak, že u<=v vezmu hranu právě jednou
    m = 0
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            if u <= v:
                m += 1

    return (sum_deg == 2*m, sum_deg, 2*m, m)
if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    g.add_edge(1,2); g.add_edge(2,3); g.add_edge(3,4); g.add_edge(4,1)  # C4
    print(f"Handshaking: ok={ok}, Σdeg={sdeg}, 2|E|={twoE}, |E|={m}")





#Počet komponent spojitosti (NEorientovaný graf)
def pocet_komponent(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Čekám neorientovaný graf.")

    visited = set()
    comp = 0

    for s in g.node_ids():
        if s in visited:
            continue
        comp += 1
        stack = [s]
        visited.add(s)
        while stack:
            u = stack.pop()
            for v in g.node(u).neighbor_ids:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
    return comp

if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    # komponenta 1: 1-2-3
    g.add_edge(1,2); g.add_edge(2,3)
    # komponenta 2: 4-5
    g.add_edge(4,5)
    print(pocet_komponent(g))  # očekávání: 2



def izolovane_uzly(g):
    if g.type != GraphType.DIRECTED:
        raise ValueError("Čekám orientovaný graf.")
    V = list(g.node_ids())
    indeg = {u: 0 for u in V}
    outdeg = {u: 0 for u in V}
    for u in V:
        for v in g.node(u).neighbor_ids:
            outdeg[u] += 1
            indeg[v] = indeg.get(v, 0) + 1
    return [u for u in V if indeg.get(u,0) == 0 and outdeg.get(u,0) == 0]


def uplny_graf(n):
    g = Graph(GraphType.UNDIRECTED)
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i+1, n):

            def GrafKružnice(n):
                if n < 3:
                    raise ValueError("Kružnice vyžaduje alespoň 3 uzly.")
                g = Graph(GraphType.UNDIRECTED)
                for i in range(1, n):
                    g.add_edge(i, i+1)
                g.add_edge(n, 1)
                return g

            def JeToKružnice(g):
                if g.type != GraphType.UNDIRECTED:
                    return False
                V = set(g.node_ids())
                if len(V) < 3:
                    return False
                E = []
                seen = set()
                for u in g.node_ids():
                    for v in g.node(u).neighbor_ids:
                        a = (u, v) if u <= v else (v, u)
                        if a not in seen:
                            seen.add(a)
                            E.append(a)
                if len(E) != len(V):
                    return False
                for u in V:
                    if len(list(g.node(u).neighbor_ids)) != 2:
                        return False
                # Ověření spojitosti
                visited = set()
                stack = [next(iter(V))]
                while stack:
                    u = stack.pop()
                    if u not in visited:
                        visited.add(u)
                        for v in g.node(u).neighbor_ids:
                            if v not in visited:
                                stack.append(v)
                

def test_check_in_out_sum_equal():
    # Create a directed graph
    g = Graph(GraphType.DIRECTED)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    g.add_edge("A", "C")

    ok, sum_in, sum_out, m = check_in_out_sum_equal(g)
    assert ok, "Součet vstupních stupňů není roven součtu výstupních stupňů"
    assert sum_in == sum_out == m
    assert sum_in == 4
    assert sum_out == 4
    assert m == 4

if __name__ == "__main__":
    test_check_in_out_sum_equal()
    print("Test passed.") 
        


def zdroje_a_sinky(g):
    V = list(g.node_ids())
    indeg = {u: 0 for u in V}
    outdeg = {u: 0 for u in V}
    for u in V:
        for v in g.node(u).neighbor_ids:
            outdeg[u] += 1
            indeg[v] = indeg.get(v, 0) + 1
    sources = [u for u in V if indeg[u] == 0]
    sinks   = [u for u in V if outdeg[u] == 0]
    return sources, sinks

if __name__ == "__main__":
    dg = Graph(GraphType.DIRECTED)
    dg.add_edge(1, 2)
    dg.add_edge(1, 3)
    dg.add_node(4)  # изолированный узел

    sources, sinks = zdroje_a_sinky(dg)
    print("sources:", sources)
    print("sinks:", sinks)






def je_strom(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Cekam neorientovany graph.")
    V = list(g.node_ids())
    if not V:
        return True
    


    m = 0 
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u <= v:
                m += 1
    if m != len(V) - 1:
        return False
    

    seen, st = {V[0]}, [V[0]]
    while st:
        u = st.pop()
        for v in g.node(u).neighbor_ids:
            if v not in seen:
                seen.add(v); st.append(v)
    return len(seen) == len(V)





if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]:
        g.add_edge(*e)
    print("je_strom(True):", je_strom(g))

    g2 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(2,4)]:
        g2.add_edge(*e)
    print("je_strom(False):", je_strom(g2))




import itertools
def line_graph(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Cekam neorientovany graf")
    

    base = []
    seen = set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
         a = (u, v) if u <= v else (v,u)
        if a not in seen:
            seen.add(a)
            base.append(a)
    L = Graph(GraphType.UNDIRECTED)


    for e in base:
        L.add_node(e)


    for e1, e2 in itertools.combinations(base, 2):
        if set(e1) & set(e2):
            L.add_edge(e1, e2)

    return L



if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(3,5)]:
        g.add_edge(*e)


    L = line_graph(g)
    print("V(L)=", len(list(L.node_ids())))





from collections import deque

def je_kruznice(g):
    if g.type != GraphType.UNDIRECTED:
        return False
    V = list(g.node_ids())
    if len(V) < 3:
        return False
    

    E, seen = [], set()
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u == v:
                return False
            a = (u, v) if u <= v else(v, u)
            if a not in seen:
                seen.add(a); E.append(a)
    if len(E) != len(V):
        return False
    


    adj = {u: set() for u in V}
    deg = {u: 0 for u in V}
    for u, v in E:
        adj[u].add(v); adj[v].add(u)
        deg[u] +=1; deg[v] += 1


    if any(deg[u] != 2 for u in V):
        return False
    


    q, seenV = deque([V[0]]), {V[0]}
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seenV:
                seenV.add(y); q.append(y)
    return len(seenV) == len(V)



if __name__ == "__main__":
    g = Graph(GraphType.UNDIRECTED)
    for e in [(1,2), (2,3), (3,4), (4,1)]:
        g.add_edge(*e)

    print("C4 ->", je_kruznice(g))


    h = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]:
        h.add_edge(*e)
    print("path ->", je_kruznice(h))







#Zdroje a sinky (digraf)
def zdroje_a_sinky(g):
    if g.type != GraphType.DIRECTED: raise ValueError("Čekám digraf.")
    V = list(g.node_ids())
    indeg = {u:0 for u in V}; outdeg = {u:0 for u in V}
    for u in V:
        for v in g.node(u).neighbor_ids:
            outdeg[u]+=1; indeg[v]=indeg.get(v,0)+1
    return [u for u in V if indeg[u]==0], [u for u in V if outdeg[u]==0]

# test
dg=Graph(GraphType.DIRECTED); 
dg.add_edge(1,2); 
dg.add_edge(1,3); 
dg.add_node(4)
print(zdroje_a_sinky(dg))  # ([1,4], [2,3,4])


#2Σ in-degree = Σ out-degree (digraf)
def soucet_in_out(dg):
    if dg.type != GraphType.DIRECTED: raise ValueError("Čekám digraf.")
    si=so=m=0
    indeg={}; outdeg={}
    for u in dg.node_ids():
        for v in dg.node(u).neighbor_ids:
            m+=1; outdeg[u]=outdeg.get(u,0)+1; indeg[v]=indeg.get(v,0)+1
    si=sum(indeg.values()); so=sum(outdeg.values())
    return (si==so==m, si, so, m)

# test
# dg=Graph(GraphType.DIRECTED); dg.add_edge(1,2); dg.add_edge(1,3); dg.add_edge(2,3)
# print(soucet_in_out(dg))  # (True, 3, 3, 3)



#3) Izolované uzly (digraf)
def izolovane_uzly(dg):
    if dg.type != GraphType.DIRECTED: raise ValueError("Čekám digraf.")
    V=list(dg.node_ids()); indeg={u:0 for u in V}; outdeg={u:0 for u in V}
    for u in V:
        for v in dg.node(u).neighbor_ids:
            outdeg[u]+=1; indeg[v]+=1
    return [u for u in V if indeg[u]==0 and outdeg[u]==0]

# test
# dg=Graph(GraphType.DIRECTED); dg.add_edge(5,2); dg.add_edge(2,3); dg.add_node(4); dg.add_node(1)
# print(izolovane_uzly(dg))  # [4,1] (pořadí může být jiné)



#4) Úplný graf Kₙ + test

def uplny_graf(n):
    g=Graph(GraphType.UNDIRECTED)
    for i in range(n): g.add_node(i)
    for i in range(n):
        for j in range(i+1,n): g.add_edge(i,j)
    return g

def testuj_uplnost(g):
    if g.type!=GraphType.UNDIRECTED: return False
    V=list(g.node_ids()); n=len(V)
    return all(len(g.node(u).neighbor_ids)==n-1 for u in V)

# test
# g=uplny_graf(4); print(testuj_uplnost(g))  # True



#5) Kružnice (jediný cyklus, neorientovaný)
from collections import deque

def je_kruznice(g):
    if g.type!=GraphType.UNDIRECTED: return False
    V=list(g.node_ids()); 
    if len(V)<3: return False
    # hrany jednou (u<=v) + stupně
    m=0; deg={u:0 for u in V}; adj={u:set() for u in V}
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u<=v:
                m+=1
            deg[u]+=1; deg[v]+=0  # jen aby byly klíče; skutečné deg dopočítáme z adj
            adj[u].add(v); adj[v].add(u)
    # přepočítat deg korektně
    deg={u:len(adj[u]) for u in V}
    if m!=len(V): return False
    if any(deg[u]!=2 for u in V): return False
    # spojitost
    q=deque([V[0]]); seen={V[0]}
    while q:
        x=q.popleft()
        for y in adj[x]:
            if y not in seen: seen.add(y); q.append(y)
    return len(seen)==len(V)

# test
# g=Graph(GraphType.UNDIRECTED); [g.add_edge(*e) for e in [(1,2),(2,3),(3,4),(4,1)]]
# print(je_kruznice(g))  # True



#6) Je strom? (neorientovaný)
def je_strom(g):
    if g.type!=GraphType.UNDIRECTED: raise ValueError("Neorientovaný.")
    V=list(g.node_ids())
    if not V: return True
    # hrany jednou
    m=0
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u<=v: m+=1
    if m!=len(V)-1: return False
    # spojitost (DFS)
    seen=set([V[0]]); st=[V[0]]
    while st:
        u=st.pop()
        for v in g.node(u).neighbor_ids:
            if v not in seen: seen.add(v); st.append(v)
    return len(seen)==len(V)

# test
# g=Graph(GraphType.UNDIRECTED); [g.add_edge(*e) for e in [(1,2),(2,3),(3,4)]]
# print(je_strom(g))  # True



#7) Počet komponent (neorientovaný)
def pocet_komponent(g):
    if g.type!=GraphType.UNDIRECTED: raise ValueError("Neorientovaný.")
    visited=set(); comp=0
    for s in g.node_ids():
        if s in visited: continue
        comp+=1; st=[s]; visited.add(s)
        while st:
            u=st.pop()
            for v in g.node(u).neighbor_ids:
                if v not in visited: visited.add(v); st.append(v)
    return comp

# test
# g=Graph(GraphType.UNDIRECTED); g.add_edge(1,2); g.add_edge(2,3); g.add_edge(4,5)
# print(pocet_komponent(g))  # 2


#8) Hranový (line) graf L(G) (neorientovaný vstup)
import itertools
def line_graph(g):
    if g.type!=GraphType.UNDIRECTED: raise ValueError("Neorientovaný.")
    base=[]; seen=set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            a=(u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); base.append(a)
    L=Graph(GraphType.UNDIRECTED)
    for e1,e2 in itertools.combinations(base,2):
        if set(e1)&set(e2): L.add_edge(e1,e2)
    return L

# test
# g=Graph(GraphType.UNDIRECTED); [g.add_edge(*e) for e in [(1,2),(2,3),(3,4),(3,5)]]
# L=line_graph(g); print(len(list(L.node_ids())))  # 4 vrcholy (= 4 hrany G)



#import itertools
def line_graph(g):
    if g.type!=GraphType.UNDIRECTED: raise ValueError("Neorientovaný.")
    base=[]; seen=set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            a=(u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); base.append(a)
    L=Graph(GraphType.UNDIRECTED)
    for e1,e2 in itertools.combinations(base,2):
        if set(e1)&set(e2): L.add_edge(e1,e2)
    return L

# test
# g=Graph(GraphType.UNDIRECTED); [g.add_edge(*e) for e in [(1,2),(2,3),(3,4),(3,5)]]
# L=line_graph(g); print(len(list(L.node_ids())))  # 4 vrcholy (= 4 hrany G)




#9) WeightedGraph (digraf z matice vah)

class WeightedGraph:
    def __init__(self, W):
        n=len(W); self.g=Graph(GraphType.DIRECTED); self.w={}
        for i in range(n): self.g.add_node(i)
        for i in range(n):
            for j in range(n):
                x=W[i][j]
                if x!=-1:
                    if x<0: raise ValueError("Váha >=0 nebo -1.")
                    self.g.add_edge(i,j); self.w[(i,j)]=float(x)
    def get_graph(self): return self.g
    def component_count(self):
        V=list(self.g.node_ids()); seen=set(); comp=0
        for s in V:
            if s in seen: continue
            comp+=1; st=[s]; seen.add(s)
            while st:
                u=st.pop()
                nbrs=set(self.g.node(u).neighbor_ids)
                for x in V:
                    if u in self.g.node(x).neighbor_ids: nbrs.add(x)
                for v in nbrs:
                    if v not in seen: seen.add(v); st.append(v)
        return comp
    def get_sorted_weights(self):
        return iter(sorted(self.w.values()))

# test
# W=[[ -1, 2, -1],[ -1,-1, 5],[ 1,-1,-1]]
# WG=WeightedGraph(W); print(list(WG.get_sorted_weights()))  # [1.0,2.0,5.0]



#class MetricSpaceGraph:
    def __init__(self, n):
        self.g=Graph(GraphType.UNDIRECTED); self.dist={}
        for i in range(n): self.g.add_node(i)
    def _k(self,a,b): return (a,b) if a<=b else (b,a)
    def addEdge(self,u,v,d):
        if u==v or d<=0: raise ValueError("Bez smyček, d>0.")
        best=None
        for w in self.g.node_ids():
            d1=self.dist.get(self._k(u,w)); d2=self.dist.get(self._k(w,v))
            if d1 is not None and d2 is not None:
                s=d1+d2; best = s if best is None or s<best else best
        if best is not None and d>best:
            raise ValueError("Porušení trojúhelníkové nerovnosti.")
        self.g.add_edge(u,v); self.dist[self._k(u,v)]=float(d)
    def getDistanceMatrix(self):
        V=sorted(self.g.node_ids()); n=len(V)
        M=[[-1.0]*n for _ in range(n)]
        for i in range(n): M[i][i]=0.0
        for (u,v),d in self.dist.items():
            i=V.index(u); j=V.index(v); M[i][j]=M[j][i]=d
        return M

# test
# M=MetricSpaceGraph(4); M.addEdge(0,1,2); M.addEdge(1,3,5); M.addEdge(0,2,4); M.addEdge(2,3,2)
# M.addEdge(0,3,6)  # ok (6 <= 2+5 nebo 4+2)


#1) Věta o podání rukou (NEorientovaný)
def over_handshaking(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Čekám neorientovaný graf.")
    # Σdeg(v) = součet počtů sousedů
    sum_deg = sum(len(g.node(u).neighbor_ids) for u in g.node_ids())
    # |E| jednou: u<=v
    m = 0
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            if u <= v:
                m += 1
    return (sum_deg == 2*m, sum_deg, 2*m, m)
#2) Součet in-degree = součet out-degree (orientovaný)
def soucet_in_out(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    indeg = {}; outdeg = {}; m = 0
    for u in dg.node_ids():
        for v in dg.node(u).neighbor_ids:
            m += 1
            outdeg[u] = outdeg.get(u, 0) + 1
            indeg[v]  = indeg.get(v, 0)  + 1
    si = sum(indeg.values()); so = sum(outdeg.values())
    return (si == so == m, si, so, m)
#k-regulární graf (NEorientovaný)
def k_regularni(g, k):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    return all(len(g.node(u).neighbor_ids) == k for u in g.node_ids())
#Úplný graf Kₙ + test
def uplny_graf(n):
    g = Graph(GraphType.UNDIRECTED)
    for i in range(n): g.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j)
    return g

def testuj_uplnost(g):
    if g.type != GraphType.UNDIRECTED: return False
    V = list(g.node_ids()); n = len(V)
    return all(len(g.node(u).neighbor_ids) == n-1 for u in V)
#5) Stupňová posloupnost (NEorientovaný)
def stupne_seznam(g):
    return sorted(len(g.node(u).neighbor_ids) for u in g.node_ids())
#6) Listy (vrcholy stupně 1) (NEorientovaný)
def listy(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    return [u for u in g.node_ids() if len(g.node(u).neighbor_ids) == 1]
#7) Počet komponent spojitosti (NEorientovaný)
def pocet_komponent(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    visited = set(); comp = 0
    for s in g.node_ids():
        if s in visited: continue
        comp += 1
        stack = [s]; visited.add(s)
        while stack:
            u = stack.pop()
            for v in g.node(u).neighbor_ids:
                if v not in visited:
                    visited.add(v); stack.append(v)
    return comp
#8) Je strom? (NEorientovaný)
def je_strom(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    V = list(g.node_ids())
    if not V: return True
    m = 0
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u <= v: m += 1
    if m != len(V) - 1: return False
    seen, st = {V[0]}, [V[0]]
    while st:
        u = st.pop()
        for v in g.node(u).neighbor_ids:
            if v not in seen:
                seen.add(v); st.append(v)
    return len(seen) == len(V)
#9) Je kružnice (jediný cyklus) (NEorientovaný)
from collections import deque

def je_kruznice(g):
    if g.type != GraphType.UNDIRECTED: return False
    V = list(g.node_ids())
    if len(V) < 3: return False
    E, seen = [], set()
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u == v: return False
            a = (u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); E.append(a)
    if len(E) != len(V): return False
    adj = {u:set() for u in V}
    for u,v in E:
        adj[u].add(v); adj[v].add(u)
    if any(len(adj[u]) != 2 for u in V): return False
    q, seenV = deque([V[0]]), {V[0]}
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seenV: seenV.add(y); q.append(y)
    return len(seenV) == len(V)
#10) Je bipartitní? (NEorientovaný)
from collections import deque

def je_bipartitni(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    color = {}
    for s in g.node_ids():
        if s in color: continue
        color[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in g.node(u).neighbor_ids:
                if v not in color:
                    color[v] = 1 - color[u]; q.append(v)
                elif color[v] == color[u]:
                    return False
    return True
#11) Trojúhelník (K₃) existuje? (NEorientovaný)
def ma_trojuhelnik(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    # sousedé jako množiny
    N = {u: set(g.node(u).neighbor_ids) for u in g.node_ids()}
    for u in g.node_ids():
        for v in N[u]:
            if v == u: continue
            # průnik sousedů u a v
            if N[u].intersection(N[v]):
                return True
    return False
#12) Postav kružnici Cₙ (NEorientovaný)
def kruznice_Cn(n):
    if n < 3: raise ValueError("n>=3")
    g = Graph(GraphType.UNDIRECTED)
    for i in range(1, n):
        g.add_edge(i, i+1)
    g.add_edge(n, 1)
    return g
#13) Cesta Pₙ (NEorientovaný)
def cesta_Pn(n):
    if n < 1: raise ValueError("n>=1")
    g = Graph(GraphType.UNDIRECTED)
    for i in range(1, n):
        g.add_edge(i, i+1)
    if n == 1: g.add_node(1)
    return g
#14) Hvězda Sₙ (NEorientovaný)
def hvezda_Sn(n):
    if n < 2: raise ValueError("n>=2")
    g = Graph(GraphType.UNDIRECTED)
    for i in range(2, n+1):
        g.add_edge(1, i)
    return g
#15) Kolo Wₙ (NEorientovaný)
def kolo_Wn(n):
    if n < 3: raise ValueError("n>=3")
    g = kruznice_Cn(n)
    for i in range(1, n+1):
        g.add_edge(0, i)
    return g
#16) Doplněk grafu Ĝ (NEorientovaný)
def doplnek_grafu(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    V = list(g.node_ids())
    adj = {u: set(g.node(u).neighbor_ids) for u in V}
    H = Graph(GraphType.UNDIRECTED)
    for u in V: H.add_node(u)
    for i,u in enumerate(V):
        for v in V[i+1:]:
            if v not in adj[u]:
                H.add_edge(u, v)
    return H
#17) Čtverec grafu G² (NEorientovaný)
def ctverec_grafu(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    V = list(g.node_ids())
    N = {u: set(g.node(u).neighbor_ids) for u in V}
    H = Graph(GraphType.UNDIRECTED)
    for u in V: H.add_node(u)
    for i,u in enumerate(V):
        reach = set(N[u])
        for x in N[u]:
            reach |= N[x]
        reach.discard(u)
        for v in V[i+1:]:
            if v in reach:
                H.add_edge(u, v)
    return H
#18) Hranový (line) graf L(G) (NEorientovaný vstup)
import itertools
def line_graph(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    base, seen = [], set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            a = (u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); base.append(a)
    L = Graph(GraphType.UNDIRECTED)
    for e in base: L.add_node(e)
    for e1, e2 in itertools.combinations(base, 2):
        if set(e1) & set(e2): L.add_edge(e1, e2)
    return L
#19) Transpozice digrafu Gᵀ (orientovaný)
def transpozice(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    H = Graph(GraphType.DIRECTED)
    for u in dg.node_ids(): H.add_node(u)
    for u in dg.node_ids():
        for v in dg.node(u).neighbor_ids:
            H.add_edge(v, u)
    return H
#20) Zdroje a sinky (orientovaný)
def zdroje_a_sinky(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    V = list(dg.node_ids())
    indeg = {u:0 for u in V}; outdeg = {u:0 for u in V}
    for u in V:
        for v in dg.node(u).neighbor_ids:
            outdeg[u]+=1; indeg[v]=indeg.get(v,0)+1
    return [u for u in V if indeg[u]==0], [u for u in V if outdeg[u]==0]
#21) Slabě souvislé komponenty (orientovaný → ignoruj orientaci)
def pocet_slabych_komponent(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    V = list(dg.node_ids()); seen=set(); comp=0
    for s in V:
        if s in seen: continue
        comp += 1
        st=[s]; seen.add(s)
        while st:
            u = st.pop()
            # sousedé "oběma směry"
            nbrs = set(dg.node(u).neighbor_ids)
            for x in V:
                if u in dg.node(x).neighbor_ids:
                    nbrs.add(x)
            for v in nbrs:
                if v not in seen: seen.add(v); st.append(v)
    return comp
#22) Topologické pořadí (Kahn) nebo „není DAG“
from collections import deque

def topologicke_poradi(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    V = list(dg.node_ids())
    indeg = {u:0 for u in V}
    for u in V:
        for v in dg.node(u).neighbor_ids:
            indeg[v] = indeg.get(v,0) + 1
    q = deque([u for u in V if indeg.get(u,0)==0])
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in dg.node(u).neighbor_ids:
            indeg[v]-=1
            if indeg[v]==0: q.append(v)
    return order if len(order)==len(V) else None
if __name__ == "__main__":
    # 1) Věta o podání rukou
    g1 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(4,1)]: g1.add_edge(*e)  # C4
    print("1) handshaking:", over_handshaking(g1))  # oček.: (True, 8, 8, 4)

    # 2) Σ in = Σ out = |E| (digraf)
    dg2 = Graph(GraphType.DIRECTED)
    for e in [(1,2),(1,3),(2,3)]: dg2.add_edge(*e)
    print("2) součet in/out:", soucet_in_out(dg2))  # oček.: (True,3,3,3)

    # 3) k-regulární (na K4 je k=3 True, na cestě P4 pro k=2 False)
    g3a = uplny_graf(4)
    print("3a) k-regulární K4 (k=3):", k_regularni(g3a, 3))  # True
    g3b = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g3b.add_edge(*e)  # P4
    print("3b) k-regulární P4 (k=2):", k_regularni(g3b, 2))  # False

    # 4) Kₙ + test
    g4 = uplny_graf(4)
    print("4) testuj_uplnost(K4):", testuj_uplnost(g4))  # True
    g4b = Graph(GraphType.UNDIRECTED); g4b.add_edge(1,2); g4b.add_edge(2,3)
    print("4) testuj_uplnost(řetěz):", testuj_uplnost(g4b))  # False

    # 5) Stupňová posloupnost
    g5 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g5.add_edge(*e)  # P4
    print("5) stupně P4:", stupne_seznam(g5))  # [1,2,2,1]

    # 6) Listy (deg=1)
    print("6) listy P4:", listy(g5))  # [1,4] (pořadí libovolné)

    # 7) Počet komponent (neorient.)
    g7 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(4,5)]: g7.add_edge(*e)
    print("7) komponenty:", pocet_komponent(g7))  # 2

    # 8) Je strom?
    g8 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g8.add_edge(*e)
    print("8a) strom (P4):", je_strom(g8))  # True
    g8.add_edge(2,4)
    print("8b) strom (s cyklem):", je_strom(g8))  # False

    # 9) Je kružnice?
    g9 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(4,1)]: g9.add_edge(*e)
    print("9a) C4 je kružnice:", je_kruznice(g9))  # True
    g9b = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g9b.add_edge(*e)
    print("9b) cesta je kružnice:", je_kruznice(g9b))  # False

    # 10) Je bipartitní?
    g10a = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g10a.add_edge(*e)
    print("10a) bipartitní (P4):", je_bipartitni(g10a))  # True
    g10b = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,1)]: g10b.add_edge(*e)  # trojúhelník
    print("10b) bipartitní (K3):", je_bipartitni(g10b))  # False

    # 11) Má trojúhelník?
    g11a = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,1)]: g11a.add_edge(*e)
    print("11a) K3 má trojúhelník:", ma_trojuhelnik(g11a))  # True
    g11b = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(4,1)]: g11b.add_edge(*e)
    print("11b) C4 má trojúhelník:", ma_trojuhelnik(g11b))  # False

    # 12) Kružnice Cₙ (stavba) + kontrola
    g12 = kruznice_Cn(5)
    print("12) C5 je kružnice:", je_kruznice(g12))  # True

    # 13) Cesta Pₙ (stavba)
    g13 = cesta_Pn(4)
    print("13) stupně P4:", stupne_seznam(g13))  # [1,2,2,1]

    # 14) Hvězda Sₙ
    g14 = hvezda_Sn(5)  # centrum=1
    print("14) listy S5:", sorted(listy(g14)))  # [2,3,4,5]

    # 15) Kolo Wₙ
    g15 = kolo_Wn(4)   # centrum 0 + C4
    # stupeň středu 0 by měl být 4
    deg0 = len(g15.node(0).neighbor_ids)
    print("15) kolo W4, deg(0):", deg0)  # 4

    # 16) Doplněk grafu
    g16 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3)]: g16.add_edge(*e)  # na 1..3 doplněk je hrana (1,3)
    H16 = doplnek_grafu(g16)
    print("16) doplněk na 1..3 má sousedy 1:", sorted(H16.node(1).neighbor_ids))  # [3]

    # 17) Čtverec grafu G²
    g17 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: g17.add_edge(*e)  # P4
    H17 = ctverec_grafu(g17)
    n1 = set(H17.node(1).neighbor_ids)
    n4 = set(H17.node(4).neighbor_ids)
    print("17) G²: sousedi(1):", sorted(n1), " | sousedi(4):", sorted(n4))  # 1 má 2 i 3; 4 má 3 i 2

    # 18) Hranový (line) graf
    g18 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(3,5)]: g18.add_edge(*e)
    L18 = line_graph(g18)
    print("18) |V(L)| =", len(list(L18.node_ids())))  # 4 (tolik, kolik je hran G)

    # 19) Transpozice digrafu
    dg19 = Graph(GraphType.DIRECTED)
    for e in [(1,2),(2,3)]: dg19.add_edge(*e)
    H19 = transpozice(dg19)
    rev_edges = []
    for u in H19.node_ids():
        for v in H19.node(u).neighbor_ids:
            rev_edges.append((u,v))
    print("19) transpozice hran:", sorted(rev_edges))  # [(2,1),(3,2)]

    # 20) Zdroje a sinky (digraf)
    dg20 = Graph(GraphType.DIRECTED)
    for e in [(1,2),(1,3)]: dg20.add_edge(*e)
    dg20.add_node(4)  # izolovaný
    print("20) zdroje/sinky:", zdroje_a_sinky(dg20))  # ([1,4], [2,3,4])

    # 21) Slabé komponenty (digraf → ignoruj orientaci)
    dg21 = Graph(GraphType.DIRECTED)
    for e in [(1,2)]: dg21.add_edge(*e)
    dg21.add_node(3)
    print("21) slabé komponenty:", pocet_slabych_komponent(dg21))  # 2

    # 22) Topologické pořadí (DAG vs. cyklus)
    dg22a = Graph(GraphType.DIRECTED)
    for e in [(1,2),(1,3),(3,4)]: dg22a.add_edge(*e)
    print("22a) topo pořadí (DAG):", topologicke_poradi(dg22a))  # např. [1,3,4,2] apod.
    dg22b = Graph(GraphType.DIRECTED)
    for e in [(1,2),(2,1)]: dg22b.add_edge(*e)  # cyklus
    print("22b) topo pořadí (cyklus):", topologicke_poradi(dg22b))  # None



#Úloha 1 — Izolované uzly v orientovaném grafu
def izolovane_uzly(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    V = list(dg.node_ids())
    indeg = {u:0 for u in V}; outdeg = {u:0 for u in V}
    for u in V:
        for v in dg.node(u).neighbor_ids:
            outdeg[u] += 1
            indeg[v]  = indeg.get(v,0) + 1
    return [u for u in V if indeg.get(u,0)==0 and outdeg.get(u,0)==0]
#Úloha 2 — Úplný graf Kₙ + test
def uplny_graf(n):
    g = Graph(GraphType.UNDIRECTED)
    for i in range(n): g.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j)
    return g

def testuj_uplnost(g):
    if g.type != GraphType.UNDIRECTED: return False
    V = list(g.node_ids()); n = len(V)
    return all(len(g.node(u).neighbor_ids) == n-1 for u in V)
#Úloha 3 — Je strom? (neorientovaný)
def je_strom(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    V = list(g.node_ids())
    if not V: return True
    m = 0
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u <= v: m += 1
    if m != len(V)-1: return False
    seen, st = {V[0]}, [V[0]]
    while st:
        u = st.pop()
        for v in g.node(u).neighbor_ids:
            if v not in seen:
                seen.add(v); st.append(v)
    return len(seen) == len(V)
#Úloha 4 — WeightedGraph (vážený orientovaný graf z matice)
class WeightedGraph:
    def __init__(self, W):
        n = len(W)
        self.g = Graph(GraphType.DIRECTED)
        self.w = {}  # (u,v) -> float
        for i in range(n): self.g.add_node(i)
        for i in range(n):
            for j in range(n):
                x = W[i][j]
                if x != -1:
                    if x < 0: raise ValueError("Váha >= 0 nebo -1.")
                    self.g.add_edge(i, j)
                    self.w[(i, j)] = float(x)

    def get_graph(self):
        return self.g

    def component_count(self):
        V = list(self.g.node_ids()); seen=set(); comp=0
        for s in V:
            if s in seen: continue
            comp += 1
            st = [s]; seen.add(s)
            while st:
                u = st.pop()
                nbrs = set(self.g.node(u).neighbor_ids)
                for x in V:
                    if u in self.g.node(x).neighbor_ids:
                        nbrs.add(x)
                for v in nbrs:
                    if v not in seen:
                        seen.add(v); st.append(v)
        return comp

    def get_sorted_weights(self):
        return iter(sorted(self.w.values()))
#if __name__ == "__main__":
    # Úloha 1
    dg = Graph(GraphType.DIRECTED)
    dg.add_edge(5,2); dg.add_edge(2,3); dg.add_node(1); dg.add_node(4)
    print("A1 izolované:", sorted(izolovane_uzly(dg)))  # [1,4]

    # Úloha 2
    gK = uplny_graf(4)
    print("A2 K4 úplný?", testuj_uplnost(gK))  # True

    # Úloha 3
    gT = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4)]: gT.add_edge(*e)
    print("A3 je_strom(P4):", je_strom(gT))   # True
    gT.add_edge(2,4)
    print("A3 je_strom(+cyklus):", je_strom(gT))  # False

    # Úloha 4
    W = [[-1,2,-1],
         [-1,-1,5],
         [ 1,-1,-1]]
    WG = WeightedGraph(W)
    print("A4 weights:", list(WG.get_sorted_weights()))  # [1.0,2.0,5.0]
    print("A4 slabé komponenty:", WG.component_count())  # 1


#Úloha 1 — Σ in-degree = Σ out-degree = |E| (orientovaný)
def soucet_in_out(dg):
    if dg.type != GraphType.DIRECTED:
        raise ValueError("Čekám digraf.")
    indeg={}; outdeg={}; m=0
    for u in dg.node_ids():
        for v in dg.node(u).neighbor_ids:
            m += 1
            outdeg[u] = outdeg.get(u,0) + 1
            indeg[v]  = indeg.get(v,0)  + 1
    si = sum(indeg.values()); so = sum(outdeg.values())
    return (si == so == m, si, so, m)
#Úloha 2 — Je kružnice (neorientovaný)
from collections import deque

def je_kruznice(g):
    if g.type != GraphType.UNDIRECTED:
        return False
    V = list(g.node_ids())
    if len(V) < 3: return False
    E, seen = [], set()
    for u in V:
        for v in g.node(u).neighbor_ids:
            if u == v: return False
            a = (u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); E.append(a)
    if len(E) != len(V): return False
    adj = {u:set() for u in V}
    for u,v in E:
        adj[u].add(v); adj[v].add(u)
    if any(len(adj[u]) != 2 for u in V): return False
    q, seenV = deque([V[0]]), {V[0]}
    while q:
        x=q.popleft()
        for y in adj[x]:
            if y not in seenV: seenV.add(y); q.append(y)
    return len(seenV) == len(V)
#Úloha 3 — Hranový (line) graf L(G) (neorientovaný vstup)
import itertools

def line_graph(g):
    if g.type != GraphType.UNDIRECTED:
        raise ValueError("Neorientovaný.")
    base, seen = [], set()
    for u in g.node_ids():
        for v in g.node(u).neighbor_ids:
            a = (u,v) if u<=v else (v,u)
            if a not in seen: seen.add(a); base.append(a)
    L = Graph(GraphType.UNDIRECTED)
    for e in base: L.add_node(e)  # aby i izolované vrcholy L(G) existovaly
    for e1, e2 in itertools.combinations(base, 2):
        if set(e1) & set(e2):
            L.add_edge(e1, e2)
    return L
#Úloha 4 — MetricSpaceGraph (neorientovaný, trojúhelníková nerovnost)
class MetricSpaceGraph:
    def __init__(self, n):
        self.g = Graph(GraphType.UNDIRECTED)
        for i in range(n): self.g.add_node(i)
        self.dist = {}  # (min(u,v),max(u,v)) -> float

    def _k(self,a,b): return (a,b) if a<=b else (b,a)

    def addEdge(self, u, v, d):
        if u == v or d <= 0:
            raise ValueError("Bez smyček a distance > 0.")
        best = None
        for w in self.g.node_ids():
            d1 = self.dist.get(self._k(u,w))
            d2 = self.dist.get(self._k(w,v))
            if d1 is not None and d2 is not None:
                s = d1 + d2
                best = s if best is None or s < best else best
        if best is not None and d > best:
            raise ValueError("Porušení trojúhelníkové nerovnosti.")
        self.g.add_edge(u, v)
        self.dist[self._k(u,v)] = float(d)

    def getDistanceMatrix(self):
        V = sorted(self.g.node_ids()); n = len(V)
        M = [[-1.0]*n for _ in range(n)]
        for i in range(n): M[i][i] = 0.0
        for (u,v),d in self.dist.items():
            i = V.index(u); j = V.index(v)
            M[i][j] = M[j][i] = d
        return M
#if __name__ == "__main__":
    # Úloha 1
    dgB = Graph(GraphType.DIRECTED)
    for e in [(1,2),(1,3),(2,3)]: dgB.add_edge(*e)
    print("B1 součet in/out:", soucet_in_out(dgB))  # (True,3,3,3)

    # Úloha 2
    gB2 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(4,1)]: gB2.add_edge(*e)
    print("B2 C4 je kružnice:", je_kruznice(gB2))  # True

    # Úloha 3
    gB3 = Graph(GraphType.UNDIRECTED)
    for e in [(1,2),(2,3),(3,4),(3,5)]: gB3.add_edge(*e)
    L = line_graph(gB3)
    print("B3 |V(L)| =", len(list(L.node_ids())))  # 4

    # Úloha 4
    M = MetricSpaceGraph(4)
    M.addEdge(0,1,2); M.addEdge(1,3,5); M.addEdge(0,2,4); M.addEdge(2,3,2)
    M.addEdge(0,3,6)  # OK (6 <= 2+5 nebo 4+2)
    print("B4 matice vzdáleností:", M.getDistanceMatrix())



#3) Úloha — MetricSpaceGraph (neorientovaný, trojúhelníková nerovnost)
class MetricSpaceGraph:
    def __init__(self, n):
        self.g = Graph(GraphType.UNDIRECTED)
        for i in range(n): self.g.add_node(i)
        self.dist = {}  # (min(u,v),max(u,v)) -> float
    def _k(self,a,b): return (a,b) if a<=b else (b,a)
    def addEdge(self, u, v, d):
        if u == v or d <= 0:
            raise ValueError("Bez smyček, distance > 0.")
        best = None
        for w in self.g.node_ids():
            d1 = self.dist.get(self._k(u,w))
            d2 = self.dist.get(self._k(w,v))
            if d1 is not None and d2 is not None:
                s = d1 + d2
                best = s if best is None or s < best else best
        if best is not None and d > best:
            raise ValueError("Porušení trojúhelníkové nerovnosti.")
        self.g.add_edge(u, v)
        self.dist[self._k(u,v)] = float(d)
    def getDistanceMatrix(self):
        V = sorted(self.g.node_ids()); n = len(V)
        M = [[-1.0]*n for _ in range(n)]
        for i in range(n): M[i][i] = 0.0
        for (u,v),d in self.dist.items():
            i = V.index(u); j = V.index(v)
            M[i][j] = M[j][i] = d
        return M



M = MetricSpaceGraph(4)
M.addEdge(0,1,2); M.addEdge(1,3,5); M.addEdge(0,2,4); M.addEdge(2,3,2)
M.addEdge(0,3,6)  # OK (6 ≤ 2+5 i 4+2)
print(M.getDistanceMatrix())











