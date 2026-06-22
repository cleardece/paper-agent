"""
Paper Agent - Knowledge Graph
Neo4j 或内存实现，存储实体关系
"""

import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger("paper-agent")


class KnowledgeGraph:
    """知识图谱 - 支持 Neo4j 或内存模式"""

    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.use_neo4j = False
        self.driver = None

        # 尝试连接 Neo4j
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                self.driver.verify_connectivity()
                self.use_neo4j = True
                logger.info("[KnowledgeGraph] Neo4j 连接成功")
            except Exception as e:
                logger.warning(f"[KnowledgeGraph] Neo4j 连接失败，使用内存模式: {e}")

        if not self.use_neo4j:
            # 内存模式
            self.nodes = {}  # id -> {type, properties}
            self.edges = []  # [(from_id, to_id, relation)]
            self.adjacency = defaultdict(list)  # from_id -> [(to_id, relation)]
            logger.info("[KnowledgeGraph] 使用内存模式")

    # ==================== 节点操作 ====================

    def add_node(self, node_id: str, node_type: str, properties: dict = None):
        """添加节点"""
        if self.use_neo4j:
            with self.driver.session() as session:
                session.run(
                    f"MERGE (n:{node_type} {{id: $id}}) SET n += $props",
                    id=node_id, props=properties or {}
                )
        else:
            self.nodes[node_id] = {"type": node_type, "properties": properties or {}}

    def get_node(self, node_id: str) -> Optional[dict]:
        """获取节点"""
        if self.use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n {id: $id}) RETURN n",
                    id=node_id
                )
                record = result.single()
                if record:
                    return dict(record["n"])
        else:
            return self.nodes.get(node_id)
        return None

    # ==================== 边操作 ====================

    def add_edge(self, from_id: str, to_id: str, relation: str, properties: dict = None):
        """添加边"""
        if self.use_neo4j:
            with self.driver.session() as session:
                session.run(
                    f"""
                    MATCH (a {{id: $from_id}})
                    MATCH (b {{id: $to_id}})
                    MERGE (a)-[r:{relation}]->(b)
                    SET r += $props
                    """,
                    from_id=from_id, to_id=to_id, props=properties or {}
                )
        else:
            self.edges.append((from_id, to_id, relation))
            self.adjacency[from_id].append((to_id, relation))

    def get_neighbors(self, node_id: str, relation: str = None) -> list[dict]:
        """获取邻居节点"""
        if self.use_neo4j:
            with self.driver.session() as session:
                if relation:
                    result = session.run(
                        f"""
                        MATCH (n {{id: $id}})-[r:{relation}]->(m)
                        RETURN m.id as id, m.type as type, type(r) as relation
                        """,
                        id=node_id
                    )
                else:
                    result = session.run(
                        """
                        MATCH (n {id: $id})-[r]->(m)
                        RETURN m.id as id, m.type as type, type(r) as relation
                        """,
                        id=node_id
                    )
                return [dict(record) for record in result]
        else:
            neighbors = []
            for to_id, rel in self.adjacency.get(node_id, []):
                if relation is None or rel == relation:
                    node = self.nodes.get(to_id, {})
                    neighbors.append({
                        "id": to_id,
                        "type": node.get("type", ""),
                        "relation": rel,
                    })
            return neighbors

    # ==================== 查询 ====================

    def find_path(self, from_id: str, to_id: str, max_depth: int = 3) -> list[list[str]]:
        """查找两个节点之间的路径"""
        if self.use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH path = (a {id: $from_id})-[*1..""" + str(max_depth) + """]->(b {id: $to_id})
                    RETURN [n IN nodes(path) | n.id] as path
                    """,
                    from_id=from_id, to_id=to_id
                )
                return [record["path"] for record in result]
        else:
            # BFS
            queue = [(from_id, [from_id])]
            visited = {from_id}
            paths = []

            while queue:
                current, path = queue.pop(0)
                if current == to_id:
                    paths.append(path)
                    continue
                if len(path) > max_depth:
                    continue
                for neighbor, _ in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            return paths

    def get_entity_chain(self, entity_type: str) -> list[dict]:
        """获取某类实体的所有节点"""
        if self.use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    f"MATCH (n:{entity_type}) RETURN n.id as id, n"
                )
                return [dict(record) for record in result]
        else:
            return [
                {"id": nid, **data}
                for nid, data in self.nodes.items()
                if data.get("type") == entity_type
            ]

    # ==================== 统计 ====================

    def stats(self) -> dict:
        """获取图谱统计"""
        if self.use_neo4j:
            with self.driver.session() as session:
                nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                edges = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                return {"nodes": nodes, "edges": edges}
        else:
            return {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
            }

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
