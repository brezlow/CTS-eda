using System;
using System.Collections.Generic;
using System.Linq;

namespace KSplittingNamespace
{
    

    public class Edge : IComparable<Edge>
    {
        public Node Node1 { get; }
        public Node Node2 { get; }
        public double Weight { get; }

        public Edge(Node node1, Node node2)
        {
            Node1 = node1;
            Node2 = node2;
            Weight = Math.Abs(node1.X - node2.X) + Math.Abs(node1.Y - node2.Y); // 曼哈顿距离
        }

        public int CompareTo(Edge other)
        {
            return Weight.CompareTo(other.Weight);
        }
    }

    public class KSplittingClustering
    {
        private readonly List<Node> nodes;
        private readonly double alpha;
        private readonly int width, length, obstacleArea, maxFanout, maxNetRC;

        public KSplittingClustering(List<Node> nodes, int width, int length, int obstacleArea, double alpha, int maxFanout, int maxNetRC)
        {
            this.nodes = nodes;
            this.width = width;
            this.length = length;
            this.obstacleArea = obstacleArea;
            this.alpha = alpha;
            this.maxFanout = maxFanout;
            this.maxNetRC = maxNetRC;
        }

        /// <summary>
        /// 执行K分裂聚类算法
        /// </summary>
        /// <returns>聚类结果</returns>
        public List<List<Node>> ExecuteClustering()
        {
            var edges = BuildCompleteGraph();
            var mstEdges = KruskalMST(edges);
            double EL = CalculateEL();

            // 切割边以形成聚类团
            mstEdges.Sort((a, b) => b.Weight.CompareTo(a.Weight)); // 按边长降序排列
            var clusters = CutEdges(mstEdges, EL);

            // 检查并修正
            clusters = CheckAndFixClusters(clusters);

            return clusters;
        }

        /// <summary>
        /// 构建完全图
        /// </summary>
        /// <returns>边列表</returns>
        private List<Edge> BuildCompleteGraph()
        {
            var edges = new List<Edge>();
            for (int i = 0; i < nodes.Count; i++)
            {
                for (int j = i + 1; j < nodes.Count; j++)
                {
                    edges.Add(new Edge(nodes[i], nodes[j]));
                }
            }
            return edges;
        }

        /// <summary>
        /// 使用Kruskal算法生成最小生成树
        /// </summary>
        /// <param name="edges">边列表</param>
        /// <returns>最小生成树的边列表</returns>
        private List<Edge> KruskalMST(List<Edge> edges)
        {
            var mstEdges = new List<Edge>();
            var unionFind = new UnionFind(nodes.Count);
            edges.Sort();

            foreach (var edge in edges)
            {
                if (unionFind.Union(edge.Node1.Id, edge.Node2.Id))
                {
                    mstEdges.Add(edge);
                }
            }

            return mstEdges;
        }

        /// <summary>
        /// 计算EL值
        /// </summary>
        /// <returns>EL值</returns>
        private double CalculateEL()
        {
            int numRegisters = nodes.Count;
            return alpha * Math.Sqrt((width * length - obstacleArea) / (double)numRegisters);
        }

        /// <summary>
        /// 切割边以形成聚类团
        /// </summary>
        /// <param name="edges">边列表</param>
        /// <param name="EL">EL值</param>
        /// <returns>聚类结果</returns>
        private List<List<Node>> CutEdges(List<Edge> edges, double EL)
        {
            var clusters = new List<List<Node>>();
            var mst = new Dictionary<Node, List<Node>>();

            foreach (var edge in edges)
            {
                if (edge.Weight <= EL)
                {
                    if (!mst.ContainsKey(edge.Node1)) mst[edge.Node1] = new List<Node>();
                    if (!mst.ContainsKey(edge.Node2)) mst[edge.Node2] = new List<Node>();
                    mst[edge.Node1].Add(edge.Node2);
                    mst[edge.Node2].Add(edge.Node1);
                }
            }

            var visited = new HashSet<Node>();
            foreach (var node in nodes)
            {
                if (!visited.Contains(node))
                {
                    var cluster = new List<Node>();
                    DFS(node, mst, visited, cluster);
                    clusters.Add(cluster);
                }
            }

            return clusters;
        }

        /// <summary>
        /// 深度优先搜索
        /// </summary>
        /// <param name="node">当前节点</param>
        /// <param name="mst">最小生成树</param>
        /// <param name="visited">已访问节点</param>
        /// <param name="cluster">当前聚类</param>
        private void DFS(Node node, Dictionary<Node, List<Node>> mst, HashSet<Node> visited, List<Node> cluster)
        {
            visited.Add(node);
            cluster.Add(node);

            if (mst.ContainsKey(node))
            {
                foreach (var neighbor in mst[node])
                {
                    if (!visited.Contains(neighbor))
                    {
                        DFS(neighbor, mst, visited, cluster);
                    }
                }
            }
        }

        /// <summary>
        /// 检查并修正聚类
        /// </summary>
        /// <param name="clusters">聚类结果</param>
        /// <returns>修正后的聚类结果</returns>
        private List<List<Node>> CheckAndFixClusters(List<List<Node>> clusters)
        {
            var validClusters = new List<List<Node>>();

            foreach (var cluster in clusters)
            {
                if (cluster.Count > maxFanout || CalculateClusterRC(cluster) > maxNetRC)
                {
                    var newClusters = SplitCluster(cluster);
                    validClusters.AddRange(newClusters);
                }
                else
                {
                    validClusters.Add(cluster);
                }
            }

            return validClusters;
        }

        /// <summary>
        /// 计算聚类的RC值
        /// </summary>
        /// <param name="cluster">聚类</param>
        /// <returns>RC值</returns>
        private double CalculateClusterRC(List<Node> cluster)
        {
            double rc = 0;
            foreach (var node in cluster)
            {
                double dx = Math.Abs(node.X - cluster[0].X); // 曼哈顿距离
                double dy = Math.Abs(node.Y - cluster[0].Y);
                rc += 0.5 * (dx * dx + dy * dy);
            }
            return rc;
        }

        /// <summary>
        /// 分裂聚类
        /// </summary>
        /// <param name="cluster">聚类</param>
        /// <returns>分裂后的聚类</returns>
        private List<List<Node>> SplitCluster(List<Node> cluster)
        {
            var edges = BuildCompleteGraphForCluster(cluster);
            var mstEdges = KruskalMST(edges);
            mstEdges.Sort((a, b) => b.Weight.CompareTo(a.Weight));

            var newClusters = CutEdges(mstEdges, mstEdges[0].Weight);
            return newClusters;
        }

        /// <summary>
        /// 为聚类构建完全图
        /// </summary>
        /// <param name="cluster">聚类</param>
        /// <returns>边列表</returns>
        private List<Edge> BuildCompleteGraphForCluster(List<Node> cluster)
        {
            var edges = new List<Edge>();
            for (int i = 0; i < cluster.Count; i++)
            {
                for (int j = i + 1; j < cluster.Count; j++)
                {
                    edges.Add(new Edge(cluster[i], cluster[j]));
                }
            }
            return edges;
        }
    }

    /// <summary>
    /// 并查集数据结构
    /// </summary>
    public class UnionFind
    {
        private int[] parent;
        private int[] rank;

        public UnionFind(int size)
        {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++)
            {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        public int Find(int x)
        {
            if (parent[x] != x)
            {
                parent[x] = Find(parent[x]);
            }
            return parent[x];
        }

        public bool Union(int x, int y)
        {
            int rootX = Find(x);
            int rootY = Find(y);

            if (rootX != rootY)
            {
                if (rank[rootX] > rank[rootY])
                {
                    parent[rootY] = rootX;
                }
                else if (rank[rootX] < rank[rootY])
                {
                    parent[rootX] = rootY;
                }
                else
                {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
                return true;
            }
            return false;
        }
    }
}
