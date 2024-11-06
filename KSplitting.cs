using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
            Weight = CalculateManhattanDistance(node1, node2);
        }

        private double CalculateManhattanDistance(Node node1, Node node2)
        {

            double centerX1 = node1.X + node1.Width / 2.0;
            double centerY1 = node1.Y + node1.Height / 2.0;
            double centerX2 = node2.X + node2.Width / 2.0;
            double centerY2 = node2.Y + node2.Height / 2.0;

            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
        }

        public int CompareTo(Edge? other) => Weight.CompareTo(other?.Weight ?? 0);
    }

    public class KSplittingClustering
    {
        private readonly List<Node> nodes;
        private readonly int width, length, obstacleArea, maxFanout, maxNetRC;
        private readonly double alpha;
        private readonly int maxEdgesPerNode;
        private KDTree kdTree;

        public KSplittingClustering(List<Node> nodes, int width, int length, int obstacleArea, double alpha, int maxFanout, int maxNetRC, int maxEdgesPerNode)
        {
            this.nodes = nodes;
            this.width = width;
            this.length = length;
            this.obstacleArea = obstacleArea;
            this.alpha = alpha;
            this.maxFanout = maxFanout;
            this.maxNetRC = maxNetRC;
            this.maxEdgesPerNode = maxEdgesPerNode;

            // 创建 KD 树以便高效查找最近邻节点
            kdTree = new KDTree(nodes);
        }

        public List<List<Node>> ExecuteClustering()
        {
            var edges = BuildSparseGraph();
            Console.WriteLine($"图边数: {edges.Count}");

            var mstEdges = KruskalMST(edges);
            Console.WriteLine($"MST 边数: {mstEdges.Count}");
            double EL = CalculateEL();
            Console.WriteLine($"EL: {EL}");

            var clusters = CutEdges(mstEdges, EL);
            Console.WriteLine($"聚类数: {clusters.Count}");
            clusters = CheckAndFixClusters(clusters);
            Console.WriteLine($"检查后聚类数:{clusters.Count}");

            return clusters;
        }

        private List<Edge> BuildSparseGraph()
        {
            var edges = new List<Edge>();

            // 使用 KD 树找到每个节点的最近邻节点并建立边，避免完全图的构建
            Parallel.ForEach(nodes, node =>
            {
                var nearestEdges = new List<Edge>();
                kdTree.NearestNeighbors(node, maxEdgesPerNode, nearestEdges);
                lock (edges)
                {
                    edges.AddRange(nearestEdges);
                }
            });

            return edges;
        }

        private List<Edge> KruskalMST(List<Edge> edges)
        {
            edges.Sort();
            var mstEdges = new List<Edge>();
            var unionFind = new UnionFind(nodes.Count);

            // 并行构建 MST 树的边集合
            Parallel.ForEach(edges, edge =>
            {
                if (unionFind.Union(edge.Node1.Id, edge.Node2.Id))
                {
                    lock (mstEdges)
                    {
                        mstEdges.Add(edge);
                    }
                }
            });

            return mstEdges;
        }

        private double CalculateEL()
        {
            int numRegisters = nodes.Count;
            return alpha * Math.Sqrt((width * length - obstacleArea) / (double)numRegisters);
        }

        private List<List<Node>> CutEdges(List<Edge> edges, double EL)
        {
            var clusters = new List<List<Node>>();
            var mst = new Dictionary<Node, List<Node>>();

            // 构建初步聚类图并切割权重高于 EL 的边
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

            // DFS 深度优先搜索形成聚类团
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

        private void DFS(Node node, Dictionary<Node, List<Node>> mst, HashSet<Node> visited, List<Node> cluster)
        {
            visited.Add(node);
            cluster.Add(node);

            if (mst.ContainsKey(node))
                foreach (var neighbor in mst[node])
                    if (!visited.Contains(neighbor))
                        DFS(neighbor, mst, visited, cluster);
        }

        private List<List<Node>> CheckAndFixClusters(List<List<Node>> clusters)
        {
            var validClusters = new List<List<Node>>();

            foreach (var cluster in clusters)
            {
                Console.WriteLine($"检查这个聚类节点中");
                if (cluster.Count > maxFanout || CalculateClusterRC(cluster) > maxNetRC)
                {
                    Console.WriteLine($"分裂不合法的聚类节点");
                    // 这里有问题，需要修复
                    // 分裂不合法的聚类
                    var newClusters = SplitCluster(cluster, 0);
                    validClusters.AddRange(newClusters);
                }
                else
                {
                    validClusters.Add(cluster);
                }
            }
            return validClusters;
        }

        private double CalculateClusterRC(List<Node> cluster)
        {
            double rc = 0;
            foreach (var node in cluster)
            {
                double dx = Math.Abs(node.X - cluster[0].X);
                double dy = Math.Abs(node.Y - cluster[0].Y);
                rc += 0.5 * (dx * dx + dy * dy);
            }
            return rc;
        }


        private List<List<Node>> SplitCluster(List<Node> cluster, int depth)
        {
            if (depth > 10) // 限制递归深度，避免无限递归
            {
                return new List<List<Node>> { cluster };
            }

            var edges = BuildSparseGraphForCluster(cluster);
            var mstEdges = KruskalMST(edges);

            // 使用权重排序找到最长边，并且先尝试分裂
            double maxWeight = mstEdges.Max(edge => edge.Weight);
            var newClusters = CutEdges(mstEdges, maxWeight);

            var validClusters = new List<List<Node>>();
            foreach (var newCluster in newClusters)
            {
                if (newCluster.Count > maxFanout || CalculateClusterRC(newCluster) > maxNetRC)
                {
                    var furtherSplitClusters = SplitCluster(newCluster, depth + 1);
                    validClusters.AddRange(furtherSplitClusters);
                }
                else
                {
                    validClusters.Add(newCluster);
                }
            }

            return validClusters;
        }


        private List<Edge> BuildSparseGraphForCluster(List<Node> cluster)
        {
            var edges = new List<Edge>();

            // 使用 KD 树找到每个节点的最近邻节点并建立边，避免完全图的构建
            Parallel.ForEach(cluster, node =>
            {
                var nearestEdges = new List<Edge>();
                kdTree.NearestNeighbors(node, maxEdgesPerNode, nearestEdges);
                lock (edges)
                {
                    edges.AddRange(nearestEdges);
                }
            });

            return edges;
        }
    }

    /// <summary>
    /// 并查集
    /// </summary>
    public class UnionFind
    {
        private int[] parent;
        private int[] rank;

        public UnionFind(int size)
        {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++) parent[i] = i;
        }

        public int Find(int x)
        {
            // 非递归实现路径压缩
            int root = x;
            while (root != parent[root])
            {
                root = parent[root];
            }
            // 路径压缩
            while (x != root)
            {
                int next = parent[x];
                parent[x] = root;
                x = next;
            }
            return root;
        }

        public bool Union(int x, int y)
        {
            int rootX = Find(x);
            int rootY = Find(y);

            if (rootX != rootY)
            {
                if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
                else if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
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



    // KD 树实现，用于加速近邻查找
    public class KDTree
    {
        private class KDNode
        {
            public required Node Data;
            public required KDNode Left;
            public required KDNode Right;
            public bool VerticalSplit;
        }

        private KDNode root;

        public KDTree(List<Node> nodes)
        {
            root = Build(nodes, true);
        }

        private KDNode Build(List<Node> nodes, bool vertical)
        {
            if (nodes.Count == 0) return null;
            nodes.Sort((a, b) => vertical ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));
            int median = nodes.Count / 2;

            return new KDNode
            {
                Data = nodes[median],
                VerticalSplit = vertical,
                Left = Build(nodes.Take(median).ToList(), !vertical),
                Right = Build(nodes.Skip(median + 1).ToList(), !vertical)
            };
        }

        public void NearestNeighbors(Node node, int maxNeighbors, List<Edge> edges)
        {
            NearestNeighbors(root, node, maxNeighbors, edges, double.MaxValue);
        }
        private double CalculateManhattanDistance(Node node1, Node node2)
        {

            double centerX1 = node1.X + node1.Width / 2.0;
            double centerY1 = node1.Y + node1.Height / 2.0;
            double centerX2 = node2.X + node2.Width / 2.0;
            double centerY2 = node2.Y + node2.Height / 2.0;

            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
        }


        private void NearestNeighbors(KDNode kdNode, Node node, int maxNeighbors, List<Edge> edges, double maxDistance)
        {
            if (kdNode == null) return;

            var dist = CalculateManhattanDistance(node, kdNode.Data);
            if (edges.Count < maxNeighbors || dist < maxDistance)
            {
                edges.Add(new Edge(node, kdNode.Data));
                if (edges.Count > maxNeighbors) edges.RemoveAt(edges.Count - 1);
                maxDistance = edges.Last().Weight;
            }

            // 确保不重复计算邻居
            KDNode primary = kdNode.VerticalSplit
                ? (node.X < kdNode.Data.X ? kdNode.Left : kdNode.Right)
                : (node.Y < kdNode.Data.Y ? kdNode.Left : kdNode.Right);
            KDNode secondary = primary == kdNode.Left ? kdNode.Right : kdNode.Left;

            NearestNeighbors(primary, node, maxNeighbors, edges, maxDistance);
            if ((kdNode.VerticalSplit ? Math.Abs(node.X - kdNode.Data.X) : Math.Abs(node.Y - kdNode.Data.Y)) < maxDistance)
                NearestNeighbors(secondary, node, maxNeighbors, edges, maxDistance);
        }

    }
}

