using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
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
        private CircuitData circuitData;
        private List<Node> originalNodes; // 全局数据结构，用于存储原始的 Node 信息

        public KSplittingClustering(List<Node> nodes, int width, int length, int obstacleArea, double alpha, int maxFanout, int maxNetRC, int maxEdgesPerNode, CircuitData circuitData)
        {
            this.nodes = nodes;
            this.width = width;
            this.length = length;
            this.obstacleArea = obstacleArea;
            this.alpha = alpha;
            this.maxFanout = maxFanout;
            this.maxNetRC = maxNetRC;
            this.maxEdgesPerNode = maxEdgesPerNode;
            this.circuitData = circuitData;
            this.originalNodes = new List<Node>(); // 初始化全局数据结构

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

            // 计算各个聚类团的“中心点”，放置缓冲器
            clusters = CheckRCValue(clusters);
            // var bufferInstances = PlaceBuffers(clusters);
            // Console.WriteLine($"放置缓冲器数目: {bufferInstances.Count}");
            Console.WriteLine($"放置缓冲器数目: {clusters.Count}");

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
            Console.WriteLine($"原始聚类数: {clusters.Count}");
            var validClusters = new List<List<Node>>();
            int clusterNumber = clusters.Count;

            foreach (var cluster in clusters)
            {
                // 先分裂超出扇出的聚类
                if (cluster.Count > maxFanout)
                {
                    var splitClusters = SplitCluster(cluster);
                    foreach (var subCluster in splitClusters)
                    {
                        if (subCluster.Count <= maxFanout)
                        {
                            validClusters.Add(subCluster);
                        }
                        else
                        {
                            validClusters.AddRange(SplitCluster(subCluster));
                        }
                    }
                }
                else
                {
                    validClusters.Add(cluster);
                }
            }

            // 合并较小的聚类团
            // validClusters = MergeSmallClusters(validClusters);

            return validClusters;
        }

        private List<List<Node>> MergeSmallClusters(List<List<Node>> clusters)
        {
            var mergedClusters = new List<List<Node>>();
            var smallClusters = new List<List<Node>>();

            foreach (var cluster in clusters)
            {
                if (cluster.Count < maxFanout / 2)
                {
                    smallClusters.Add(cluster);
                }
                else
                {
                    mergedClusters.Add(cluster);
                }
            }

            // 合并小的聚类团
            while (smallClusters.Count > 0)
            {
                var cluster = smallClusters[0];
                smallClusters.RemoveAt(0);

                foreach (var smallCluster in smallClusters.ToList())
                {
                    if (cluster.Count + smallCluster.Count <= maxFanout)
                    {
                        cluster.AddRange(smallCluster);
                        smallClusters.Remove(smallCluster);
                    }
                }

                mergedClusters.Add(cluster);
            }

            return mergedClusters;
        }




        private List<List<Node>> SplitCluster(List<Node> cluster)
        {
            // 计算边集并尝试构建最小生成树（MST）
            var edges = BuildSparseGraphForCluster(cluster);
            var mstEdges = KruskalMST(edges);

            // 根据 MST 中的最长边尝试分裂
            double maxWeight = mstEdges.Max(edge => edge.Weight);
            var newClusters = CutEdges(mstEdges, maxWeight);

            // 如果分裂后结果与原始聚类团相似，避免无意义的分裂，直接返回原聚类团
            if (newClusters.Count == 1 && newClusters[0].Count == cluster.Count)
            {
                return new List<List<Node>> { cluster };
            }

            // 返回分裂后的新聚类团
            return newClusters.Count > 0 ? newClusters : new List<List<Node>> { cluster };
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


        private List<List<Node>> CheckRCValue(List<List<Node>> clusters)
        {
            var validClusters = new List<List<Node>>();
            int clusterNumber = 0;

            foreach (var cluster in clusters)
            {
                clusterNumber++;
                var buffer = GetCentetPointPosition(cluster);
                var bufferLoad = CalculateBufferLoad(cluster, buffer);

                if (bufferLoad <= maxNetRC)
                {
                    validClusters.Add(cluster);
                }
                else
                {
                    // 如果 RC 值超过阈值，对该聚类分裂
                    Console.WriteLine($"聚类：{clusterNumber}距离{bufferLoad}发生一次分裂");
                    var splitClusters = SplitCluster(cluster);
                    var checkedClusters = CheckRCValue(splitClusters);
                    validClusters.AddRange(checkedClusters);
                }
            }

            return validClusters;
        }

        public Node GetCentetPointPosition(List<Node> cluster)
        {

            var clustering = new CenterPointNamespace.Clustering();

            var CenterPointPosition = clustering.CalculateBottomLevelCenterPoint(cluster, circuitData.BufferSize.Width, circuitData.BufferSize.Height);

            // 检查缓冲器位置是否与已有元件重叠
            if (IsOverlapping(CenterPointPosition))
            {
                // 如果重叠，尝试在附近找到一个不重叠的位置
                CenterPointPosition = FindNonOverlappingPosition(CenterPointPosition);
            }

            return CenterPointPosition;
        }


        public List<BufferInstance> PlaceBuffers(List<List<Node>> clusters)
        {
            var bufferInstances = new List<BufferInstance>();
            var clustering = new CenterPointNamespace.Clustering();

            foreach (var cluster in clusters)
            {
                var centerPoint = clustering.CalculateBottomLevelCenterPoint(cluster, circuitData.BufferSize.Width, circuitData.BufferSize.Height);

                // 检查缓冲器位置是否与已有元件重叠
                if (IsOverlapping(centerPoint))
                {
                    // 如果重叠，尝试在附近找到一个不重叠的位置
                    centerPoint = FindNonOverlappingPosition(centerPoint);
                }

                var bufferInstance = new BufferInstance
                {
                    Name = $"BUF_{bufferInstances.Count + 1}",
                    Position = (centerPoint.X, centerPoint.Y),
                    ContainedNodes = new List<Node>(cluster) // 记录聚类团中的元件
                };
                bufferInstances.Add(bufferInstance);

                // 将原始的 Node 信息存储到全局数据结构中
                originalNodes.AddRange(cluster);

                // 将中心点放置到一个新的 Node 中
                var newNode = new Node(centerPoint.X, centerPoint.Y, 0, bufferInstance.Name.GetHashCode(), circuitData.BufferSize.Width, circuitData.BufferSize.Height);
                nodes.Add(newNode);
            }

            return bufferInstances;
        }
        private bool IsOverlapping(Node centerPoint)
        {
            foreach (var ff in circuitData.FFInstances)
            {
                if (IsOverlapping(centerPoint, new Node(ff.Position.X, ff.Position.Y, 0, 0, circuitData.FFSize.Width, circuitData.FFSize.Height)))
                {
                    return true;
                }
            }


            foreach (var buffer in circuitData.BufferInstances)
            {
                if (IsOverlapping(centerPoint, new Node(buffer.Position.X, buffer.Position.Y, 0, 0, circuitData.BufferSize.Width, circuitData.BufferSize.Height)))
                {
                    return true;
                }
            }

            return false;
        }

        private bool IsOverlapping(Node node1, Node node2)
        {
            return !(node1.X + node1.Width <= node2.X || node2.X + node2.Width <= node1.X ||
                     node1.Y + node1.Height <= node2.Y || node2.Y + node2.Height <= node1.Y);
        }

        private Node FindNonOverlappingPosition(Node centerPoint)
        {
            int step = 10; // 步长，可以根据需要调整
            for (int dx = -step; dx <= step; dx += step)
            {
                for (int dy = -step; dy <= step; dy += step)
                {
                    var newCenterPoint = new Node(centerPoint.X + dx, centerPoint.Y + dy, 0, 0, centerPoint.Width, centerPoint.Height);
                    if (!IsOverlapping(newCenterPoint))
                    {
                        return newCenterPoint;
                    }
                }
            }

            // 如果找不到不重叠的位置，返回原位置
            return centerPoint;
        }
        private double CalculateBufferLoad(List<Node> cluster, Node buffer)
        {
            double rc = circuitData.NetUnitR * circuitData.NetUnitC;
            double load = 0.0;

            foreach (var node in cluster)
            {
                double distance = CalculateManhattanDistance(buffer, node);
                load += Math.Pow(distance, 2);
            }

            return 0.5 * rc * load;
        }
        private double CalculateManhattanDistance(Node node1, Node node2)
        {
            double centerX1 = node1.X + node1.Width / 2.0;
            double centerY1 = node1.Y + node1.Height / 2.0;
            double centerX2 = node2.X + node2.Width / 2.0;
            double centerY2 = node2.Y + node2.Height / 2.0;

            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
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
                public required Node Data { get; set; }
                public required KDNode Left { get; set; }
                public required KDNode Right { get; set; }
                public bool VerticalSplit { get; set; }
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
                int medianIndex = nodes.Count / 2;

                return new KDNode
                {
                    Data = nodes[medianIndex],
                    VerticalSplit = vertical,
                    Left = Build(nodes.Take(medianIndex).ToList(), !vertical),
                    Right = Build(nodes.Skip(medianIndex + 1).ToList(), !vertical)
                };
            }


            public void NearestNeighbors(Node node, int maxNeighbors, List<Edge> edges)
            {
                NearestNeighbors(root, node, maxNeighbors, edges, double.MaxValue);
            }

            private void NearestNeighbors(KDNode kdNode, Node node, int maxNeighbors, List<Edge> edges, double maxDistance)
            {
                if (kdNode == null) return;

                double distance = CalculateManhattanDistance(node, kdNode.Data);
                if (edges.Count < maxNeighbors || distance < maxDistance)
                {
                    edges.Add(new Edge(node, kdNode.Data));
                    if (edges.Count > maxNeighbors)
                    {
                        edges.Sort((a, b) => a.Weight.CompareTo(b.Weight));
                        edges.RemoveAt(edges.Count - 1);
                        maxDistance = edges.Last().Weight;
                    }
                }

                KDNode primary = kdNode.VerticalSplit
                    ? (node.X < kdNode.Data.X ? kdNode.Left : kdNode.Right)
                    : (node.Y < kdNode.Data.Y ? kdNode.Left : kdNode.Right);
                KDNode secondary = primary == kdNode.Left ? kdNode.Right : kdNode.Left;

                NearestNeighbors(primary, node, maxNeighbors, edges, maxDistance);
                if ((kdNode.VerticalSplit ? Math.Abs(node.X - kdNode.Data.X) : Math.Abs(node.Y - kdNode.Data.Y)) < maxDistance)
                {
                    NearestNeighbors(secondary, node, maxNeighbors, edges, maxDistance);
                }
            }

            private double CalculateManhattanDistance(Node node1, Node node2)
            {
                double centerX1 = node1.X + node1.Width / 2.0;
                double centerY1 = node1.Y + node1.Height / 2.0;
                double centerX2 = node2.X + node2.Width / 2.0;
                double centerY2 = node2.Y + node2.Height / 2.0;

                return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
            }


        }
    }
}