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
        private readonly int width, length, BufferSize_Height, BufferSize_Width, maxFanout, maxNetRC;
        private readonly double alpha, NetUnitR, NetUnitC, obstacleArea;
        private readonly int maxEdgesPerNode;
        private KDTree kdTree;

        // 缓存字典
        private Dictionary<int, List<Edge>> edgeCache = new Dictionary<int, List<Edge>>();
        private Dictionary<int, List<Edge>> mstCache = new Dictionary<int, List<Edge>>();

        private readonly List<CircuitComponent> CircuitComponents;
        private readonly List<BufferInstance> TotalBuffer;

        private readonly bool isBottomLayer;

        public KSplittingClustering(List<Node> nodes, int width, int length, int BufferSize_Height, int BufferSize_Width, double obstacleArea, double alpha, double NetUnitR, double NetUnitC, int maxFanout, int maxNetRC, int maxEdgesPerNode, List<CircuitComponent> circuitComponents, List<BufferInstance> totalBuffer, bool isBottomLayer = true)
        {
            this.nodes = nodes;
            this.width = width;
            this.length = length;
            this.NetUnitR = NetUnitR;
            this.NetUnitC = NetUnitC;
            this.BufferSize_Height = BufferSize_Height;
            this.BufferSize_Width = BufferSize_Width;
            this.obstacleArea = obstacleArea;
            this.alpha = alpha;
            this.maxFanout = maxFanout;
            this.maxNetRC = maxNetRC;
            this.maxEdgesPerNode = maxEdgesPerNode;
            this.CircuitComponents = circuitComponents;
            this.TotalBuffer = totalBuffer;
            this.isBottomLayer = isBottomLayer;

            // 创建 KD 树以便高效查找最近邻节点
            kdTree = new KDTree(nodes);
        }

        public List<Node> ExecuteClustering()
        {
            // 构建稀疏图
            var edges = BuildSparseGraph();
            Console.WriteLine($"图边数: {edges.Count}");

            // 计算最小生成树（MST）
            var mstEdges = KruskalMST(edges);
            Console.WriteLine($"MST 边数: {mstEdges.Count}");

            // 计算 EL 值
            double EL = CalculateEL();
            Console.WriteLine($"EL: {EL}");

            // 切割边以形成聚类
            var clusters = CutEdges(mstEdges, EL);
            Console.WriteLine($"聚类数: {clusters.Count}");

            // 检查并修复聚类
            clusters = CheckAndFixClusters(clusters);
            Console.WriteLine($"检查后聚类数:{clusters.Count}");

            // 如果聚类数小于等于 10，直接合并成一个大的聚类
            if (clusters.Count <= maxFanout)
            {
                bool hebin = true;
                var combinedCluster = clusters.SelectMany(cluster => cluster).ToList();
                var combinedClusters = new LinkedList<List<Node>>();
                combinedClusters.AddLast(combinedCluster);

                // 检查 RC 负载并进行必要的分裂
                var (validClusters, buffers) = ValidateClustersByRC(combinedClusters, hebin);
                var updatedBuffers = GenerateBufferInstances(validClusters, buffers, TotalBuffer);

                // 如果经过检查和分裂后，buffer 数目仍然小于等于 10，直接返回
                if (updatedBuffers.Count <= maxFanout)
                {
                    return updatedBuffers;
                }
            }

            // 计算各个聚类团的“中心点”，放置缓冲器
            var (finalValidClusters, finalBuffers) = ValidateClustersByRC(clusters);
            var finalUpdatedBuffers = GenerateBufferInstances(finalValidClusters, finalBuffers, TotalBuffer);

            return finalUpdatedBuffers;
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

            // 确保 UnionFind 对象的容量足够大
            foreach (var edge in edges)
            {
                unionFind.EnsureCapacity(Math.Max(edge.Node1.Id, edge.Node2.Id) + 1);
            }

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
            return alpha * Math.Sqrt(((long)width * (long)length - (obstacleArea)) / (double)numRegisters);
        }

        private LinkedList<List<Node>> CutEdges(List<Edge> edges, double EL)
        {
            var clusters = new LinkedList<List<Node>>();
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
                    clusters.AddLast(cluster);
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


        /// <summary>
        /// 检查聚类团是否满足扇出约束，如果不满足则分裂
        /// </summary>
        /// <param name="clusters"></param>
        /// <returns></returns>
        /// <summary>
        /// 检查聚类团是否满足扇出约束，如果不满足则分裂
        /// </summary>
        /// <param name="clusters"></param>
        /// <returns></returns>
        private LinkedList<List<Node>> CheckAndFixClusters(LinkedList<List<Node>> clusters)
        {
            Console.WriteLine($"原始聚类数: {clusters.Count}");
            var validClusters = new List<List<Node>>();

            var currentNode = clusters.First;
            while (currentNode != null)
            {
                var cluster = currentNode.Value;
                var nextNode = currentNode.Next;

                // 分裂超出扇出的聚类
                if (cluster.Count > maxFanout)
                {
                    Console.WriteLine($"聚类团数量:{cluster.Count}发生一次分裂");
                    clusters.Remove(currentNode);

                    // 使用新的分裂函数按倍数分裂
                    var splitClusters = SplitClusterByFanout(cluster, maxFanout);
                    foreach (var subCluster in splitClusters)
                    {
                        validClusters.Add(subCluster);
                    }
                }
                else
                {
                    validClusters.Add(cluster); // 满足扇出，直接加入有效列表
                }

                currentNode = nextNode;
            }

            // 将新的链表接上原本的检查后的链表
            foreach (var validCluster in validClusters)
            {
                clusters.AddLast(validCluster);
            }

            return clusters;
        }

        /// <summary>
        /// 按倍数分裂聚类，使每个子聚类的数量小于等于 maxFanout
        /// </summary>
        /// <param name="cluster">初始聚类</param>
        /// <param name="maxFanout">扇出上限</param>
        /// <returns>分裂后的聚类列表</returns>
        private List<List<Node>> SplitClusterByFanout(List<Node> cluster, int maxFanout)
        {
            int numSplits = (int)Math.Ceiling((double)cluster.Count / maxFanout); // 计算需要分裂的数量
            var splitClusters = new List<List<Node>>(numSplits);

            // 初始化子聚类
            for (int i = 0; i < numSplits; i++)
            {
                splitClusters.Add(new List<Node>());
            }

            // 将节点均匀分配到每个子聚类
            for (int i = 0; i < cluster.Count; i++)
            {
                splitClusters[i % numSplits].Add(cluster[i]);
            }

            // 为每个子聚类调整几何位置，以确保均衡且无交叉
            AdjustClusterPositions(splitClusters);

            return splitClusters;
        }

        /// <summary>
        /// 调整子聚类的位置以保持几何均衡并避免交叉
        /// </summary>
        /// <param name="splitClusters">待调整的子聚类列表</param>
        private void AdjustClusterPositions(List<List<Node>> splitClusters)
        {
            // 获取原始聚类的几何中心点
            var centerX = splitClusters.SelectMany(c => c).Average(node => node.X);
            var centerY = splitClusters.SelectMany(c => c).Average(node => node.Y);

            // 基于每个子聚类的位置偏移量（模拟辐射状分布）进行位置调整
            double angleStep = 2 * Math.PI / splitClusters.Count;
            double radius = 10.0; // 调整距离

            for (int i = 0; i < splitClusters.Count; i++)
            {
                double angle = i * angleStep;
                double offsetX = centerX + radius * Math.Cos(angle);
                double offsetY = centerY + radius * Math.Sin(angle);

                // 为子聚类中的每个节点应用位置偏移
                foreach (var node in splitClusters[i])
                {
                    node.X += (int)offsetX;
                    node.Y += (int)offsetY;
                }
            }
        }



        /// <summary>
        /// 递归分裂聚类团,检查扇出约束
        /// </summary>
        /// <param name="cluster"></param>
        /// <param name="validClusters"></param>
        /// <param name="clusters"></param>
        /// <param name="depth"></param>
        /// <param name="maxDepth"></param>
        private void SplitAndCheckCluster(List<Node> initialCluster, List<List<Node>> validClusters, LinkedList<List<Node>> clusters, int maxDepth)
        {
            Stack<(List<Node> cluster, int depth)> stack = new Stack<(List<Node>, int)>();
            stack.Push((initialCluster, 0));

            while (stack.Count > 0)
            {
                var (cluster, depth) = stack.Pop();

                if (depth > maxDepth)
                {
                    validClusters.Add(cluster);
                    ClearClusterCache(cluster);
                    continue;
                }

                var splitClusters = SplitCluster(cluster);
                cluster.Clear();
                cluster = null;

                foreach (var subCluster in splitClusters)
                {
                    if (subCluster.Count <= maxFanout)
                    {
                        validClusters.Add(subCluster);
                        ClearClusterCache(subCluster);
                    }
                    else
                    {
                        stack.Push((subCluster, depth + 1));
                        ClearClusterCache(subCluster);
                    }
                }
            }
        }


        private void ClearClusterCache(List<Node> cluster)
        {
            int clusterHash = cluster.GetHashCode();
            edgeCache.Remove(clusterHash);
            mstCache.Remove(clusterHash);
        }


        /// <summary>
        /// 分裂聚类团
        /// </summary>
        /// <param name="cluster"></param>
        /// <returns></returns>
        private LinkedList<List<Node>> SplitCluster(List<Node> cluster)
        {
            // 计算边集并尝试构建最小生成树（MST）
            var edges = GetOrBuildEdges(cluster);
            var mstEdges = GetOrBuildMST(cluster, edges);

            // 根据 MST 中的最长边尝试分裂
            double maxWeight = mstEdges.Max(edge => edge.Weight);
            var newClusters = SplitByRemovingMaxEdge(mstEdges, maxWeight);

            // 如果分裂后结果与原始聚类团相似，避免无意义的分裂，直接返回原聚类团
            if (newClusters.Count == 1 && newClusters.First().Count == cluster.Count)
            {
                return new LinkedList<List<Node>>(new[] { cluster });
            }

            // 返回分裂后的新聚类团
            return newClusters.Count > 0 ? new LinkedList<List<Node>>(newClusters) : new LinkedList<List<Node>>(new[] { cluster });
        }

        private List<List<Node>> SplitByRemovingMaxEdge(List<Edge> mstEdges, double maxWeight)
        {
            // 找到最长的边并移除
            var maxEdge = mstEdges.First(edge => edge.Weight == maxWeight);
            mstEdges.Remove(maxEdge);

            // 构建两个新的聚类
            var cluster1 = new List<Node>();
            var cluster2 = new List<Node>();
            var visited = new HashSet<Node>();

            // 使用 DFS 将节点分配到两个聚类中
            DFSForSplit(maxEdge.Node1, mstEdges, visited, cluster1);
            DFSForSplit(maxEdge.Node2, mstEdges, visited, cluster2);

            return new List<List<Node>> { cluster1, cluster2 };
        }

        private void DFSForSplit(Node node, List<Edge> edges, HashSet<Node> visited, List<Node> cluster)
        {
            var stack = new Stack<Node>();
            stack.Push(node);

            while (stack.Count > 0)
            {
                var currentNode = stack.Pop();
                if (!visited.Contains(currentNode))
                {
                    visited.Add(currentNode);
                    cluster.Add(currentNode);

                    foreach (var edge in edges)
                    {
                        if (edge.Node1 == currentNode && !visited.Contains(edge.Node2))
                        {
                            stack.Push(edge.Node2);
                        }
                        else if (edge.Node2 == currentNode && !visited.Contains(edge.Node1))
                        {
                            stack.Push(edge.Node1);
                        }
                    }
                }
            }
        }

        private List<Edge> GetOrBuildEdges(List<Node> cluster)
        {
            int clusterHash = cluster.GetHashCode(); // 获取 cluster 的哈希值
            if (edgeCache.TryGetValue(clusterHash, out List<Edge> cachedEdges))
            {
                return cachedEdges; // 使用缓存的边集
            }

            // 如果没有缓存，构建边集并缓存
            List<Edge> edges = BuildSparseGraphForCluster(cluster);
            edgeCache[clusterHash] = edges;
            return edges;
        }

        private List<Edge> GetOrBuildMST(List<Node> cluster, List<Edge> edges)
        {
            int clusterHash = cluster.GetHashCode(); // 获取 cluster 的哈希值
            if (mstCache.TryGetValue(clusterHash, out List<Edge> cachedMST))
            {
                return cachedMST; // 使用缓存的 MST
            }

            // 如果没有缓存，计算 MST 并缓存
            List<Edge> mst = KruskalMST(edges);
            mstCache[clusterHash] = mst;
            return mst;
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

        /// <summary>
        /// 检查每个聚类的RC负载是否符合要求，不符合则递归分裂，返回符合要求的聚类列表及其对应的buffer节点列表
        /// </summary>
        /// <param name="clusters">输入的聚类列表</param>
        /// <param name="depth">当前递归深度</param>
        /// <returns>元组：符合RC负载要求的聚类列表和对应的buffer节点列表</returns>
        private (LinkedList<List<Node>>, List<Node>) ValidateClustersByRC(LinkedList<List<Node>> clusters, bool hebin = false)
        {
            const int MaxRecursionDepth = 10;
            double rc = NetUnitR * NetUnitC;
            var validClusters = new LinkedList<List<Node>>();
            var correspondingBuffers = new List<Node>();

            // 锁对象用于同步对共享集合的访问
            var validClustersLock = new object();
            var buffersLock = new object();

            // 使用缓存字典，避免重复计算中心点
            var centerPointCache = new Dictionary<List<Node>, Node>();

            var stack = new Stack<(List<Node> cluster, int depth)>();

            foreach (var cluster in clusters)
            {
                stack.Push((cluster, 0));
            }

            while (stack.Count > 0)
            {
                var (cluster, depth) = stack.Pop();

                if (cluster == null || cluster.Count == 0)
                {
                    continue;
                }

                // 计算或获取缓存的中心点
                Node buffer;
                lock (centerPointCache)
                {
                    if (!centerPointCache.TryGetValue(cluster, out buffer))
                    {
                        buffer = GetCentetPointPosition(cluster);
                        centerPointCache[cluster] = buffer;
                    }
                }
                var bufferDelay = CalculateBufferLoad(cluster, buffer);
                buffer.Delay = bufferDelay;
                var bufferLoad = 0.5 * bufferDelay * rc;

                if (bufferLoad <= maxNetRC || hebin)
                {
                    // 加锁添加到 validClusters 和 correspondingBuffers
                    lock (validClustersLock) validClusters.AddLast(cluster);
                    lock (buffersLock) correspondingBuffers.Add(buffer);
                }
                else if (depth < MaxRecursionDepth)
                {
                    // 分裂聚类并递归验证
                    var splitClusters = SplitCluster(cluster);
                    foreach (var splitCluster in splitClusters)
                    {
                        stack.Push((splitCluster, depth + 1));
                    }
                }
                else
                {
                    // 达到最大递归深度，直接添加原始聚类和对应buffer
                    lock (validClustersLock) validClusters.AddLast(cluster);
                    lock (buffersLock) correspondingBuffers.Add(buffer);
                }
            }

            return (validClusters, correspondingBuffers);
        }

        /// <summary>
        /// 为每个有效聚类生成BufferInstance并返回对应的buffer节点列表
        /// </summary>
        /// <param name="validClusters">已符合RC负载要求的聚类列表</param>
        /// <param name="buffers">与每个聚类对应的buffer节点列表</param>
        /// <param name="bufferInstances">要更新的缓冲实例列表</param>
        /// <returns>每个聚类对应的buffer节点列表</returns>
        private List<Node> GenerateBufferInstances(LinkedList<List<Node>> validClusters, List<Node> buffers, List<BufferInstance> bufferInstances)
        {
            int clusterNumber = bufferInstances.Count + 1;

            for (int i = 0; i < validClusters.Count; i++)
            {
                var cluster = validClusters.ElementAt(i);
                var buffer = buffers[i];

                // 更新 buffer 的 ID 和 Name，确保它们与 bufferInstance 的名称一致
                buffer.Id = clusterNumber;
                buffer.Name = $"BUF{clusterNumber}";

                // 计算缓冲器的位置
                int bufferPositionWidth = buffer.X - (BufferSize_Width / 2);
                int bufferPositionHeight = buffer.Y - (BufferSize_Height / 2);
                buffer.X = bufferPositionWidth;
                buffer.Y = bufferPositionHeight;


                // 创建新的 BufferInstance
                var bufferInstance = new BufferInstance
                {
                    Name = buffer.Name,  // 与 buffer 的名称保持一致
                    Position = (bufferPositionWidth, bufferPositionHeight),
                    ContainedNodeNames = cluster.Select(node => node.Name).ToList(),
                };

                bufferInstances.Add(bufferInstance);
                clusterNumber++; // 更新 clusterNumber
            }

            return buffers;
        }




        public Node GetCentetPointPosition(List<Node> cluster)
        {

            var clustering = new CenterPointNamespace.Clustering();
            double rc = NetUnitR * NetUnitC;

            Node CenterPointPosition;
            if (isBottomLayer)
            {
                CenterPointPosition = clustering.CalculateBottomLevelCenterPoint(cluster, BufferSize_Width, BufferSize_Height, width, length);
            }
            else
            {
                if (cluster.Count < 2)
                {
                    // 如果聚类团节点数少于两个，返回该节点旁边一点距离的点
                    var singleNode = cluster.First();
                    CenterPointPosition = new Node(singleNode.X + 1, singleNode.Y + 1, 0, "BUF", BufferSize_Width, BufferSize_Height);

                }
                else
                {
                    CenterPointPosition = clustering.CalculateIntermediateLevelCenterPoint(cluster, rc, BufferSize_Width, BufferSize_Height);
                }
            }

            // 检查缓冲器位置是否与已有元件重叠
            if (IsOverlapping(CenterPointPosition))
            {
                // 如果重叠，尝试在附近找到一个不重叠的位置
                CenterPointPosition = FindNonOverlappingPosition(CenterPointPosition);
            }

            // 检查缓冲器位置是否超出电路面积
            if (CenterPointPosition.X < 0 || CenterPointPosition.Y < 0)
            {
                int halfBufferWidth = BufferSize_Width / 2;
                int halfBufferHeight = BufferSize_Height / 2;

                // 确保缓冲器位置在电路面积范围内
                int adjustedX = Math.Max(halfBufferWidth, CenterPointPosition.X);
                int adjustedY = Math.Max(halfBufferHeight, CenterPointPosition.Y);
                CenterPointPosition.X = adjustedX;
                CenterPointPosition.Y = adjustedY;
            }

            return CenterPointPosition;
        }



        private Node FindNonOverlappingPosition(Node centerPoint)
        {
            int step = 10; // 步长，可以根据需要调整
            for (int dx = -step; dx <= step; dx += step)
            {
                for (int dy = -step; dy <= step; dy += step)
                {
                    var newCenterPoint = new Node(centerPoint.X + dx, centerPoint.Y + dy, centerPoint.Id, centerPoint.Name, centerPoint.Width, centerPoint.Height);

                    if (!IsOverlapping(newCenterPoint))
                    {
                        return newCenterPoint;
                    }
                }
            }

            // 如果找不到不重叠的位置，返回原位置
            return centerPoint;
        }
        private bool IsOverlapping(Node centerPoint)
        {
            foreach (var component in CircuitComponents)
            {
                if (IsOverlapping(centerPoint, component))
                {
                    return true;
                }
            }
            return false;
        }

        private bool IsOverlapping(Node centerPoint, CircuitComponent component)
        {
            // 计算中心点对应的缓冲器的左下角和右上角坐标
            double bufferLeft = centerPoint.X - BufferSize_Width / 2.0;
            double bufferRight = centerPoint.X + BufferSize_Width / 2.0;
            double bufferBottom = centerPoint.Y - BufferSize_Height / 2.0;
            double bufferTop = centerPoint.Y + BufferSize_Height / 2.0;

            // 计算元件的左下角和右上角坐标
            double componentLeft = component.X;
            double componentRight = component.X + component.Width;
            double componentBottom = component.Y;
            double componentTop = component.Y + component.Height;

            // 判断是否重叠
            return !(bufferRight <= componentLeft || bufferLeft >= componentRight ||
                     bufferTop <= componentBottom || bufferBottom >= componentTop);
        }

        private double CalculateBufferLoad(List<Node> cluster, Node buffer)
        {
            double load = 0.0;

            foreach (var node in cluster)
            {
                double distance = CalculateManhattanDistance(buffer, node);
                load += Math.Pow(distance, 2);
            }

            return load;
        }
        private double CalculateManhattanDistance(Node node1, Node node2)
        {
            double centerX1 = node1.X + node1.Width / 2.0;
            double centerY1 = node1.Y + node1.Height / 2.0;
            double centerX2 = node2.X + node2.Width / 2.0;
            double centerY2 = node2.Y + node2.Height / 2.0;

            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
        }
        private double CalculateAverageManhattanDistance(List<Node> cluster, Node centerPoint)
        {
            double totalDistance = 0.0;
            int nodeCount = cluster.Count;

            foreach (var node in cluster)
            {
                totalDistance += CalculateManhattanDistance(node, centerPoint);
            }

            return totalDistance / nodeCount;
        }



        /// <summary>
        /// 并查集
        /// </summary>
        public class UnionFind
        {
            private List<int> parent;
            private List<int> rank;

            public UnionFind(int size)
            {
                parent = new List<int>(size);
                rank = new List<int>(size);
                for (int i = 0; i < size; i++)
                {
                    parent.Add(i);
                    rank.Add(0);
                }
            }

            public int Find(int x)
            {
                if (x < 0 || x >= parent.Count)
                {
                    throw new IndexOutOfRangeException($"Index {x} is out of range.");
                }

                while (parent[x] != x)
                {
                    parent[x] = parent[parent[x]]; // 路径压缩
                    x = parent[x];
                }
                return x;
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

            public void EnsureCapacity(int size)
            {
                while (parent.Count < size)
                {
                    parent.Add(parent.Count);
                    rank.Add(0);
                }
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