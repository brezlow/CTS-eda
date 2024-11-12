namespace CenterPointNamespace
{
    public class Clustering
    {
        /// <summary>
        /// 计算最底层聚类团的“中心点”位置。
        /// 使用横纵坐标的均值作为聚类团的中心点。
        /// </summary>
        /// <param name="cluster">最底层聚类团中的节点列表</param>
        /// <returns>返回计算得到的“中心点”节点</returns>
        public Node CalculateBottomLevelCenterPoint(List<Node> cluster, int BufferSize_width, int BufferSize_height)
        {
            if (cluster == null || cluster.Count == 0)
                throw new ArgumentException("聚类团不能为空");

            // 计算横纵坐标的均值，并四舍五入为整数
            int centerX = (int)Math.Round(cluster.Average(node => node.X));
            int centerY = (int)Math.Round(cluster.Average(node => node.Y));

            // 假设缓冲器的尺寸为 (0, 0) 或根据实际情况调整
            return new Node(centerX, centerY, 0, "BUF", BufferSize_width, BufferSize_height);
        }

        public Node CalculateIntermediateLevelCenterPoint(List<Node> upperCluster, double rc, int BufferSize_width, int BufferSize_height)
        {
            if (upperCluster == null || upperCluster.Count < 2)
                throw new ArgumentException("聚类团需要至少两个节点以补偿时延差异");

            // 计算初始中心点（均值法）
            int initialX = (int)Math.Round(upperCluster.Average(node => node.X));
            int initialY = (int)Math.Round(upperCluster.Average(node => node.Y));

            // 计算最大和最小延迟平方值，用于确定gamma
            double maxDelaySquared = upperCluster.Max(node => node.Delay);
            double minDelaySquared = upperCluster.Min(node => node.Delay);
            int gamma = (int)Math.Round(Math.Sqrt(maxDelaySquared - minDelaySquared));

            // 初始候选中心点
            Node bestCenterPoint = new Node(initialX, initialY, 0, "BUF", BufferSize_width, BufferSize_height);
            double minDifferenceSum = double.MaxValue;

            // 在搜索半径 gamma 内寻找最优中心点位置
            for (int dx = -gamma; dx <= gamma; dx++)
            {
                for (int dy = -gamma; dy <= gamma; dy++)
                {
                    Node candidateCenter = new Node(initialX + dx, initialY + dy, 0, "BUF", BufferSize_width, BufferSize_height);
                    double differenceSum = 0;

                    foreach (var node in upperCluster)
                    {
                        // 当前节点的延迟平方
                        double liSquared = 0.69 * 0.5 * rc * Math.Pow(node.Delay, 2);

                        // 候选中心点到当前节点的曼哈顿距离平方
                        double siSquared = Math.Pow(GetManhattanDistance(candidateCenter, node), 2);

                        // 累加延迟差异
                        differenceSum += Math.Abs(liSquared - siSquared);
                    }

                    // 更新最优中心点
                    if (differenceSum < minDifferenceSum)
                    {
                        minDifferenceSum = differenceSum;
                        bestCenterPoint = candidateCenter;
                    }
                }
            }

            return bestCenterPoint;
        }

        /// <summary>
        /// 计算两个节点间的曼哈顿距离
        /// </summary>
        /// <param name="nodeA">第一个节点</param>
        /// <param name="nodeB">第二个节点</param>
        /// <returns>返回两个节点之间的曼哈顿距离</returns>
        private double GetManhattanDistance(Node nodeA, Node nodeB)
        {
            // 计算中心点坐标
            int centerX1 = nodeA.X + nodeA.Width / 2;
            int centerY1 = nodeA.Y + nodeA.Height / 2;
            int centerX2 = nodeB.X + nodeB.Width / 2;
            int centerY2 = nodeB.Y + nodeB.Height / 2;

            // 计算曼哈顿距离
            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
        }
    }
}