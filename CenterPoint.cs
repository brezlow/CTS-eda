﻿namespace CenterPointNamespace
{
    public class Clustering
    {
        /// <summary>
        /// 计算最底层聚类团的“中心点”位置。
        /// 使用横纵坐标的均值作为聚类团的中心点。
        /// </summary>
        /// <param name="cluster">最底层聚类团中的节点链表</param>
        /// <returns>返回计算得到的“中心点”节点</returns>
        public Node CalculateBottomLevelCenterPoint(LinkedList<Node> cluster, int BufferSize_width, int BufferSize_height)
        {
            if (cluster == null || cluster.Count == 0)
                throw new ArgumentException("聚类团不能为空");

            // 计算横纵坐标的均值
            double sumX = 0;
            double sumY = 0;
            int count = 0;

            foreach (var node in cluster)
            {
                sumX += node.X;
                sumY += node.Y;
                count++;
            }

            int centerX = (int)Math.Round(sumX / count);
            int centerY = (int)Math.Round(sumY / count);

            // 假设缓冲器的尺寸为 (0, 0) 或根据实际情况调整
            return new Node(centerX, centerY, 0, 0, BufferSize_width, BufferSize_height);
        }

        public Node CalculateIntermediateLevelCenterPoint(LinkedList<Node> upperCluster, int gamma, int BufferSize_width, int BufferSize_height)
        {
            if (upperCluster == null || upperCluster.Count < 2)
                throw new ArgumentException("聚类团需要至少两个节点以补偿时延差异");

            // 初步中心点（均值计算）
            double sumX = 0;
            double sumY = 0;
            int count = 0;

            foreach (var node in upperCluster)
            {
                sumX += node.X;
                sumY += node.Y;
                count++;
            }

            int initialX = (int)Math.Round(sumX / count);
            int initialY = (int)Math.Round(sumY / count);

            // 设置一个初始的中心点
            Node bestCenterPoint = new Node(initialX, initialY, 0, 0, BufferSize_width, BufferSize_height);
            double minDifferenceSum = double.MaxValue;

            // 在搜索半径 gamma 内寻找最优中心点位置
            for (int dx = -gamma; dx <= gamma; dx++)
            {
                for (int dy = -gamma; dy <= gamma; dy++)
                {
                    Node candidateCenter = new Node(initialX + dx, initialY + dy, 0, 0, BufferSize_width, BufferSize_height);
                    double differenceSum = 0;

                    foreach (var node in upperCluster)
                    {
                        // (L_i^2)̅：当前节点的延迟平方
                        double liSquared = Math.Pow(node.Delay, 2);

                        // (S_i^2)̅：候选中心点到当前节点的曼哈顿距离的平方
                        double siSquared = Math.Pow(GetManhattanDistance(candidateCenter, node), 2);

                        // 计算延迟差异的绝对值并累加
                        differenceSum += Math.Abs(liSquared - siSquared);
                    }

                    // 更新最优中心点位置
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
