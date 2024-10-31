using System;
using System.Collections.Generic;
using System.Linq;

namespace CenterPointNamespace
{
    public class Node
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Delay { get; set; } // 平均延迟，用于中层聚类的补偿计算
    }

    public class Clustering
    {
        /// <summary>
        /// 计算最底层聚类团的“中心点”位置。
        /// 使用横纵坐标的均值作为聚类团的中心点。
        /// </summary>
        /// <param name="cluster">最底层聚类团中的节点列表</param>
        /// <returns>返回计算得到的“中心点”节点</returns>
        public Node CalculateBottomLevelCenterPoint(List<Node> cluster)
        {
            if (cluster == null || cluster.Count == 0)
                throw new ArgumentException("聚类团不能为空");

            double xSum = cluster.Sum(node => node.X);
            double ySum = cluster.Sum(node => node.Y);

            return new Node
            {
                X = xSum / cluster.Count,
                Y = ySum / cluster.Count
            };
        }

        /// <summary>
        /// 计算中高层次的聚类团“中心点”位置。
        /// 通过补偿时延差异来找到符合条件的缓冲器放置位置。
        /// </summary>
        /// <param name="upperCluster">上层聚类团中的节点列表</param>
        /// <param name="gamma">曼哈顿半径，用于中心点搜索范围</param>
        /// <returns>返回经过补偿计算后的“中心点”节点</returns>
        public Node CalculateIntermediateLevelCenterPoint(List<Node> upperCluster, double gamma)
        {
            if (upperCluster == null || upperCluster.Count < 2)
                throw new ArgumentException("聚类团需要至少两个节点以补偿时延差异");

            // 首先基于均值求一个“初步中心点”
            double initialX = upperCluster.Average(node => node.X);
            double initialY = upperCluster.Average(node => node.Y);

            Node centerPoint = new Node { X = initialX, Y = initialY };

            // 搜索半径 gamma 内找到满足条件的补偿位置
            double minDifference = double.MaxValue;
            Node bestCenterPoint = centerPoint;

            for (double dx = -gamma; dx <= gamma; dx += 1)
            {
                for (double dy = -gamma; dy <= gamma; dy += 1)
                {
                    Node candidateCenter = new Node { X = initialX + dx, Y = initialY + dy };
                    double differenceSum = 0;

                    foreach (var node in upperCluster)
                    {
                        double liSquared = Math.Pow(node.Delay, 2);
                        double siSquared = Math.Pow(GetManhattanDistance(candidateCenter, node), 2);
                        differenceSum += Math.Abs(liSquared - siSquared);
                    }

                    if (differenceSum < minDifference)
                    {
                        minDifference = differenceSum;
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
            return Math.Abs(nodeA.X - nodeB.X) + Math.Abs(nodeA.Y - nodeB.Y);
        }
    }
}
