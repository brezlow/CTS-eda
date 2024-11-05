using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

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
        public Node CalculateBottomLevelCenterPoint(List<Node> cluster)
        {
            if (cluster == null || cluster.Count == 0)
                throw new ArgumentException("聚类团不能为空");

            // 计算横纵坐标的均值，并四舍五入为整数
            int centerX = (int)Math.Round(cluster.Average(node => node.X));
            int centerY = (int)Math.Round(cluster.Average(node => node.Y));

            // 返回一个新的节点作为中心点，延迟可以设为0或忽略（如果此处不需要补偿）
            return new Node(centerX, centerY, delay: 0, Id: 0);
        }



        /// <summary>
        /// 计算中高层次的聚类团“中心点”位置。
        /// 通过补偿时延差异来找到符合条件的缓冲器放置位置。
        /// </summary>
        /// <param name="upperCluster">上层聚类团中的节点列表</param>
        /// <param name="gamma">曼哈顿半径，用于中心点搜索范围</param>
        /// <returns>返回经过补偿计算后的“中心点”节点</returns>
        public Node CalculateIntermediateLevelCenterPoint(List<Node> upperCluster, int gamma, int width, int height)
        {
            if (upperCluster == null || upperCluster.Count < 2)
                throw new ArgumentException("聚类团需要至少两个节点以补偿时延差异");

            // 初步中心点（均值计算）
            int initialX = (int)Math.Round(upperCluster.Average(node => node.X));
            int initialY = (int)Math.Round(upperCluster.Average(node => node.Y));

            // 设置一个初始的中心点
            Node bestCenterPoint = new Node(initialX, initialY, delay: 0, Id: 0);
            double minDifferenceSum = double.MaxValue;

            // 在搜索半径 gamma 内寻找最优中心点位置
            for (int dx = -gamma; dx <= gamma; dx++)
            {
                for (int dy = -gamma; dy <= gamma; dy++)
                {
                    Node candidateCenter = new Node(initialX + dx, initialY + dy, delay: 0, Id: 0);
                    double differenceSum = 0;

                    foreach (var node in upperCluster)
                    {
                        // (L_i^2)̅：当前节点的延迟平方
                        double liSquared = Math.Pow(node.Delay, 2);

                        // (S_i^2)̅：候选中心点到当前节点的曼哈顿距离的平方
                        double siSquared = Math.Pow(GetManhattanDistance(candidateCenter, node, width, height), 2);

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
        private double GetManhattanDistance(Node nodeA, Node nodeB, int width, int height)
        {

            // 计算中心点坐标，向下取整确保结果为整数
            int centerX1 = nodeA.X + width / 2;
            int centerY1 = nodeA.Y + height / 2;
            int centerX2 = nodeB.X + width / 2;
            int centerY2 = nodeB.Y + height / 2;

            // 计算曼哈顿距离
            return Math.Abs(centerX1 - centerX2) + Math.Abs(centerY1 - centerY2);
        }


    }
}
