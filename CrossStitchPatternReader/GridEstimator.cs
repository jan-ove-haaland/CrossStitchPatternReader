//
// Copyright (c) 2020, Jan Ove Haaland, all rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CrossStitchPatternReader
{
    class GridEstimator
    {
        Dictionary<Point, PointF> index2pixel = new Dictionary<Point, PointF>();
        double[] r;

        public void Add(Point gridIndex, PointF pixelPos)
        {
            index2pixel.Add(gridIndex, pixelPos);
        }

        public void Process()
        {
            // x =  a Xi + b Yi  + c
            // y = -b Xi + a Yi  + d

            int N = index2pixel.Count;
            int N2 = 2 * N;
            double[][] fx = new double[N2][];
            double[] fy = new double[N2];
            int i = 0;

            foreach (var e in index2pixel)
            {
                double Xi = e.Key.X;
                double Yi = e.Key.Y;
                double x = e.Value.X;
                double y = e.Value.Y;

                fy[i + 0] = x;
                fy[i + 1] = y;

                fx[i + 0] = new double[4] { Xi,  Yi, 1, 0 };
                fx[i + 1] = new double[4] { Yi, -Xi, 0, 1 };
                i += 2;
            }

            r = MathNet.Numerics.Fit.MultiDim(fx, fy, false, MathNet.Numerics.LinearRegression.DirectRegressionMethod.Svd);
        }

        public PointF Get(float Xi, float Yi)
        {
            double a = r[0];
            double b = r[1];
            double c = r[2];
            double d = r[3];

            double x = a * Xi + b * Yi + c;
            double y = -b * Xi + a * Yi + d;
            return new PointF((float)x, (float)y);
        }

        public Point GetP(float Xi, float Yi)
        {
            var p = Get(Xi, Yi);
            return new Point((int)Math.Round(p.X), (int)Math.Round(p.Y));
        }
    }
}
