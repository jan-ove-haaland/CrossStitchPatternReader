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
    class LineEstimator
    {
        Dictionary<Point, float> index2val = new Dictionary<Point, float>();
        double[] r;

        public void Add(Point gridIndex, float val)
        {
            index2val.Add(gridIndex, val);
        }

        public void Process()
        {
            // v =  a + bx + cy  + dx2 + exy + fy2

            int N = index2val.Count;
            double[][] fx = new double[N][];
            double[] fy = new double[N];
            int i = 0;

            foreach (var e in index2val)
            {
                double x = e.Key.X;
                double y = e.Key.Y;
                double v = e.Value;

                fy[i] = v;
                fx[i++] = new double[] { 1, x, y, x*x, x*y, y*y };
            }

            r = MathNet.Numerics.Fit.MultiDim(fx, fy, false, MathNet.Numerics.LinearRegression.DirectRegressionMethod.Svd);
        }

        public double Get(float x, float y)
        {
            double val = r[0] + r[1] * x + r[2] * y + r[3] * x*x + r[4] * x*y + r[5] * y*y;
            return val;
        }

    }

    class GridEstimator
    {
        LineEstimator lx = new LineEstimator();
        LineEstimator ly = new LineEstimator();

        public void Add(Point gridIndex, PointF pixelPos)
        {
            lx.Add(gridIndex, pixelPos.X);
            ly.Add(gridIndex, pixelPos.Y);
        }

        public void Process()
        {
            lx.Process();
            ly.Process();
        }

        public PointF Get(float Xi, float Yi)
        {
            double x = lx.Get(Xi, Yi);
            double y = ly.Get(Xi, Yi);
            return new PointF((float)x, (float)y);
        }

        public Point GetP(float Xi, float Yi)
        {
            var p = Get(Xi, Yi);
            return new Point((int)Math.Round(p.X), (int)Math.Round(p.Y));
        }
    }
    class SquareGridEstimator
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
