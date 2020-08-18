//
// Copyright (c) 2020, Jan Ove Haaland, all rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CrossStitchPatternReader
{
    class Cell
    {
        static Point GetDir(Dir d) { return new Point(d == Dir.Left ? -1 : d == Dir.Right ? 1 : 0, d == Dir.Up ? -1 : d == Dir.Down ? 1 : 0); }
        static Dir Rot180(Dir d) { return (Dir)(((int)d + 2) % 4); }
        public enum Dir { Left = 0, Up = 1, Right = 2, Down = 3 }
        public Cell[] neighbours = new Cell[4];

        public bool hasGridIndex = false;
        public Point gi;

        public Point[] ca;
        public VectorOfPoint cvp;

        public double sarea, area, sidelength;
        public PointF cpf;
        public Point cp;

        public Mat imMask = null, image = null, imagePadded = null;

        public Cell() { }
        public Cell(Point[] contour)
        {
            ca = contour;
            cvp = new VectorOfPoint(ca);
            sarea = CvInvoke.ContourArea(cvp, true);
            area = Math.Abs(sarea);
            sidelength = Math.Sqrt(area);

        }

        public void FindCenter()
        {
            var rr = CvInvoke.FitEllipse(cvp);
            cpf = rr.Center;
            cp = new Point((int)Math.Round(cpf.X), (int)Math.Round(cpf.Y));
        }


        public Point Diff(Point p) { return new Point(p.X - cp.X, p.Y - cp.Y); }
        public Point Diff(Cell c) { return Diff(c.cp); }

        public int Dist2(Point p)
        {
            var d = Diff(p);
            return d.X * d.X + d.Y * d.Y;
        }
        public int Dist2(Cell c) { return Dist2(c.cp); }

        public Mat ExtractImage(Mat im, int size)
        {
            int x0 = (from e in ca select e.X).Min();
            int x1 = (from e in ca select e.X).Max();
            int y0 = (from e in ca select e.Y).Min();
            int y1 = (from e in ca select e.Y).Max();
            Mat s = new Mat(im, new Rectangle(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
            float cx = cp.X - x0;
            float cy = cp.Y - y0;
            float dc = size * 0.5f;

            //float cx = s.Width * 0.5f;
            //float cy = s.Height * 0.5f;
            Mat rm = new Mat();
            double ra = -EstimateRotationFromNeighbours();
            float rx = (float)Math.Cos(ra);
            float ry = (float)Math.Sin(ra);
            PointF[] asrc = new PointF[] { new PointF(cx, cy), new PointF(cx + 1, cy), new PointF(cx, cy + 1) };
            PointF[] adst = new PointF[] { new PointF(dc, dc), new PointF(dc + rx, dc + ry), new PointF(dc - ry, dc + rx) };

            Mat atrans = CvInvoke.GetAffineTransform(asrc, adst);
            CvInvoke.WarpAffine(s, rm, atrans, new Size(size, size), Inter.Cubic);
            return rm;
        }


        public Cell this[Dir d]
        {
            get { return neighbours[(int)d]; }
            set { Debug.Assert(neighbours[(int)d] == null || neighbours[(int)d] == value); neighbours[(int)d] = value; }
        }

        public void Link(Cell c, Dir d)
        {
            this[d] = c;
            c[Rot180(d)] = this;
        }


        public void SetGridIndex(Point p)
        {
            Debug.Assert(!hasGridIndex || gi == p);
            gi = p;
            hasGridIndex = true;
        }
        private void SetGridIndex(Dir d, int x, int y)
        {
            if (this[d] != null)
                this[d].SetGridIndex(new Point(x, y));
        }

        public void CalcNeighboursGridIndex()
        {
            int x = gi.X;
            int y = gi.Y;
            SetGridIndex(Dir.Left, x - 1, y);
            SetGridIndex(Dir.Up, x, y - 1);
            SetGridIndex(Dir.Right, x + 1, y);
            SetGridIndex(Dir.Down, x, y + 1);
        }


        Point? EsitmateNeighbourCenter(Dir dir)
        {
            var a = this[dir];
            if (a == null) return null;
            var b = a[dir];
            if (b == null) return null;
            return new Point(2 * a.cp.X - b.cp.X, 2 * a.cp.Y - b.cp.Y);
        }

        public bool EstimateFromNeighbours()
        {
            List<Cell> ns = new List<Cell>();
            List<Point> estimates = new List<Point>();
            foreach (Dir dir in Enum.GetValues(typeof(Dir)))
            {
                if (this[dir] != null)
                    ns.Add(this[dir]);
                var p = EsitmateNeighbourCenter(dir);
                if (p.HasValue)
                    estimates.Add(p.Value);
            }
            if (estimates.Count < 3) return false;

            int cx = (int)Math.Round((from e in estimates select e.X).Average());
            int cy = (int)Math.Round((from e in estimates select e.Y).Average());
            cp = new Point(cx, cy);
            sidelength = (from e in ns select e.sidelength).Average();
            return true;
        }

        public double EstimateRotationFromNeighbours()
        {
            double dx = 0, dy = 0;
            foreach (Dir dir in Enum.GetValues(typeof(Dir)))
            {
                var c = this[dir];
                if (c == null) continue;
                var d = Diff(c);
                var r = Math.Atan2(d.Y, d.X);
                r -= Math.PI / 2 * (2 + (int)dir);
                dx += Math.Cos(r);
                dy += Math.Sin(r);
            }
            return Math.Atan2(dy, dx);
        }

        public float Match(Cell c)
        {
            if (imagePadded == null)
            {
                int padPix = 1;
                imagePadded = new Mat(image.Rows + 2 * padPix, image.Cols + 2 * padPix, DepthType.Cv8U, 1);
                Mat w = new Mat(imagePadded, new Rectangle(new Point(padPix, padPix), image.Size));
                image.CopyTo(w);
            }

            Mat tempRes = new Mat();
            //CvInvoke.MatchTemplate(imagePadded, c.image, tempRes, TemplateMatchingType.SqdiffNormed);
            CvInvoke.MatchTemplate(imagePadded, c.image, tempRes, TemplateMatchingType.CcoeffNormed);
            double min = 0, max = 0;
            Point minLoc = new Point(), maxLoc = new Point();
            CvInvoke.MinMaxLoc(tempRes, ref min, ref max, ref minLoc, ref maxLoc);
            //var r = min;
            var r = 1 - max;
            return (float)r;
        }
    }
}
