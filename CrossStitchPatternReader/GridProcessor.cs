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
using System.Threading;
using System.Threading.Tasks;

namespace CrossStitchPatternReader
{
    class GridProcessor
    {
        List<MCvScalar> colors = new List<MCvScalar>();
        public GridProcessor()
        {
            for (int rgb = 0; rgb < 3 * 3 * 3; rgb++)
            {
                int r = (rgb % 3) * 127;
                int g = (rgb / 3 % 3) * 127;
                int b = (rgb / 9 % 3) * 127;
                if (!(r == g && g == b))
                    colors.Add(new MCvScalar(r, g, b));
            }
        }

        public void Process(string imagePath)
        {
            var im = CvInvoke.Imread(imagePath, Emgu.CV.CvEnum.ImreadModes.Grayscale);
            Mat imx = new Mat(), th = new Mat();
            CvInvoke.MedianBlur(im, imx, 3);

            CvInvoke.Resize(imx, th, new Size(50, 50), 0, 0, Inter.Area);
            CvInvoke.GaussianBlur(th, im, new Size(7, 7), 0);
            CvInvoke.Resize(im, th, imx.Size, 0, 0, Inter.Linear);
            CvInvoke.Compare(imx, th, im, CmpType.GreaterThan);

            var imrgb = new Mat(im.Rows, im.Cols, DepthType.Cv8U, 3);
            CvInvoke.CvtColor(im, imrgb, ColorConversion.Gray2Bgr);

            var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            CvInvoke.FindContours(im, contours, null, RetrType.Ccomp, ChainApproxMethod.ChainApproxSimple);
            var ca = contours.ToArrayOfArray();

            var cells = new List<Cell>();
            List<double> ca_aa = new List<double>(ca.Length);

            for (int i = 0; i < ca.Length; i++)
            {
                var c = new Cell(ca[i]);
                cells.Add(c);
                ca_aa.Add(c.sarea);
            }
            ca_aa.Sort();

            double ca_max = 0, ca_min = 0;
            for (int i = 100; i >= 0 && ca_aa[i] / ca_aa[i + 1] < 1.01; i--)
                ca_max = -ca_aa[i];

            for (int i = 100; i < ca_aa.Count && ca_aa[i] / ca_aa[i + 1] < 1.01; i++)
                ca_min = -ca_aa[i];

            double side_avg = Math.Sqrt((ca_min + ca_max) / 2);

            List<Cell> gridCells = new List<Cell>();

            for (int i = 0; i < ca.Length; i++)
            {
                var col = colors[i % colors.Count];
                var c = cells[i];
                if (c.area > ca_max || c.area < ca_min)
                    continue;

                c.FindCenter();
                double minR = c.sidelength / 2.25;
                //CvInvoke.Circle(imrgb, c.cp, (int)minR, col, 1);
                bool ok = true;
                var minR2 = minR * minR;
                for (int j = 0; j < c.ca.Length; j++)
                {
                    if (c.Dist2(c.ca[j]) < minR2)
                        ok = false;
                }

                //for (int j = 1; j < c.ca.Length; j++)
                //    CvInvoke.Line(imrgb, c.ca[j - 1], c.ca[j], col, ok ? 9 : 3);

                if (ok)
                    gridCells.Add(c);
            }


            double maxCentDist2 = 2 * side_avg * side_avg;
            for (int i = 0; i < gridCells.Count; i++)
            {
                var a = gridCells[i];
                for (int j = 0; j < gridCells.Count; j++)
                {
                    var b = gridCells[j];
                    if (i == j) continue;
                    if (a.Dist2(b) < maxCentDist2)
                    {
                        var d = a.Diff(b);
                        bool isX = Math.Abs(d.X) > Math.Abs(d.Y);
                        Cell.Dir dir = isX ? d.X < 0 ? Cell.Dir.Left : Cell.Dir.Right : d.Y < 0 ? Cell.Dir.Up : Cell.Dir.Down;
                        a.Link(b, dir);

                        //CvInvoke.Line(imrgb, a.cp, b.cp, new MCvScalar(0, 255, isX ? 255 : 0), 3);
                    }
                }
            }

            Queue<Cell> qc = new Queue<Cell>();
            gridCells[0].SetGridIndex(new Point()); // Arbitrary Origo
            qc.Enqueue(gridCells[0]);
            while (qc.Count > 0)
            {
                var c = qc.Dequeue();
                foreach (var nc in c.neighbours)
                    if (nc != null && !nc.hasGridIndex)
                        qc.Enqueue(nc);
                c.CalcNeighboursGridIndex();
            }

            int gixr = (from c in gridCells select c.gi.X).Min();
            int giyr = (from c in gridCells select c.gi.Y).Min();
            foreach (var c in cells)
                c.gi = new Point(c.gi.X - gixr, c.gi.Y - giyr);
            gixr = (from c in gridCells select c.gi.X).Max() + 1;
            giyr = (from c in gridCells select c.gi.Y).Max() + 1;

            var gridEst = new GridEstimator();
            foreach (var c in gridCells)
                gridEst.Add(c.gi, c.cpf);

            gridEst.Process();
            for (int Xi = 0; Xi < gixr; Xi++)
            {
                Point p1 = gridEst.GetP(Xi + 0.5f, 0.5f);
                Point p2 = gridEst.GetP(Xi + 0.5f, giyr - 0.5f);
                CvInvoke.Line(imrgb, p1, p2, new MCvScalar(0, 100, 255), 5);
            }
            for (int Yi = 0; Yi < giyr; Yi++)
            {
                Point p1 = gridEst.GetP(0.5f, Yi + 0.5f);
                Point p2 = gridEst.GetP(gixr - 0.5f, Yi + 0.5f);
                CvInvoke.Line(imrgb, p1, p2, new MCvScalar(0, 100, 255), 5);
            }


            Cell[,] cg = new Cell[gixr, giyr];
            foreach (var c in gridCells)
                cg[c.gi.X, c.gi.Y] = c;

            for (int xi = 0; xi < gixr; xi++)
                for (int yi = 0; yi < giyr; yi++)
                {
                    var c = cg[xi, yi];
                    if (c == null) continue;
                    var col = colors[(xi + yi) % colors.Count];
                    double minR = c.sidelength / 2.25;
                    //CvInvoke.Circle(imrgb, c.cp, (int)minR, col, 4);
                }

            var sidesize = (int)(side_avg * 0.8);

            for (int xi = 0; xi < gixr; xi++)
                for (int yi = 0; yi < giyr; yi++)
                {
                    var c = cg[xi, yi];
                    if (c == null) continue;
                    c.image = c.ExtractImage(imx, sidesize);
                    c.imMask = c.ExtractImage(im, sidesize);
                }

            List<float> diffs = new List<float>();
            for (int xi = 0; xi < gixr; xi++)
                for (int yi = 0; yi < giyr; yi++)
                {
                    var c = cg[xi, yi];
                    if (c == null) continue;
                    var t = cg[10, 10];
                    var r = c.Match(t);

                    diffs.Add(r);
                }

            float diffMax = diffs.Max();

            Mat gridImage = new Mat(sidesize * giyr, sidesize * gixr, DepthType.Cv8U, 3);
            gridImage.SetTo(new MCvScalar(128, 128, 128));
            Mat gridMatchImage = new Mat(sidesize * giyr, sidesize * gixr, DepthType.Cv8U, 3);
            gridMatchImage.SetTo(new MCvScalar(128, 128, 128));

            var templateCell = cg[10, 10];
            for (int xi = 0; xi < gixr; xi++)
                for (int yi = 0; yi < giyr; yi++)
                {
                    var c = cg[xi, yi];
                    if (c == null) continue;
                    var r = c.Match(templateCell);

                    var imrgb2 = new Mat();
                    CvInvoke.CvtColor(c.image, imrgb2, ColorConversion.Gray2Bgr);
                    Mat w = new Mat(gridImage, new Rectangle(new Point(xi * sidesize, yi * sidesize), c.image.Size));
                    imrgb2.CopyTo(w);

                    w = new Mat(gridMatchImage, new Rectangle(new Point(xi * sidesize, yi * sidesize), c.image.Size));
                    int rr = (int)(r / diffMax * 255);
                    w.SetTo(new MCvScalar(0, 255-rr, rr));
                    imrgb2.CopyTo(w, c.imMask);
                }

            new Emgu.CV.UI.ImageViewer(gridImage, "gridImage").Show(); // Allows zoom and pan
            //CvInvoke.Imshow("gridImage", gridImage);
            CvInvoke.Imshow("gridMatchImage", gridMatchImage);

            new Emgu.CV.UI.ImageViewer(imrgb, "work grid").Show(); // Allows zoom and pan
                                                                   //            CvInvoke.Imshow("work grid", imrgb);
            CvInvoke.Imwrite("work grid.png", imrgb);

            while (CvInvoke.WaitKey(100) == -1)
                ;
        }

    }
}
