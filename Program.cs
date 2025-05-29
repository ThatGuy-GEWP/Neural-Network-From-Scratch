using SFML.Graphics;
using SFML.System;
using System.Diagnostics;
using Color = SFML.Graphics.Color;
using Image = SFML.Graphics.Image;

namespace TNNLib
{
    internal static class Program
    {
        static void ClearCurrentLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, currentLineCursor);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }

        static Sprite spr = new Sprite();
        static Texture imgTex;

        public static double Map(this double value, double fromSource, double toSource, double fromTarget, double toTarget)
        {
            return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
        }

        static void VisualizeNetwork(Vector2f offsetPos, Network net, RenderWindow app)
        {
            app.Clear(new Color(5, 5, 10));

            CircleShape shape = new CircleShape(4f, 16);
            shape.Origin = new Vector2f(shape.Radius, shape.Radius);

            float xMult = 50;
            float yMult = shape.Radius * 2f;

            offsetPos += new Vector2f(10, 10);

            for (int i = 0; i < net.layers.Count; i++)
            {
                Layer curLayer = net.layers[i];

                for (int j = 0; j < curLayer.neurons.Length; j++)
                {
                    double weightMinValue = double.MaxValue;
                    double weightMaxValue = double.MinValue;

                    for (int c = 0; c < curLayer.neurons[j].connectionsOut.Count; c++)
                    {
                        Connection con = curLayer.neurons[j].connectionsOut[c];

                        if (con.weight < weightMinValue)
                        {
                            weightMinValue = con.weight;
                        }

                        if (con.weight > weightMaxValue)
                        {
                            weightMaxValue = con.weight;
                        }
                    }

                    if(weightMinValue == weightMaxValue)
                    {
                        weightMinValue = -1f;
                        weightMaxValue = 1f;
                    }

                    for (int c = 0; c < curLayer.neurons[j].connectionsOut.Count; c++)
                    {
                        Connection con = curLayer.neurons[j].connectionsOut[c];

                        Vector2f startPos = offsetPos + new SFML.System.Vector2f(i * xMult, j * yMult);
                        Vector2f endPos = offsetPos + new SFML.System.Vector2f((i + 1) * xMult, con.To.layerIndex * yMult);

                        double colSample = Map(curLayer.neurons[j].value, weightMinValue, weightMaxValue, -1, 1);

                        byte r = (byte)Map(colSample, -1.0, 0, 255, 0);
                        byte g = (byte)Map(colSample, 0.0, 1.0, 0, 255);
                        byte b = (byte)(Map(colSample, -0.1, 0.0, 0, 5) + Map(colSample, 0.0, 0.1, 5, 0));

                        Color col = new Color(r, g, b);

                        app.Draw([new Vertex(startPos, col), new Vertex(endPos, col)], PrimitiveType.Lines);
                    }

                }

                double minValue = double.MaxValue;
                double maxValue = double.MinValue;

                for (int j = 0; j < curLayer.neurons.Length; j++)
                {
                    if (curLayer.neurons[j].value < minValue)
                    {
                        minValue = curLayer.neurons[j].value;
                    }
                    if (curLayer.neurons[j].value > maxValue)
                    {
                        maxValue = curLayer.neurons[j].value;
                    }
                }

                if(minValue == maxValue) { minValue = -1f; maxValue = 1f; }

                for (int j = 0; j < curLayer.neurons.Length; j++)
                {
                    shape.Position = offsetPos + new SFML.System.Vector2f(i * xMult, j * yMult);

                    double colSample = Map(curLayer.neurons[j].value, minValue, maxValue, -1, 1);

                    byte r = (byte)Map(colSample, -1.0, 0, 255, 0);
                    byte g = (byte)Map(colSample, 0.0, 1.0, 0, 255);

                    shape.FillColor = new Color(r, g, 0);

                    app.Draw(shape);
                }
            }

            //app.Display();
        }

        static Texture Visualize(Network net, RenderWindow app)
        {
            Image asImg = new Image(imgTex.Size.X, imgTex.Size.Y);
            Image strictOutput = new Image(imgTex.Size.X, imgTex.Size.Y);

            for (int x = 0; x < imgTex.Size.X; x++)
            {
                for (int y = 0; y < imgTex.Size.Y; y++)
                {
                    net.FillInputLayer(GetInputs(x, y, (int)imgTex.Size.X, (int)imgTex.Size.Y));
                    net.FeedForward();

                    double val = net.GetNeuron(-1, 0).value;


                    Color finCol = new Color(
                        (byte)Math.Clamp(Map(net.GetNeuron(-1, 0).value, -1f, 1f, 0, 255), 0, 255),
                        (byte)Math.Clamp(Map(net.GetNeuron(-1, 1).value, -1f, 1f, 0, 255), 0, 255),
                        (byte)Math.Clamp(Map(net.GetNeuron(-1, 2).value, -1f, 1f, 0, 255), 0, 255)
                        );


                    asImg.SetPixel((uint)x, (uint)y, finCol);
                    strictOutput.SetPixel((uint)x, (uint)y, finCol);
                }
            }

            

            spr.Position = new SFML.System.Vector2f(512, 0);
            spr.Texture = imgTex;
            spr.TextureRect = new IntRect(0, 0, (int)imgTex.Size.X, (int)imgTex.Size.Y);
            spr.Scale = new SFML.System.Vector2f(512 / (int)imgTex.Size.X, 512 / (int)imgTex.Size.Y);
            app.Draw(spr);

            spr.Position = new SFML.System.Vector2f(512+256, 512);
            spr.Texture = new Texture(strictOutput);
            spr.TextureRect = new IntRect(0, 0, (int)imgTex.Size.X, (int)imgTex.Size.Y);
            spr.Scale = new SFML.System.Vector2f(256 / (int)imgTex.Size.X, 256 / (int)imgTex.Size.Y);
            app.Draw(spr);

            spr.Position = new SFML.System.Vector2f(0, 0);
            spr.Texture = new Texture(asImg, false);
            spr.TextureRect = new IntRect(0, 0, (int)imgTex.Size.X, (int)imgTex.Size.Y);
            spr.Scale = new SFML.System.Vector2f(512 / (int)imgTex.Size.X, 512 / (int)imgTex.Size.Y);


            app.Draw(spr);
            spr.Texture.Dispose();

            return new Texture(asImg);
        }


        const int sincosCount = 12;

        static double[] GetInputs(int x, int y, int sizeX = 32, int sizeY = 32)
        {
            double ix = (((((double)x / (double)sizeX) - 0.5) * 2.0) * 1.0) * Math.PI;
            double iy = (((((double)y / (double)sizeY) - 0.5) * 2.0) * 1.0) * Math.PI;

            List<double> inputsList = new List<double>(sincosCount*2 + 2);

            double amplitude = 1.0;

            double period = 1.0f;

            inputsList.Add(ix);
            for(int i = 0; i < sincosCount; i++)
            {
                inputsList.Add((Math.Sin(ix * (period * (i+1)))) * amplitude);
                inputsList.Add((Math.Cos(ix * (period * (i+1)))) * amplitude);
            }

            inputsList.Add(iy);
            for (int i = 0; i < sincosCount; i++)
            {
                inputsList.Add((Math.Sin(iy * (period * (i+1)))) * amplitude);
                inputsList.Add((Math.Cos(iy * (period * (i+1)))) * amplitude);
            }

            return inputsList.ToArray();

            return [
                ix,
                Math.Sin(ix),
                Math.Cos(ix),
                Math.Sin(2.0*ix),
                Math.Cos(2.0*ix),
                Math.Sin(3.0*ix),
                Math.Cos(3.0*ix),
                Math.Sin(4.0*ix),
                Math.Cos(4.0*ix),

                iy,
                Math.Sin(iy),
                Math.Cos(iy),
                Math.Sin(2.0*iy),
                Math.Cos(2.0*iy),
                Math.Sin(3.0*iy),
                Math.Cos(3.0*iy),
                Math.Sin(4.0*iy),
                Math.Cos(4.0*iy),
            ];
        }


        static void Main(string[] args)
        {
            RenderWindow App = new RenderWindow(new SFML.Window.VideoMode(512*2, 512 + (512/2)), "TNNLib");
            App.SetFramerateLimit(0);
            App.SetVerticalSyncEnabled(false);


            Network net = new Network();


            net.AddLayer(GetInputs(0,0).Length, new Linear());
            net.AddLayer(32, new LeakyReLU());
            net.AddLayer(24, new LeakyReLU());
            net.AddLayer(16, new LeakyReLU());
            net.AddLayer(3, new Linear());

            net.Build(true, 0.8f);

            //Task.Delay(2000).Wait();

            Layer lastLayer = net.layers[net.layers.Count-1];


            Image img = new Image("train_target.png");


            uint targRes = (uint)MathF.Floor(64f);


            RenderTexture rt = new RenderTexture(targRes, targRes);
            RectangleShape rs = new RectangleShape();
            rs.Texture = new Texture(img);
            rs.Size = new Vector2f(targRes, targRes);

            rt.Draw(rs);

            rt.Display();

            rs.Texture.Smooth = true;

            imgTex = rt.Texture;
            img = imgTex.CopyToImage();

            rs.Texture.Dispose();
            rs.Dispose();

            double[][] inputs = new double[img.Size.X * img.Size.Y][];
            double[][] outputs = new double[img.Size.X * img.Size.Y][];

            List<Texture> networkOutputs = new List<Texture>();


            for (int x = 0; x < img.Size.X; x++)
            {
                for(int y = 0; y < img.Size.Y; y++)
                {
                    int indx = x * (int)img.Size.Y + y;

                    //Console.WriteLine($"{x},{y}: {indx}");


                    inputs[indx] = GetInputs(x, y, (int)img.Size.X, (int)img.Size.Y);

                    // https://youtu.be/TkwXa7Cvfr8?t=1155


                    Color pixCol = img.GetPixel((uint)x, (uint)y);

                    outputs[indx] = [Map(pixCol.R, 0, 255, -1f, 1f), Map(pixCol.G, 0, 255, -1f, 1f), Map(pixCol.B, 0, 255, -1f, 1f)];
                }
            }

            img.Dispose();



            double[] startScore = new double[lastLayer.neurons.Length];
            for(int i = 0; i < lastLayer.neurons.Length; i++)
            {
                startScore[i] = lastLayer.neurons[i].value;
            }

            net.FillInputLayer(inputs[0]);
            net.FeedForward();
            Console.WriteLine($"Start Cost:{net.GetCost(outputs[0]):N4}");


            double learn_rate = 0.005;
            double momentum = 0.90;
            int batches = 1024;
            int whenToCheck = 1000;

            Console.WriteLine($"{inputs.GetLength(0) / batches} Tests per batch");

            double lastScore = -1;

            int itterCount = 0;
            while(itterCount < 100_000)
            {
                Stopwatch learnWatch = Stopwatch.StartNew();
                net.Learn(learn_rate, momentum, inputs, outputs, batches);
                learnWatch.Stop();

                if (itterCount % whenToCheck == 0) 
                {
                    Stopwatch costAndRenderWatch = Stopwatch.StartNew();
                    double totalCost = net.GetTotalCost(inputs, outputs);
                    if(lastScore == -1)
                    {
                        lastScore = totalCost;
                    }

                    double Accuracy = Math.Round(100.0 - (totalCost * 100), 4);

                    Console.WriteLine($"Epoch {itterCount}, Accuracy:{Accuracy}% Cost:{totalCost}");
                    net.FeedForward();

                    App.DispatchEvents();
                    App.Clear();

                    VisualizeNetwork(new Vector2f(0, 512), net, App);
                    networkOutputs.Add(Visualize(net, App));
                    App.Display();
                    costAndRenderWatch.Stop();
                    Console.WriteLine("Cost+Render time:" + costAndRenderWatch.Elapsed.TotalSeconds);
                    //if (totalCost < 0.09f) { break; }
                }

                itterCount++;
            }



            using (var gif = AnimatedGif.AnimatedGif.Create($"I_[{itterCount}]_LR[{learn_rate}]_M[{momentum}]_SC[{sincosCount}].gif", 33))
            {
                for (int i = 0; i < networkOutputs.Count; i++)
                {
                    networkOutputs[i].CopyToImage().SaveToFile("tmp.png");

                    gif.AddFrame("tmp.png");

                    File.Delete("tmp.png");
                }

            }

            
        }


        public static byte[] getBytesOfImg(Image img)
        {
            List<byte> bytes = new List<byte>();

            for (int x = 0; x < img.Size.X; x++)
            {
                for (int y = 0; y < img.Size.Y; y++)
                {
                    Color col = img.GetPixel((uint)x, (uint)y);
                    bytes.Add(col.R);
                    bytes.Add(col.G);
                    bytes.Add(col.B);
                    //bytes.Add(col.A);
                }
            }

            return bytes.ToArray();
        }
    }
}
