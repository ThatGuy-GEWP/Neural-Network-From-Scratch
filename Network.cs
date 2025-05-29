using System.Diagnostics;

namespace TNNLib
{
    public abstract class ActivationFunction
    {
        public abstract double Function(double x);
        public abstract double Derivative(double x);
    }

    public class TanH : ActivationFunction
    {
        public override double Function(double x)
        {
            return Math.Tanh(x);
        }

        public override double Derivative(double x)
        {
            return 1.0 - Math.Pow(Math.Tanh(x), 2.0);
        }
    }

    public class LeakyReLU : ActivationFunction
    {
        public override double Function(double x)
        {
            return Math.Max(0.1 * x, x);
        }

        public override double Derivative(double x)
        {
            return x > 0.0 ? 1.0 : 0.1;
        }
    }

    public class Linear : ActivationFunction
    {
        public override double Function(double x)
        {
            return x;
        }

        public override double Derivative(double x)
        {
            return 1.0;
        }
    }

    public class Neuron
    {
        public Layer layer;

        public int layerIndex = -1;

        public Neuron(Layer layer, int index) { this.layerIndex = index; this.layer = layer; }

        public List<Connection> connectionsIn = new List<Connection>();
        public List<Connection> connectionsOut = new List<Connection>();

        static ActivationFunction activationFunction = new TanH();

        public double bias = 0.0;

        /// <summary>
        /// output of the neuron with the activation function
        /// </summary>
        public double value = 0.0;

        /// <summary>
        /// weighted sum of the neuron WITHOUT the activation function
        /// </summary>
        public double weightedSum = 0.0;

        public double biasVelocity = 0.0;

        /// <summary>
        /// the wanted bias of this neuron. (for backpropogation)
        /// </summary>
        public double costGradientB = 0.0;

        // https://en.wikipedia.org/wiki/E_(mathematical_constant)
        const double e = 2.7182810;


        // https://www.geeksforgeeks.org/activation-functions-neural-networks/#types-of-activation-functions-in-deep-learning
        public double Activation(double X)
        {
            return layer.activationFunction.Function(X);
        }

        public double ActivationDerivative(double X)
        {
            return layer.activationFunction.Derivative(X);
        }

        public void ConnectTo(Neuron neuron)
        {
            Connection con = new Connection(this, neuron, 0.0);
            connectionsOut.Add(con);
            neuron.connectionsIn.Add(con);
        }

        public void RunConnections()
        {
            if(connectionsIn.Count <= 0) // if we are an input neuron
            {
                weightedSum = value;
                //value = value;
                return;
            }

            double weightedSumLocal = 0.0;
            for (int i = 0; i < connectionsIn.Count; i++)
            {
                weightedSumLocal += connectionsIn[i].From.value * connectionsIn[i].weight;
            }

            weightedSum = weightedSumLocal + bias;
            value = Activation(weightedSum);
        }

        public void CalcDerivative(double newWanted = 0.0, bool useWanted = false, double momentum = 0.0f)
        {
            //if (connectionsIn.Count < 1) return;

            double AZ = ActivationDerivative(weightedSum);

            if (connectionsOut.Count < 1)
            {
                this.costGradientB = AZ * GetCostDerivitive(newWanted);
            }
            else
            {
                double newNodeValue = 0;
                for (int i = 0; i < connectionsOut.Count; i++)
                {
                    Connection outCon = connectionsOut[i];
                    newNodeValue += outCon.weight * outCon.To.costGradientB;
                }
                newNodeValue *= AZ;
                costGradientB = newNodeValue;
            }

            for (int i = 0; i < connectionsIn.Count; i++)
            {
                Connection con = connectionsIn[i];
                double derivativeCostWeight = con.From.value * costGradientB;
                con.costGradientW += derivativeCostWeight;
            }
        }

        public void ApplyNudges(double learnRate, double momentum)
        {
            double velocity = ((biasVelocity * momentum) - costGradientB) * learnRate;
            bias += velocity;
            biasVelocity = velocity;

            for (int i = 0; i < connectionsIn.Count; i++)
            {
                connectionsIn[i].ApplyWanted(learnRate, momentum);
            }

            costGradientB = 0;
        }

        public double GetCost(double localWantedValue)
        {
            return Math.Pow(value - localWantedValue, 2.0);
        }

        public double GetCostDerivitive(double localWantedValue)
        {
            return 2.0 * (value - localWantedValue);
        }
    }

    public class Connection
    {
        /// <summary>
        /// the weight of this connection
        /// </summary>
        public double weight;

        /// <summary>
        /// the velocity of the gradient of this connection
        /// </summary>
        public double weightVelocity;

        /// <summary>
        /// the derivative of this connection
        /// </summary>
        public double costGradientW = 0.0;

        public readonly Neuron From;
        public readonly Neuron To;

        public Connection(Neuron from, Neuron to, double weight)
        {
            this.From = from;
            this.To = to;
            this.weight = weight;
        }

        public int appliedCount = 0;
        public void ApplyWanted(double learnRate, double momentum)
        {
            double velocity = ((weightVelocity * momentum) - costGradientW) * learnRate;
            weightVelocity = velocity;

            this.weight += velocity;
            costGradientW = 0;
        }
    }

    public class Layer
    {
        public Network network;
        public Neuron[] neurons;

        public ActivationFunction activationFunction;

        public int index;

        public void SetValues(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                neurons[i].value = values[i];
                neurons[i].weightedSum = values[i];
            }
        }

        public void FillConnections(double withWeight)
        {
            for(int i = 0; i < neurons.Length; i++)
            {
                for(int j = 0; j < neurons[i].connectionsIn.Count; j++)
                {
                    Connection connection = neurons[i].connectionsIn[j];
                    connection.weight = withWeight;
                    //Console.WriteLine(neurons[i].connectionsIn[j].weight);
                }

                for (int j = 0; j < neurons[i].connectionsOut.Count; j++)
                {
                    Connection connection = neurons[i].connectionsOut[j];
                    connection.weight = withWeight;
                    //Console.WriteLine(neurons[i].connectionsOut[j].weight);
                }
            }
        }

        public void RandomizeConnections(int seed, float maxValue = 1.0f)
        {
            Random rnd = new Random(seed);

            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].bias = ((rnd.NextSingle()) - 0.5) * 2.0;

                for (int j = 0; j < neurons[i].connectionsIn.Count; j++)
                {
                    Connection connection = neurons[i].connectionsIn[j];
                    connection.weight = (((rnd.NextSingle()) - 0.5) * 2.0) * maxValue;
                }

                for (int j = 0; j < neurons[i].connectionsOut.Count; j++)
                {
                    Connection connection = neurons[i].connectionsOut[j];
                    connection.weight = (((rnd.NextSingle()) - 0.5) * 2.0) * maxValue;
                    //Console.WriteLine(neurons[i].connectionsOut[j].weight);
                }
            }
        }

        public Layer(Network network, int count)
        {
            activationFunction = new TanH();
            this.network = network;
            neurons = new Neuron[count];

            for (int i = 0; i < count; i++)
            {
                neurons[i] = new Neuron(this, i);
            }
        }

        public Layer(Network network, int count, ActivationFunction func)
        {
            this.activationFunction = func;
            this.network = network;
            neurons = new Neuron[count];

            for (int i = 0; i < count; i++)
            {
                neurons[i] = new Neuron(this, i);
            }
        }

        public void SetIndex(int index)
        {
            this.index = index;
        }

        public void ConnectToNextLayer(Layer toLayer)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                Neuron curNeuron = neurons[i];
                for (int j = 0; j < toLayer.neurons.Length; j++)
                {
                    Neuron toNeuron = toLayer.neurons[j];
                    curNeuron.ConnectTo(toNeuron);
                }
            }
        }

        public void RunLayer()
        {
            int toProcess = neurons.Length;
            ManualResetEvent resetEvent = new ManualResetEvent(false);

            ThreadPool.SetMaxThreads(128, 128);
            ThreadPool.SetMinThreads(6, 6);

            Stopwatch threadWatch = Stopwatch.StartNew();

            for (int i = 0; i < neurons.Length; i++)
            {
                int j = i;

                //ThreadPool.QueueUserWorkItem(
                //    new WaitCallback(x =>
                //    {
                //        neurons[j].RunConnections();
                //        if(Interlocked.Decrement(ref toProcess) == 0)
                //        {
                //            resetEvent.Set();
                //        }
                //    }
                //    ));
                neurons[j].RunConnections();
            }

            //resetEvent.WaitOne();
            threadWatch.Stop();
            //Console.WriteLine("Took: "+threadWatch.Elapsed.TotalSeconds.ToString("N5")+" to feed forward");
        }

        public double GetCost(double[] wantedValues)
        {
            double cost = 0.0;

            for (int i = 0; i < wantedValues.Length; i++)
            {
                cost += neurons[i].GetCost(wantedValues[i]);
            }

            return cost / wantedValues.Length;
        }

        public void ApplyNudges(double learnRate, double momentum)
        {
            List<Task> tasks = new List<Task>();
            for(int i = 0; i < neurons.Length; i++)
            {
                int l = i + 0;
                tasks.Add(Task.Factory.StartNew(() =>
                {
                    neurons[l].ApplyNudges(learnRate, momentum);
                }));
            }

            foreach(Task task in tasks)
            {
                task.Wait();
            }
        }
    }


    public class Network
    {
        public List<Layer> layers = new List<Layer>();

        public Network()
        {

        }

        public void FillInputLayer(double[] with)
        {
            Layer input = layers[0];

            for(int i = 0; i < with.Length; i++)
            {
                input.neurons[i].value = with[i];
                input.neurons[i].weightedSum = with[i];
            }

        }

        public Neuron GetNeuron(int layer, int neuronIndex)
        {
            if(layer == -1) { layer = layers.Count - 1; }
            return layers[layer].neurons[neuronIndex];
        }

        public void FeedForward()
        {
            for(int i = 0; i < layers.Count; i++)
            {
                layers[i].RunLayer();
            }
        }

        public void BackPropogate(double[] wantedValues)
        {



            for(int i = layers.Count - 1; i >= 0; i--)
            {
                Layer curLayer = layers[i];

                if (i == layers.Count - 1)
                {
                    int toDo = curLayer.neurons.Length;

                    for (int j = 0; j < curLayer.neurons.Length; j++)
                    {
                        int l = j + 0;
                        curLayer.neurons[l].CalcDerivative(wantedValues[l], true);
                    }
                }
                else
                {
                    for (int j = 0; j < curLayer.neurons.Length; j++)
                    {
                        int l = j + 0;
                        curLayer.neurons[l].CalcDerivative();
                    }

                }
            }
        }

        Random rnd = new Random();


        int[] GetShuffledIndexes(double[][] inputs)
        {
            int toMake = inputs.GetLength(0);

            int[] toShuffle = new int[toMake];
            for(int i = 0;  i < toMake; i++)
            {
                toShuffle[i] = i;
            }

            for(int i = 0; i < toShuffle.Length; i++)
            {
                int targ = rnd.Next(toShuffle.Length);

                (toShuffle[i], toShuffle[targ]) = (toShuffle[targ], toShuffle[i]);
            }

            for (int i = 0; i < toShuffle.Length; i++)
            {
                int targ = rnd.Next(toShuffle.Length);

                (toShuffle[i], toShuffle[targ]) = (toShuffle[targ], toShuffle[i]);
            }

            return toShuffle;
        }


        int lastSubsetIndex = -1;
        int[] lastSubsetShuffled;

        public void Learn(double learnRate, double momentum, double[][] inputs, double[][] expectedOutputs, int batches = 16)
        {
            if (batches != 1 && batches > 1)
            {
                int batchSize = inputs.GetLength(0) / batches;

                if(lastSubsetIndex == -1 || lastSubsetIndex >= batches)
                {
                    lastSubsetShuffled = GetShuffledIndexes(inputs);
                    lastSubsetIndex = 0;
                }

                int indexOffset = batchSize * lastSubsetIndex;

                for(int i = 0; i < batchSize; i++)
                {
                    FillInputLayer(inputs[lastSubsetShuffled[i+indexOffset]]);
                    FeedForward();
                    BackPropogate(expectedOutputs[lastSubsetShuffled[i+indexOffset]]);
                }

                ApplyGradients(learnRate, momentum, batchSize);

                lastSubsetIndex++;
            }
            else
            {
                int[] randomIndexs = GetShuffledIndexes(inputs);

                for (int i = 0; i < inputs.GetLength(0); i++)
                {
                    FillInputLayer(inputs[i]);
                    FeedForward();
                    BackPropogate(expectedOutputs[i]);
                }

                ApplyGradients(learnRate, momentum, inputs.GetLength(0));
            }
        }

        public double GetTotalCost(double[][] inputs, double[][] expectedOutputs)
        {
            double totalCost = 0.0;

            for (int j = 0; j < inputs.Length; ++j)
            {
                layers[0].SetValues(inputs[j]);
                FeedForward();
                totalCost += GetCost(expectedOutputs[j]);
            }

            return totalCost / expectedOutputs.GetLength(0);
        }

        public void ApplyGradients(double learnRate, double momentum, int samples = 1)
        {
            for(int i = 0; i < layers.Count; i++)
            {
                layers[i].ApplyNudges(learnRate / (double)samples, momentum);
            }
        }

        /// <summary>
        /// Connects the layers of the network.
        /// </summary>
        public void Build(bool randomize = true, float maxValue = 0.1f)
        {
            for(int i = 0; i < layers.Count-1; i++)
            {
                Layer next = layers[i+1];
                layers[i].ConnectToNextLayer(next);
                layers[i].RandomizeConnections(rnd.Next(502010421)+i, maxValue);
            }
        }

        /// <summary>
        /// Adds a layer to the network
        /// </summary>
        public void AddLayer(Layer layer, int index)
        {
            layers.Insert(index, layer);
            layer.index = index;
        }
        
        /// <inheritdoc cref="AddLayer(Layer, int)"/>
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
            layer.index = layers.Count - 1;
        }

        /// <inheritdoc cref="AddLayer(Layer, int)"/>
        public void AddLayer(int neuronCount, ActivationFunction activationFunc)
        {
            AddLayer(new Layer(this, neuronCount));
        }

        public double GetCost(double[] wantedOutput)
        {
            Layer lastLayer = layers[layers.Count - 1];

            return lastLayer.GetCost(wantedOutput);
        }
    }
}
