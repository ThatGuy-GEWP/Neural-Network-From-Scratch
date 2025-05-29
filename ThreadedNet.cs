using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TNNLib
{
    public struct ThreadedNeuron
    {
        public double value;

        public double bias;

        public double[] weightsIn;


        public ThreadedNeuron(double bias, double[] weightsIn)
        {
            this.bias = bias;
            this.weightsIn = weightsIn;
        }

        public ThreadedNeuron(Neuron copyFrom)
        {
            this.bias = copyFrom.bias;
            if (copyFrom.connectionsIn.Count == 0) { this.weightsIn = new double[0]; return; }

            List<double> weights = new List<double>(copyFrom.connectionsIn.Count);

            for(int i = 0; i <  copyFrom.connectionsIn.Count; i++)
            {
                weights[i] = copyFrom.connectionsIn[i].weight;
            }

            this.weightsIn = weights.ToArray();
        }

        public float getSum() { return 0f; }
    }

    public struct ThreadedWeight(double weight)
    {
        public double weight = weight;
    }


    public struct ThreadedNet
    {
        public ThreadedNeuron[][] layers;

        public ActivationFunction[] activationFunctions;

        public ThreadedNet(Network net)
        {
            List<ThreadedNeuron[]> allLayers = new List<ThreadedNeuron[]>();
            List<ActivationFunction> activationFunctions = new List<ActivationFunction>();

            foreach(Layer l in net.layers)
            {
                List<ThreadedNeuron> curLayer = new List<ThreadedNeuron>();

                activationFunctions.Add(l.activationFunction);

                for (int i = 0; i < l.neurons.Length; i++)
                {
                    curLayer.Add(new ThreadedNeuron(l.neurons[i]));
                }
            }

            layers = allLayers.ToArray();
            this.activationFunctions = activationFunctions.ToArray();
        }

        public void SetInputs(double[] toValues)
        {
            for(int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].value = toValues[i];
            }
        }
    }
}
