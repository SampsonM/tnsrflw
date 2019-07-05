# **tnsrflw**

This repo is the result and the learnings found from following the Tensorflow.js tutorials/ guides


## **Preparing data for training**

This step involves converting the data to tensors using the best practices of shuffling and normalising the data

### **`Step 1` - getData()**

Retrieve your data from desired source and clean your data to map and filter only the required data for your model

### **`Step 2` - run()**

In this function we can see the  getData function is invoked and the [tfvis](https://github.com/tensorflow/tfjs-vis) library is used to render a graph to display our data as a scatterplot
  
We can then run this function on DOMContentLoaded

```javascript
 const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300
    }
  );

  document.addEventListener("DOMContentLoaded", run);
```

### **`Step 3` - createModel()**

Here we define the model architecture, this determines which functions/ algorithm will run when the model is excecuting.

With neural networks the algorithm is a set of layers of nuerons with weights that govern there output, it is the training part which the model then learns the best values for the weights.

The [sequential](https://js.tensorflow.org/api/latest/#sequential) model inputs flow straight to its output

```javascript
  const model = tf.sequential(); 
```

We can then add [tf.layers](https://js.tensorflow.org/api/latest/#layers) that are the building blocks of each model and will perform some operation to transform its input to its output

A [tf.layers.dense](https://js.tensorflow.org/api/latest/#layers.dense) multiplies its input by a matrix (called weights) then adds a number (called the bias), with the first layer we need to define the input shape.

```javascript
  // Add a single hidden layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  
  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));
```


### **`Step 5` - convertToTensor()**

Next we must convert our data into tensors, this takes a little more work than previous steps so a little more explanation is needed

#
**`5.1`** Wrap All calculations in a [tf.tidy](https://js.tensorflow.org/api/latest/#tidy) function to ensure clean up of all intermediate tensors

#
**`5.2`** Shuffle the data using the [tf.util.shuffle](https://js.tensorflow.org/api/latest/#util.shuffle) function, this shuffles your data using the [Fisher-Yates](https://medium.com/@oldwestaction/randomness-is-hard-e085decbcbb2) algorithm

#
**`5.3`** Map over data to pull only relevant data-points and input array into [tf.tensord2d](https://js.tensorflow.org/api/latest/#tensor2d) to create tensors

```javascript
  const inputs = data.map(d => d.horsepower)
  const labels = data.map(d => d.mpg);
```

One tensor is created for the `inputs` and one is created for the `labels` which are the outputs of the model

```javascript
  const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
  const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
```

The shape of the tensor will be:

```bash
[num_examples, num_features_per_example]
or
[ [1], [2], [3] ]
```

And we can see we have `inputs.length` of examples or `3`, and the number of input features, in our case `1` which is the *`horsepower`*

#
**`5.4`** Normalising the data into 0-1 range using the [min-max](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)) scaling is important because the internals of many ML models built with TF are designed to work with smaller numbers and typically generate better results

  - common ranges are `0 to 1` or `-1 to 1`
  

```javascript
  const inputMax = inputTensor.max();
  const inputMin = inputTensor.min();  

  const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
```

**`5.5`** Return the input and label min & max values and the normalised tensors in an object

### **`Step 6` - trainModel()**



#
## **Best practices**

- Always shuffle your data before handing it to the training algorithms
  - Shuffling prevents the model learning patterns from the structure of our data and not the data it self and prevents sensitivity to the structure subgroups

- You should always consider normalising your data before training
  - Normalising your data after turning it into tensors allows the use of vectorisation in TF to do the min-max scaling operations without the need to write explicit for loops