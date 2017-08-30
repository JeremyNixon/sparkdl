/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.dl

import breeze.linalg.{*, argmax, convert, fliplr, flipud, kron, max, Axis}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import java.io.File
import java.util.{ArrayList, List}

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.ml.ann.{FeedForwardTopology, FeedForwardTrainer}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasStepSize, HasTol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{Dataset, Row}


/**
 * Abstraction inherited by all layers. Allows each layer to share a forward
 * function for the forward pass, and a previous delta and compute gradient
 * function for the backwards pass.
 */
abstract class Layer {
  // Initialize variables that are shared by all layers. Note that not every variable will
  // be used by ever layer type (for example, the activation layers do not have weights).
  /**
   * Layer type, for example "Conv", "Dense", "Activation", and others.
   */
  var layer_type: String = null

  /**
   * The delta is the error signal passed back through the network during backpropagation.
   */
  var delta: BDM[Double] = null

  /**
   * The weights of each layer contain the parameters that transform the input feature map during
   * the forward pass.
   */
  var weights: BDM[Double] = null

  /**
   * The first moment captures a moving average of the gradient, which is applied with a number of
   * the optimizers (Momentum, Adam, and others.
   */
  var moment1: BDM[Double] = null

  /**
   * The second moment captures a moving average of the squared gradient, or the magnitude of the
   * gradient. It is applied with the Adam optimizer.
   */
  var moment2: BDM[Double] = null


  // Initialize functions that are shared by all layers. Note that not every layer type will
  // use every function (for example, the activation layers do not call compute_gradient).
  /**
   * Compute the output of a layer given the input data or the input from the previous layer in
   * the forward pass.
   *
   * @param forward_data
   * @return
   */
  def forward(forward_data: BDM[Double]): BDM[Double]

  /**
   * Generate the error for the next layer back given the error from the previous layer
   * during backpropagation.
   *
   * @param delta
   * @return
   */
  def prev_delta(delta: BDM[Double]): BDM[Double]

  /**
   * Given the error signal, compute the gradient for layers with parameters during backproagation.
   *
   * @param backward_data
   * @param delta
   * @return
   */
  def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double]

  /**
   * Initializes weights for convolutional layer. Weights are random between -.005 and .005.
   * Shape of the convolutional filter matrix is a vertical stack.
   *
   * @param num_filters
   * @param prev_channels
   * @param filter_height
   * @param filter_width
   * @param kind
   * @return
   */
  def init_conv_parameters(num_filters: Int, prev_channels: Int, filter_height: Int,
                           filter_width: Int, kind: String): BDM[Double] = {
    // Convolutional filter matrix is a vertical stack of filters. Order is that for each
    // previous channel, fill in the filters that map to the next impage map.
    val filters = BDM.zeros[Double](filter_height * num_filters * prev_channels, filter_width)
    val r = scala.util.Random
    for (j <- 0 until prev_channels) {
      for (i <- 0 until num_filters) {
        val y_index = filter_height * j + filter_height * i
        if (kind == "weights") {
          filters(y_index until y_index + filter_height, ::) :=
            BDM.ones[Double](filter_height, filter_width).map(x => r.nextDouble-0.5) :* .01
        } else {
          filters(y_index until y_index + filter_height, ::) :=
            BDM.zeros[Double](filter_height, filter_width)
        }
      }
    }
    filters
  }
}

/**
 * Sequential stores the network architecture created by the user.
 * Sequential also maintains the networks layers and learned weights for inference.
 */
class Sequential() {
  /**
   * List for storing the layers in the network architecture.
   */
  var layers: List[Layer] = new ArrayList[Layer]

  /**
   * Initialize optimizer
   */
  var optimizer: Optimizer = null

  /**
   * Initialize Learning Rate
   */
  var lr: Double = .01

  /**
   * Initialize momentum term, exponential weight for moving average of the gradient
   */
  var s: Double = .9

  /**
   * Initialize exponential weight for moving average of the history of gradient sizes
   */
  var r: Double = .999

  /**
   * Determies which loss function to evaluate
   */
  var loss: String = "categorical_crossentropy"

  /**
   * Determines which metric to display
   */
  var metrics: String = "accuracy"

  /**
   * Determines which optimizer to apply
   */
  var optimizer_type: String = "adam"

  /**
   * Appends layer to the end of the network topology.
   *
   * @param new_layer
   * @return
   */
  def add(new_layer: Layer): Unit = {
    layers.add(new_layer)

    // Allows Dropout layer to be specified by user without manually adding activation function
    if (new_layer.layer_type == "Dropout") {
      layers.add(new Activation("linear"))
    }
  }

  /**
   * Display network loss metrics to user
   *
   * @param train_eval
   * @param labels
   */
  def evaluate(train_eval: BDM[Double], labels: BDV[Double]): Unit = {
    var f = train_eval
    // Forward through topology
    for (layer <- 0 to this.layers.size-1) {
      f = this.layers.get(layer).forward(f)
    }
    val softmax = f

    // Column in softmax with maximum value corresponds to prediction
    val predictions = argmax(softmax, Axis._1)

    // Compute proportion of labels that are correct
    val diff = predictions-convert(labels, Int)
    var count = 0
    for (i <- 0 to diff.size-1) {
      if (diff(i) == 0) {count += 1}
      else {}
    }
    print("Train Accuracy: ")
    println(count.toDouble/diff.size.toDouble)

  }

  /**
   * Samples with replacement for stochastic gradient descent.
   *
   * @param x_train
   * @param y_train
   * @param batch_size
   * @return
   */
  def get_batch(x_train: BDM[Double], y_train: BDV[Int], batch_size: Int):
  (BDM[Double], BDV[Int]) = {
    val rand = scala.util.Random
    val x_batch = BDM.zeros[Double](batch_size, x_train.cols)
    val y_batch = BDV.zeros[Int](batch_size)

    for (i <- 0 until batch_size) {
      val batch_index = rand.nextInt(x_train.rows-batch_size)
      x_batch(i, ::) := x_train(batch_index, ::)
      y_batch(i) = y_train(batch_index)
    }
    (x_batch, y_batch)
  }

  /**
   * Passes along the information relating the loss, optimizer and metrics for evaluation.
   * Compile call is just for Keras API compatibility.
   *
   * @param loss
   * @param optimizer
   * @param metrics
   */
  def compile(loss: String, optimizer: Optimizer, metrics: String): Unit = {
    this.optimizer = optimizer
    this.loss = loss
    this.metrics = metrics
  }

  /**
   * Fits the parameters of the network.
   *
   * @param dataset
   * @param num_iters
   * @param batch_size
   * @return
   */
  def fit(dataset: Dataset[_], num_iters: Int = 1000, batch_size: Int = 16): Sequential = {

    // Grab relevant variables for optimizaiton from the optimizer object.
    val lr = this.optimizer.lr
    val s = this.optimizer.s
    val r = this.optimizer.r
    val optimizer = this.optimizer.optimizer_type

    // Convert Spark Dataframe to Breeze Dense Matrix and Dense Vector
    val xArray = dataset.select("features").rdd.map(v => v.getAs[Vector](0))
      .map(v => new BDV[Double](v.toArray)).collect()
    val x = new BDM[Double](xArray.length, xArray(0).length)
    for (i <- 0 until xArray.length) {x(i, ::) := xArray(i).t}

    val yArray = dataset.select("labels").rdd.collect()
    val y = new BDV[Double](yArray.length)
    for (i <- 0 until yArray.length) {y(i) = yArray(i)(0).asInstanceOf[Double]}

    // val ones = DenseMatrix.ones[Double](x.rows, 1)
    val x_train = x
    val y_train = convert(y, Int)

    val class_count = this.layers.get(this.layers.size-2).asInstanceOf[Dense].get_num_hidden

    def conditional(value: Int, seek: Int): Int = {if (value == seek) {-1} else {0}}

    val numerical_stability = .00000001
    val rand = scala.util.Random

    for (iterations <- 0 to num_iters) {

      val (x_batch, y_batch) = get_batch(x_train, y_train, batch_size)

      var f = x_batch
      // Forward
      for (layer <- 0 to this.layers.size-1) {
        f = this.layers.get(layer).forward(f)
      }
      var softmax = f

      // Backward

      val softmax_delta = BDM.zeros[Double](batch_size, class_count)
      for (i <- 0 to softmax_delta.cols-1) {
        softmax_delta(::, i) := softmax_delta(::, i) + convert(convert(y_batch, Int)
          .map(x => conditional(x, i)), Double)
      }

      softmax = softmax + softmax_delta
      this.layers.get(this.layers.size-1).delta = softmax :/ batch_size.toDouble

      // Compute Errors
      for (i <- this.layers.size-2 to 0 by -1) {
        this.layers.get(i).delta = this.layers.get(i).prev_delta(this.layers.get(i + 1).delta)
      }

      // Compute and Update Gradients
      for (i <- this.layers.size-2 to 0 by -1) {
        if (this.layers.get(i).layer_type == "Dense" ||
          this.layers.get(i).layer_type == "Convolution2D") {
          val gradient =
            if (i == 0) {
              if (this.layers.get(i).layer_type == "Dense") {
                this.layers.get(i).asInstanceOf[Dense].compute_gradient(x_batch,
                  this.layers.get(i + 1).delta)
              } else {
                this.layers.get(i).asInstanceOf[Convolution2D].compute_gradient(x_batch,
                  this.layers.get(i + 1).delta)
              }
            } else {
              if (this.layers.get(i).layer_type == "Dense") {
                this.layers.get(i).asInstanceOf[Dense].compute_gradient(this.layers.get(i-1)
                  .asInstanceOf[Activation].hidden_layer, this.layers.get(i + 1).delta)
              } else {
                this.layers.get(i).asInstanceOf[Convolution2D].compute_gradient(this.layers.get(i-1)
                  .asInstanceOf[Activation].hidden_layer, this.layers.get(i + 1).delta)
              }
            }

          val layer = this.layers.get(i)

          if (optimizer == "sgd") {
            layer.weights -= lr * gradient
          }
          else if (optimizer == "momentum") {
            layer.moment1 = s * layer.moment1 + lr * gradient
            layer.weights -= layer.moment1
          }
          else if (optimizer == "adam") {
            layer.moment1 = s * layer.moment1 + (1-s) * gradient
            layer.moment2 = r * layer.moment2 + (1-r) * (gradient :* gradient)
            val m1_unbiased = layer.moment1 :/ (1 - (math.pow(s, iterations + 1)))
            val m2_unbiased = layer.moment2 :/ (1- (math.pow(r, iterations + 1)))
            layer.weights -= lr * m1_unbiased :/ (sqrt(m2_unbiased) + numerical_stability)
          }
        }
      }
      if (iterations % 10 == 0) {
        evaluate(x(0 until 101, ::), y(0 until 101))
      }
    }
    //    println(evaluate(x(0 until 1000, ::), y(0 until 1000)))
    this
  }

  /**
   * Helper function for saving model to disk.
   *
   * @param f
   * @param op
   */
  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  /**
   * Model persistance function for saving model to disk.
   *
   * @param name
   */
  def save(name: String): Unit = {
    val model = this
    var layers: Array[String] = new Array[String](model.layers.size)
    var weights: List[BDM[Double]] = new ArrayList[BDM[Double]]

    for (i <- 0 until model.layers.size) {
      layers(i) = model.layers.get(i).layer_type
      if (model.layers.get(i).layer_type == "Dense" || model.layers.get(i).layer_type ==
        "Convolution2D") {
        weights.add(model.layers.get(i).weights)
      } else {}
    }

    val dir = new File(name)
    dir.mkdir()

    this.printToFile(new File(name + "/layers.txt")) {
      p => layers.foreach(p.println)
    }

    var weights_index = 0
    for(i <- 0 until model.layers.size) {
      if (model.layers.get(i).layer_type == "Dense" || model.layers.get(i).layer_type ==
        "Convolution2D") {
        breeze.linalg.csvwrite(new File(name + "/weights" + i.toString),
          weights.get(weights_index))
        weights_index += 1
      }
    }
  }


}

/**
 * Class for functions that operate over models
 */
class Model() {

  /**
   * Load model from disk into memory.
   * @param name
   * @return
   */
  def load(name: String): Sequential = {
    val seq: Sequential = new Sequential()
    val lines = scala.io.Source.fromFile(name + "/layers.txt").mkString.split("\n")

    for (i <- 0 until lines.length) {
      if (lines(i) == "Convolution2D") {
        seq.add(new Convolution2D(8, 1, 3, 3, 28, 28))
        seq.layers.get(i).weights = breeze.linalg.csvread(new File(
          name + "/weights" + i.toString))
      }
      if (lines(i) == "Dense") {
        seq.add(new Dense(6272, 10))
        seq.layers.get(i).weights = breeze.linalg.csvread(new File(name + "/weights" + i.toString))
      }
      if (lines(i) == "Activation" && i == 1) {seq.add(new Activation("relu"))}
      if (lines(i) == "Activation" && i == 3) {seq.add(new Activation("softmax"))}
    }
    seq
  }
}

/**
 * Fully Connected Neural Network Layer.
 *
 * @param input_shape
 * @param num_hidden
 */
class Dense(input_shape: Int, num_hidden: Int) extends Layer {
  val r = scala.util.Random
  this.weights = BDM.ones[Double](input_shape, num_hidden).map(x => r.nextDouble-0.5) :* .01
  this.moment1 = BDM.zeros[Double](input_shape, num_hidden)
  this.moment2 = BDM.zeros[Double](input_shape, num_hidden)
  var hidden_layer: BDM[Double] = null
  this.delta = null
  this.layer_type = "Dense"

  def get_num_hidden: Int = num_hidden
  def get_input_shape: Int = input_shape

  override def forward(forward_data: BDM[Double]): BDM[Double] = {
    // Breeze's matrix multiplication syntax allows us to simply use *
    hidden_layer = forward_data * weights
    hidden_layer
  }

  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    delta * weights.t
  }

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    backward_data.t * delta
  }
}

class Dropout(proportion: Double) extends Layer {
  val r = scala.util.Random
  this.layer_type = "Dropout"


  override def forward(forward_data: BDM[Double]): BDM[Double] = {
    // May need to do something fancy at inference time
    forward_data.map(x => if (r.nextDouble < proportion) {0} else {x})
  }

  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    delta
  }

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - ACTIVATION LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }

}

/**
 * Convolutional layer.
 *
 * @param num_filters
 * @param prev_channels
 * @param filter_height
 * @param filter_width
 * @param img_height
 * @param img_width
 */
class Convolution2D(
                     num_filters: Int, // Number of filters in convolutional layer
                     prev_channels: Int, // Number of filters in previous layer
                     filter_height: Int, // Height of filters in this convoltional layer
                     filter_width: Int, // Width of filters in this convolutional layer
                     img_height: Int, // Height of filter map input to this layer
                     img_width: Int // Width of filter map input to this layer
                   ) extends Layer {
  this.layer_type = "Convolution2D"
  this.weights = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "weights")
  this.moment1 = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "moment")
  this.moment2 = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "moment")
  this.delta = null
  val len = img_height * img_width

  /**
   * Takes an image and filter as input and returns the output feature map.
   *
   * @param image
   * @param filter
   * @return
   */
  def convolution(image: BDM[Double], filter: BDM[Double]): BDM[Double] = {
    val image_height = image.rows
    val image_width = image.cols
    val local_filter_height = filter.rows
    val local_filter_width = filter.cols
    val padded = BDM.zeros[Double](image_height + 2 * (filter_height/2),
      image_width + 2* (filter_width/2))
    for (i <- 0 until image_height) {
      for (j <- 0 until image_width) {
        padded(i + (filter_height / 2), j + (filter_height / 2)) = image(i, j)
      }
    }
    val convolved = BDM.zeros[Double](image_height -local_filter_height + 1 + 2 *
      (filter_height/2), image_width - local_filter_width + 1 + 2 * (filter_width/2))
    for (i <- 0 until convolved.rows) {
      for (j <- 0 until convolved.cols) {
        var aggregate = 0.0
        for (k <- 0 until local_filter_height) {
          for (l <- 0 until local_filter_width) {
            aggregate += padded(i + k, j + l) * filter(k, l)
          }
        }
        convolved(i, j) = aggregate
      }
    }
    convolved
  }

  /**
   * Applies a convolutional transform to each input feature map.
   *
   * @param input_data
   * @return
   */
  override def forward(input_data: BDM[Double]): BDM[Double] = {
    val outputs = BDM.zeros[Double](input_data.rows, img_height * img_width *
      num_filters * prev_channels)
    for (i <- 0 until input_data.rows) {
      for (j <- 0 until prev_channels) {
        for (k <- 0 until prev_channels) {
          for (l <- 0 until num_filters) {
            val index1 = l * len + k * len
            val index2 = (l + 1) * len + k * len
            val data_index1 = j * len
            val data_index2 = (j + 1) * len
            val filter_index = k * filter_height + l * filter_height
            val img = input_data(i, data_index1 until data_index2).t.toDenseMatrix
              .reshape(img_height, img_width)
            val fil = weights(filter_index until (filter_index + filter_height), ::)
            outputs(i, index1 until index2) := convolution(img, fil)
              .reshape(1, img_height*img_width).toDenseVector.t
          }
        }
      }
    }
    outputs
  }

  /**
   * Computes the error for the gradient for this layer and to pass back through the network.
   *
   * @param delta
   * @return
   */
  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    val output_delta = BDM.zeros[Double](delta.rows, delta.cols/num_filters)
    for (i <- 0 until delta.rows) {
      for (j <- 0 until prev_channels) {
        for (k <- 0 until num_filters) {
          val filter_index = filter_height * j + filter_height * k
          val x_index = j * len + k * len
          val img = delta(i, x_index until x_index + len).t.toDenseMatrix
            .reshape(img_height, img_width)
          val filter = flipud(fliplr(weights(filter_index until filter_index + filter_height, ::)))
          val x_output = j * len
          output_delta(i, x_output until x_output + len) +=
            convolution(img, filter).reshape(1, img_height*img_width).toDenseVector.t
        }
      }
    }
    output_delta :/ num_filters.toDouble
  }

  /**
   * Computes the gradient for this layer's convolutional filters.
   *
   * @param backward_data
   * @param delta
   * @return
   */
  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    val gradient = BDM.zeros[Double](filter_height * num_filters * prev_channels, filter_width)
    for (i <- 0 until backward_data.rows) {
      for (j <- 0 until prev_channels) {
        for (k <- 0 until num_filters) {
          val y_index = filter_height * j + filter_height * k
          val data_index = prev_channels * j
          val delta_index = k * len
          gradient(y_index until y_index + filter_height, ::) +=
            convolution(backward_data(i, data_index until (data_index + len)).t.toDenseMatrix.
              reshape(img_height, img_width), delta(i, delta_index until (delta_index + len)).
              t.toDenseMatrix.reshape(img_height, img_width))
        }
      }
    }
    gradient :/ backward_data.rows.toDouble
  }
}

/**
 * Max Pooling layer to introduce local invariance.
 *
 * @param pool_height
 * @param pool_width
 * @param pool_stride_x
 * @param pool_stride_y
 * @param prev_channels
 * @param num_filters
 * @param img_height
 * @param img_width
 */
class MaxPooling2D(pool_height: Int, pool_width: Int, pool_stride_x: Int, pool_stride_y: Int,
                   prev_channels: Int, num_filters: Int, img_height: Int, img_width: Int) extends Layer {
  this.layer_type = "MaxPooling2D"
  val len = img_height * img_width

  /**
   * Apply max pooling to each feature map.
   *
   * @param input_data
   * @return
   */
  override def forward(input_data: BDM[Double]): BDM[Double] = {
    val outputs = BDM.zeros[Double](input_data.rows, (img_height * img_width *
      prev_channels) / (pool_width * pool_height))
    for (i <- 0 until input_data.rows) {
      for (j <- 0 until prev_channels) {
        val img = input_data(i, j * len until (j + 1) * len).t.toDenseMatrix
          .reshape(img_height, img_width).t
        for (k <- 0 until img_height by pool_stride_y) {
          for (l <- 0 until img_width by pool_stride_x) {
            outputs(i, j*img_height / pool_height * img_width / pool_width +
              k / pool_stride_y * img_width / pool_stride_x + l / pool_stride_x) =
              max(img(k until k + pool_height, l until l + pool_width))
          }
        }
      }
    }
    outputs
  }

  /**
   * Computes the error through this layer to pass back through the network.
   *
   * @param delta
   * @return
   */
  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    val output_delta = BDM.zeros[Double](delta.rows, len * prev_channels)
    for (i <- 0 until delta.rows) {
      for (j <- 0 until prev_channels) {
        val x_index = j * img_height / pool_height * img_width / pool_width
        val img = delta(i, x_index until x_index + img_height / pool_height *
          img_width / pool_width).t.toDenseMatrix.reshape(img_height / pool_height,
          img_width / pool_width)
        val x_output = j * len
        output_delta(i, x_output until x_output + len) :=
          kron(img, BDM.ones[Double](pool_height, pool_width)).reshape(1,
            img_height * img_width).toDenseVector.t
      }
    }
    output_delta
  }

  /**
   * Max pooling has no weights, so this function just passes through.
   *
   * @param backward_data
   * @param delta
   * @return
   */
  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - POOLING LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }
}

/**
 * Holder for optimizers of the topology. Set up for alignment with the Keras API.
 */
class Optimizer() {
  var lr: Double = 0.01
  var s: Double = 0.9
  var r: Double = 0.999
  var optimizer_type: String = "adam"

  def adam(lr: Double = 0.01, s: Double = 0.9, r: Double = 0.999): Optimizer = {
    this.lr = lr
    this.s = s
    this.r = r
    this.optimizer_type = "adam"
    this
  }

  def momentum(lr: Double = 0.01, s: Double = 0.9): Optimizer = {
    this.lr = lr
    this.s = s
    this.optimizer_type = "momentum"
    this
  }

  def SGD(lr: Double = 0.01): Optimizer = {
    this.lr = lr
    this.optimizer_type = "sgd"
    this
  }
}

/**
 * Class for activation functions, including the topology's output layer.
 *
 * @param kind
 */
class Activation(var kind: String) extends Layer {
  this.layer_type = "Activation"
  var hidden_layer: BDM[Double] = null
  this.delta = null
  var output_softmax: BDM[Double] = null

  /**
   * Evaluate the give activation function.
   *
   * @param input_data
   * @return
   */
  override def forward(input_data: BDM[Double]) : BDM[Double] = {

    if (kind == "relu") {
      hidden_layer = input_data.map(x => max(0.0, x))
      hidden_layer
    }

    else if (kind == "linear") {
      hidden_layer = input_data
      hidden_layer
    }

    else if (kind == "softmax") {
      val softmax = exp(input_data)
      val divisor = breeze.linalg.sum(softmax(*, ::))
      for (i <- 0 to softmax.cols-1) {
        softmax(::, i) := softmax(::, i) :/ divisor
      }
      softmax
    }

    else {
      println("MAJOR ERROR1")
      input_data
    }
  }

  def relu_grad(value: Double): Double = {if (value <= 0) {0} else {1}}


  /**
   * Compute the error through the activation function to pass back through the network.
   *
   * @param delta
   * @return
   */
  override def prev_delta(delta: BDM[Double]): BDM[Double] = {
    if (kind == "relu") {
      delta :* hidden_layer.map(relu_grad)
      delta
    }
    else if (kind == "linear") {
      delta
    }
    else {
      println("MAJOR ERROR2")
      delta
    }
  }

  /**
   * Activation functions don't have weights, and so this function just passes through.
   *
   * @param backward_data
   * @param delta
   * @return
   */
  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - ACTIVATION LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }
}

