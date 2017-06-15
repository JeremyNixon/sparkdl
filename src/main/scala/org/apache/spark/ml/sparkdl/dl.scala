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
  var layer_type: String = null
  var delta: BDM[Double] = null
  def prev_delta(delta: BDM[Double]): BDM[Double]
  def forward(forward_data: BDM[Double]): BDM[Double]
  def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double]
  var weights: BDM[Double] = null
  var moment1: BDM[Double] = null
  var moment2: BDM[Double] = null

  /**
   * Initializes weights for convolutional layer.
   * Weights are random between -.005 and .005.
   */
  def init_conv_parameters(num_filters: Int, prev_channels: Int, filter_height: Int,
     filter_width: Int, kind: String): BDM[Double] = {
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
  var layers: List[Layer] = new ArrayList[Layer]

  def add(new_layer: Layer): Unit = {
    layers.add(new_layer)
  }

  def evaluate(x: BDM[Double], y: BDV[Double]): Double = {
    var f = x
    // Forward
    for (layer <- 0 to this.layers.size-1) {
      f = this.layers.get(layer).forward(f)
    }
    var softmax = f
    val predictions = argmax(softmax, Axis._1)
    val diff = predictions-convert(y, Int)
    var count = 0
    for (i <- 0 to diff.size-1) {
      if (diff(i) == 0) {count += 1}
      else {}
    }
    count.toDouble/diff.size.toDouble
  }

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


  def fit(dataset: Dataset[_], lr: Double = .001,
           num_iters: Int = 1000, optimizer: String = "adam", s: Double = .9,
           batch_size: Int = 16, r: Double = .999): Sequential = {

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
        println(evaluate(x(0 until 100, ::), y(0 until 100)))
      }
    }
//    println(evaluate(x(0 until 1000, ::), y(0 until 1000)))
    this
  }


}

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



class Convolution2D(num_filters: Int, prev_channels: Int, filter_height: Int,
   filter_width: Int, img_height: Int, img_width: Int) extends Layer {
  this.layer_type = "Convolution2D"
  this.weights = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "weights")
  this.moment1 = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "moment")
  this.moment2 = init_conv_parameters(num_filters,
    prev_channels, filter_height, filter_width, "moment")
  this.delta = null
  val len = img_height * img_width

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

class MaxPooling2D(pool_height: Int, pool_width: Int, pool_stride_x: Int, pool_stride_y: Int,
           prev_channels: Int, num_filters: Int, img_height: Int, img_width: Int) extends Layer {
  this.layer_type = "MaxPooling2D"
  val len = img_height * img_width

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

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - POOLING LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }
}

class Activation(var kind: String) extends Layer {
  this.layer_type = "Activation"
  var hidden_layer: BDM[Double] = null
  this.delta = null
  var output_softmax: BDM[Double] = null

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

  override def compute_gradient(backward_data: BDM[Double], delta: BDM[Double]): BDM[Double] = {
    println("MAJOR ERROR - ACTIVATION LAYER SHOULD NOT COMPUTE GRADIENT")
    backward_data
  }
}

