# Deep Learning for Spark MLlib

Distributed deep learning on Spark. Keras-like API and integration. 

_Spark Package [homepage](https://spark-packages.org/package/JeremyNixon/sparkdl)._

## Convolutional Neural Network

```
import org.apache.spark.ml.dl._
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)
val data = sqlContext.read.format("libsvm").load("path_to_dataset.txt")
val dataset = data.withColumnRenamed("label", "labels")


val model = new Sequential()
model.add(new Convolution2D(8, 1, 3, 3, 28, 28))
model.add(new Activation("relu"))
model.add(new Dropout(0.5))
model.add(new Dense(6272, 10))
model.add(new Activation("softmax"))

model.compile(loss="categorical_crossentropy",
 optimizer=new Optimizer().adam(lr=.001),
 metrics="Accuracy")

val trained = model.fit(dataset, num_iters=500)
```

## Feedforward Neural Network
```

import org.apache.spark.ml.dl._
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)
val data = sqlContext.read.format("libsvm").load("path_to_dataset.txt")
val dataset = data.withColumnRenamed("label", "labels")


val model = new Sequential()
model.add(new Dense(784, 100))
model.add(new Activation("relu"))
model.add(new Dense(100, 10))
model.add(new Activation("softmax"))

model.compile(loss="categorical_crossentropy",
 optimizer=new Optimizer().adam(lr=.001),
 metrics="Accuracy")

val trained = model.fit(dataset, num_iters=500)

```


## Contribution Guide
 To contribute to the project, you'll need to build and modify your cloned fork of the project. 
 
 Once you've made your changes, run:
 ```
 sbt assembly
 ```
 to build. Then you'll need to publish to your local maven or ivy with a different version number. Modify the version number in pom.xml and in build.sbt, for example from 1.0.0 to 1.0.1. Then call:
 ```
 sbt publish-local
 # sbt publishM2 to publish to maven will also work, resulting in a different name.
 ```
 To publish to ivy. 
 Once your local copy has been published, you can call it from spark packages. This call will be different from the original call - your scala version will be appended to the name and the root of the call will now be default. The version number will also change to the version you've provided. For example:
 ```
 ./spark-shell --packages default:sparkdl_2.11:0.0.1
 ```
 Where you can run your modified code.




