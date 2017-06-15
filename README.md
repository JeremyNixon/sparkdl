# Deep Learning for Spark MLlib

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