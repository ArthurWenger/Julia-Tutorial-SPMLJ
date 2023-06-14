# In this exercise we will try to predict the quality class of wines given some chemical characteristics

# In detail, the attributes of this dataset are:
#   1) Alcohol
#   2) Malic acid
#   3) Ash
#   4) Alcalinity of ash  
#   5) Magnesium
#   6) Total phenols
#   7) Flavanoids
#   8) Nonflavanoid phenols
#   9) Proanthocyanins
#   10) Color intensity
#   11) Hue
#   12) OD280/OD315 of diluted wines
#   13) Proline 

# Further information concerning this dataset can be found online on the [UCI Machine Learning Repository dedicated page](https://archive.ics.uci.edu/ml/datasets/wine) or in particular on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names)

# Our prediction concerns the quality class of the wine (1, 2 or 3) that is given as first column of the data.

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, Plots and BetaML.
# Also, seed the random seed with the integer `123`.

cd(@__DIR__)         
using Pkg, Random          
Pkg.activate(".")   
Pkg.instantiate() 
Random.seed!(123)

# 2) Load the packages/modules DelimitedFiles, Pipe, HTTP, Plots, BetaML

using DelimitedFiles, Pipe, HTTP, Plots, BetaML

# 3) Load from internet or from local file the input data as a Matrix.
# You can use `readdlm`` using the comma as field separator.
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

data    = @pipe HTTP.get(dataURL).body |> readdlm(_, ',')

# 4) Now create the X matrix of features using the second to final columns of the data you loaded above and the Y vector by taking the 1st column. Transform the Y vector to a vector of integers using the `Int()` function (broadcasted). Make shure you have a 178×13 matrix and a 178 elements vector

X = data[:,2:end]
Y = Int.(data[:,1] )

# 5) Partition the data in (xtrain,xval) and (ytrain,yval) keeping 80% of the data for training and reserving 35% for testing. Keep the default option to shuffle the data, as the input data isn't.

((xtrain,xval), (ytrain,yval)) = partition([X,Y], [0.8,0.2])

# 6) As the output is multinomial we need to encode ytrain. We use the `oneHotEncoder()` function to make `ytrain_oh`

ytrain_oh = oneHotEncoder(ytrain) 

# 7) Define a Neural Network model with the following characteristics:
#   - 3 dense layers with respectively 13, 20 and 3 nodes and activation function relu
#   - a `VectorFunctionLayer` with 3 nodes and `softmax` as activation function
#   - `crossEntropy` as the neural network cost function

l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],crossEntropy)

# 8) Train your model using `ytrain` and a scaled version of `xtrain`` (where all columns have zero mean and 1 standard deviaiton) for 100 epochs and use a batch size of 6 records.
# Save the output of your training function to `trainingLogs`

trainingLogs = train!(mynn,scale(xtrain),ytrain_oh,batchSize=6,epochs=100)

# 9) Predict the training labels ŷtrain and the validation labels ŷval. Recall you did the training on the scaled features!

ŷtrain   = predict(mynn, scale(xtrain)) 
ŷval     = predict(mynn, scale(xval))  

# 10) Compute the train and test accuracies using the function `accuracy`

trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  

# 11) Compute and print a ConfutionMatrix of the validation data true vs. predicted

cm = ConfusionMatrix(ŷval,yval)
println(cm)

# 12) Run the following commands to plots the average loss per epoch 

plot(trainingLogs.ϵ_epochs, label="ϵ per epoch")


# 13) (Optional) Run the same workflow without scaling the data or using `squaredCost` as cost function. How this affect the quality of your predictions ? 

Random.seed!(123)
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
ytrain_oh = oneHotEncoder(ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],crossEntropy)
trainingLogs = train!(mynn,xtrain,ytrain_oh,batchSize=6,epochs=100)
ŷtrain   = predict(mynn, xtrain)
ŷval     = predict(mynn, xval) 
trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  
plot(trainingLogs.ϵ_epochs, label="ϵ per epoch (unscaled version)")

Random.seed!(123)
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
ytrain_oh = oneHotEncoder(ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],squaredCost)
trainingLogs = train!(mynn,xtrain,ytrain_oh,batchSize=6,epochs=100)
ŷtrain   = predict(mynn, xtrain)
ŷval     = predict(mynn, xval) 
trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  
plot(trainingLogs.ϵ_epochs, label="ϵ per epoch (squaredCost version)")