# EXERCISE 4.1: House value prediction with Neural Networks (regression)


```@raw html
<p>&nbsp;</p>
<img src="imgs/bostonHousingErrorPerEpoch.png" alt="Error per epoch" style="height:250px;"> 
<img src="imgs/bostonHousingEstVsTrueValues.png" alt="Estimated vs True house values" style="height:250px;"> 
<p>&nbsp;</p>
```

In this problem, we are given a dataset containing average house values in different Boston suburbs, together with the suburb characteristics (proportion of owner-occupied units built prior to 1940, index of accessibility to radial highways, etc...)
Our task is to build a neural network model and train it in order to predict the average house value on each suburb.

The detailed attributes of the dataset are:
  1. CRIM      per capita crime rate by town
  2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
  3. INDUS     proportion of non-retail business acres per town
  4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  5. NOX       nitric oxides concentration (parts per 10 million)
  6. RM        average number of rooms per dwelling
  7. AGE       proportion of owner-occupied units built prior to 1940
  8. DIS       weighted distances to five Boston employment centres
  9. RAD       index of accessibility to radial highways
  10. TAX      full-value property-tax rate per \$10,000
  11. PTRATIO  pupil-teacher ratio by town
  12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  13. LSTAT    % lower status of the population
  14. MEDV     Median value of owner-occupied homes in \$1000's


Further information concerning this dataset can be found on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)

Our prediction concern the median value (column 14 of the dataset)


**Skills employed:**
- download and import data from internet
- design and train a Neural Network for regression tasks using `BetaML`
- use the additional `BetaML` functions `partition`, `oneHotEncoder`, `scale`, `meanRelError`


## Instructions

If you have already cloned or downloaded the whole [course repository](https://github.com/sylvaticus/SPMLJ/) the folder with the exercise is on `[REPOSITORY_ROOT]/lessonsMaterial/04_NN/bostonHousing`.
Otherwise download a zip of just that folder [here](https://downgit.github.io/#/home?url=https://github.com/sylvaticus/SPMLJ/tree/main/lessonsMaterial/04_NN/bostonHousing).

In the folder you will find the file `BostonHousingValue.jl` containing the julia file that **you will have to complete to implement the missing parts and run the file** (follow the instructions on that file). 
In that folder you will also find the `Manifest.toml` file. The proposal of resolution below has been tested with the environment defined by that file.  
If you are stuck and you don't want to lookup to the resolution above you can also ask for help in the forum at the bottom of this page.
Good luck! 

## Resolution

Click "ONE POSSIBLE SOLUTION" to get access to (one possible) solution for each part of the code that you are asked to implement.

--------------------------------------------------------------------------------
### 1) Setting up the environment...
Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages `Pipe`, `HTTP`, `CSV`, `DataFrames`, `Plots` and `BetaML`.


```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate() 
using Random
Random.seed!(123)
```
```@raw html
</details>
```


--------------------------------------------------------------------------------
### 2) Load the packages 
Load the packages `Pipe`, `HTTP`, `CSV`, `DataFrames`, `Plots` and `BetaML`.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
using Pipe, HTTP, CSV, DataFrames, Plots, BetaML
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 3) Load the data
Load from internet or from local file the input data into a DataFrame or a Matrix.
You will need the CSV options `header=false` and `ignorerepeated=true`

```julia
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
```

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame
```
```@raw html
</details>
```


--------------------------------------------------------------------------------
### 4) Implement one-hot encoding of categorical variables
The 4th column is a dummy related to the information if the suburb bounds a certain Boston river. Use the BetaML function `oneHotEncoder` to encode this dummy into two separate vectors, one for each possible value. Note that you will need to transform the range {0,1} into {1,2} before running the oneHotEncoder function (this can be done by simply uinsg `data[:,4] .+1`)

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
riverDummy = oneHotEncoder(data[:,4] .+1)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 5) Put together the feature matrix
Now create the X matrix of features concatenating horizzontaly the 1st to 3rd column of `data`, the 5th to 13th columns and the two columns you created with the one hot encoding. Make sure you have a 506×14 matrix.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
X = hcat(Matrix(data[:,[1:3;5:13]]),riverDummy)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 6) Build the label vector
Similarly define Y to be the 14th column of data

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
Y = data[:,14]
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 7) Partition the data
Partition the data in (`xtrain`,`xval`) and (`ytrain`,`yval`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 8) Define the neural network architecture
Define a Neural Network model with the following characteristics:
  - 3 dense layers with respectively 14, 20 and 1 nodes and activation function `relu`
  - cost function `squaredCost` 

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
l1 = DenseLayer(14,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,1,f=relu)
mynn = buildNetwork([l1,l2,l3],squaredCost)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 9) Train the model
Train the model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviaiton) for 400 epochs and use a batch size of 6 records.
Save the output of your training function to `trainingLogs`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
trainingLogs = train!(mynn,scale(xtrain),ytrain,batchSize=6,epochs=400)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 10) Predict the labels
Predict the training labels `ŷtrain` and the validation labels `ŷval`. Recall you did the training on the scaled features!

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
ŷtrain   = predict(mynn, scale(xtrain)) 
ŷval     = predict(mynn, scale(xval))  
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 11) Evaluate the model
Compute the train and test relative mean error using the function `meanRelError` with the parameter `normRec` set to `false`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
trainRME = meanRelError(ŷtrain,ytrain,normRec=false) 
testRME  = meanRelError(ŷval,yval,normRec=false)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 12) Plot the errors and the estimated values vs the true ones
Run the following commands to plots the average loss per epoch and the true vs estimation validation values:

```julia
plot(trainingLogs.ϵ_epochs)
scatter(yval,ŷval,xlabel="true values", ylabel="estimated values", legend=nothing)
```
--------------------------------------------------------------------------------
### 13) (Optional) Use unscaled data
Run the same workflow without scaling the data. How this affect the quality of your predictions ? 

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
Random.seed!(123) # To get the same random numbers as before...
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
l1 = DenseLayer(14,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,1,f=relu)
mynn = buildNetwork([l1,l2,l3],squaredCost)
trainingLogs = train!(mynn,xtrain,ytrain,batchSize=6,epochs=400)
ŷtrain   = predict(mynn, xtrain) 
ŷval     = predict(mynn, xval)
trainRME = meanRelError(ŷtrain,ytrain,normRec=false) 
testRME  = meanRelError(ŷval,yval,normRec=false)
plot(trainingLogs.ϵ_epochs)
scatter(yval,ŷval,xlabel="true values", ylabel="estimated values", legend=nothing)
```
```@raw html
</details>
```
