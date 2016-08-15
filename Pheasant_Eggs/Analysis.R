## ---- warning=FALSE, message=FALSE---------------------------------------
# Clear workspace
rm(list = setdiff(ls(), lsf.str())) 

# Load all libraries to be used in subsequent analyses
library(knitr)
library(rgl) # plot3d
library(rglwidget) # for 3D plotting
knit_hooks$set(webgl = hook_webgl) # to get interactive 3D plots in HTML mode
library(cluster) # silhouette diagram
library(corrgram) # correlogram
library(dendextend) # to colour dendrogram branches
library(RColorBrewer)

# Read data
data <- read.table(file="data.csv", header=T, sep=",", skip=0, row.names="Egg.ID")
# Clean data 
REMOVE <- c(79, 116, 485) # Egg.IDs to remove 
data <- data[!(row.names(data) %in% REMOVE), ] # include everything apart from REMOVE
# Normalise data (x' = [x - mean(x)] / std(x))
predictorNames <- setdiff(names(data), c("Ordering", "Pen", "File", "File.name"))
xTrain <- data[, predictorNames] # unsupervised learning we don't split into train/test
meanXTrain <- apply(xTrain, 2, mean) # mean of each attribute
sdXTrain <- apply(xTrain, 2, sd) # standard deviation of each attribute
xTrain <- sweep(xTrain, 2, meanXTrain, FUN="-") # remove means
xTrain <- sweep(xTrain, 2, sdXTrain, FUN="/") # normalise by standard deviation

## ---- warning=FALSE, message=FALSE, fig.width=10, fig.height=10----------
# Pair-wise scatter plots
pairs(x=xTrain, upper.panel=NULL, cex.labels=1, pch='.', col="grey")
# Compute Pearson correlation coefficients
corrgram(x=xTrain, lower.panel=panel.shade, upper.panel=panel.conf)

## ---- warning=FALSE, message=FALSE---------------------------------------
pca <- prcomp(x=xTrain, retx=TRUE, center=FALSE, scale.=FALSE) # apply PCA
summary(pca)
varianceExplained <- pca$sdev^2 / sum(pca$sdev^2) # compute explained variance
varianceExplained <- varianceExplained[1:10] # plot only the top 10 PCs
# Plot % variance explained by each principal component (PC)
barplot(100*varianceExplained, las=1, ylab='Variance Explained (%)', 
        names.arg=paste("PC", seq(length(varianceExplained)), sep=""))
# Plot cumulative % variance explained by each principal component (PC)
plot(cumsum(100*varianceExplained), las=1, ylab='Cumulative Variance Explained (%)', 
     xlab="Principal Component (PC)", type="b", pch=19)
# Plot pair-wise scatter plot and pearson corelation coefficient for top 4 PCs 
pairs(x=pca$x[, 1:4], upper.panel=NULL, cex.labels=1, pch='.', col="grey")
corrgram(x=pca$x[, 1:4], lower.panel=panel.shade, upper.panel=panel.conf)

## ---- warning=FALSE, message=FALSE, fig.width=5, fig.height=10, fig.align='center'----
par(mfrow=c(4, 1))
yMin <- min(pca$rotation^2)
yMax <- max(pca$rotation^2)
# Plot and compare weights^2 as sum(weights^2)=1
for (i in seq(4))
{
    barplot(pca$rotation[, i]^2, col="grey", ylim=c(yMin, yMax), las=2, 
            main=paste("PC", i, sep=""), ylab="Weight$^2$")
}

## ---- warning=FALSE, message=FALSE, webgl=TRUE, fig.width=8, fig.height=8, fig.align='center'----
par(mfrow=c(1, 1)) # reset plotting area
# Name features
PC1 <- "Eggshell brightness"
PC2 <- "Eggshell greenness"
PC3 <- "Egg size"
xTrain <- pca$x[data$Pen==1, 1:4] # keep only first 4 PCs and pen=1
# Plot PC1 vs PC2
plot(xTrain[, 1], xTrain[, 2], xlab=PC1, ylab=PC2, type="n")  
text(xTrain[, 1], xTrain[, 2], labels=row.names(xTrain), col="grey")
# Plot PC1 vs PC2 vs PC3
plot3d(xTrain[, 1], xTrain[, 2], xTrain[, 3], xlab=PC1, ylab=PC2, zlab=PC3, type="n")
text3d(xTrain[, 1], xTrain[, 2], xTrain[, 3], row.names(xTrain), col="grey")

## ---- warning=FALSE, message=FALSE, webgl=TRUE, fig.width=8, fig.height=8, fig.align='center'----
set.seed(101) # for reproducibility
fit <- kmeans(xTrain, centers=3, nstart=50, iter.max=1001)
# Visualise result in 2D
plot(xTrain[, 1], xTrain[, 2], xlab=PC1, ylab=PC2, type="n",  cex.lab=1.4, cex.axis=1.4)  
text(xTrain[, 1], xTrain[, 2], labels=row.names(xTrain), col=fit$cluster, cex=1.4) # egg ID
# Visualise result in 3D
plot3d(xTrain[, 1], xTrain[, 2], xTrain[, 3], xlab=PC1, ylab=PC2, zlab=PC3, type="n")
text3d(xTrain[, 1], xTrain[, 2], xTrain[, 3], row.names(xTrain), col=fit$cluster)

## ---- warning=FALSE, message=FALSE, fig.width=10, fig.height=10, fig.align='center'----
set.seed(101) # for reproducibility
colours <- brewer.pal(n=5, name="Set1")
kRange <- seq(from=2, to=5, by=1)
par(mfrow=c(2, 2))
for (k in kRange)
{
    fit <- kmeans(xTrain, centers=k, nstart=50, iter.max=1001)
    silh <- silhouette(x=fit$cluster, dist=daisy(xTrain)^2) 
    plot(silh, main=paste("k=", k), col=colours[1:k])
}

## ---- warning=FALSE, message=FALSE, fig.width=10, fig.height=8, fig.align='center'----
par(mfrow=c(1, 1)) # reset plot to 1 x 1 
distance <- dist(xTrain, method="euclidean") # distance function = euclidean distance
fit <- hclust(d=distance, method="ward.D2") # Ward's method
dend <- as.dendrogram(fit) # convert hclust to dendrogram class
dend <- color_branches(dend, k=3, col=c(2, 1, 3)) # match k-means colours
plot(dend, ylab="Distance", xlab="Egg ID", cex.lab=1.3, cex.axis=1.3)

