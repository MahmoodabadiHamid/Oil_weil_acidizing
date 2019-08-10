# dim returns the no of examples(rows) and columns present in the dataset.
# names command tell the column names.
# str tells the data types of each column
d=X022_preprocessing_after_normalizing_values[,-c(1,2,3,4,19,20,8)]

#Let us use the cluster package available in R. If not install.
# we apply clustering algorithm on this data set which do not have labels.
# We use partitioning Around Medoids(PAM)which is Partitioning (clustering)
#of the data into k clusters "around medoids",a more robust version of K-means.
# We have passed the number of desirable clusters on this data set.
library(cluster)
cl3 <- pam(d, 4)$clustering
cl3


#clusplot returns a plot which describes the formation of three clusters of data points.
clusplot(d, cl3, color = TRUE)

#from the clusterplot, it is clerly shown some data points belong to both classes.
# we will examine this in detail with the fanny object.
#fanny d tells the amount of membership(rounded) of each data point to a particular cluster.
# The firat data point belongs to Cluster A, that is Class A by 91%
fanny_d<-fanny(d,4)
fanny_d
write.csv(fanny_d$membership,"E:/company/python/Oil_well_acidizing/Dataset/Second Dataset/fuzzy_membership.csv")
write.csv(fanny_d$clustering,"E:/company/python/Oil_well_acidizing/Dataset/Second Dataset/fuzzy_cluster.csv")

#We will observe this phenomena in more detail with another plot.
library(ggplot2)
library(factoextra)
fviz_cluster(fanny_d, ellipse.type = "norm", repel = TRUE,
             palette = "jco", ggtheme = theme_minimal(),
             legend = "right")
# We will have one more plot and visualize the clusters formed.
fviz_silhouette(fanny_d, palette = "jco",
                ggtheme = theme_minimal())

