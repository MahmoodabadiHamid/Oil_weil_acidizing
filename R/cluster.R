library("ggplot2")
library("cluster")
library("factoextra")
library("magrittr")
# Load  and prepare the data


d <- X022_preprocessing_after_normalizing_values %>%
  na.omit() %>%          # Remove missing values (NA)
  scale()                # Scale variables
d=d[,-c(1,2,3,4,19,20,8)]
# View the firt 3 rows
head(d, n = 3)

# get_dist(): for computing a distance matrix between the rows of a data matrix. Compared to the standard dist() function, it supports correlation-based distance measures including "pearson", "kendall" and "spearman" methods.
# fviz_dist(): for visualizing a distance matrix

res.dist <- get_dist(d, stand = TRUE, method = "pearson")

fviz_dist(res.dist,
          gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
library("factoextra")
fviz_nbclust(d, kmeans, method = "gap_stat")          
          
          
set.seed(123)
km.res <- kmeans(d, 4, nstart = 25)
# Visualize
library("factoextra")
fviz_cluster(km.res, data = d,
                       ellipse.type = "convex",
                       palette = "jco",
                       ggtheme = theme_minimal())
############### the k-medoids/pam clustering can be computed as follow
# Compute PAM
library("cluster")
pam.res <- pam(d, 4)
# Visualize
fviz_cluster(pam.res)
          
############Hierarchical clustering
# Compute hierarchical clustering
res.hc <- USArrests %>%
            scale() %>%                    # Scale the data
            dist(method = "euclidean") %>% # Compute dissimilarity matrix
            hclust(method = "ward.D2")     # Compute hierachical clustering
          
# Visualize using factoextra
# Cut in 4 groups and color by groups
fviz_dend(res.hc, k = 4, # Cut in four groups
                    cex = 0.5, # label size
                    k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
                    color_labels_by_k = TRUE, # color labels by groups
                    rect = TRUE # Add rectangle around groups
          )
          