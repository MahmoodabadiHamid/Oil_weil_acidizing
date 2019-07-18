library(bnlearn)
a=X02_preprocessing_after_normalizing_values[,-c(1,22,25)]
dim(a)
a
#b=empty.graph(names(a))
#plot(b)
g=plot(gs(a))
#arcs(c)
#modelstring(c)
d=bn.fit(hc(a),a)
arcs(d)
coef(d)
modelstring(d)
nparams(d)
h=plot(hc(a))
#as.data.frame()
#is.data.frame(a)
learning.test
aa=X01_preprocessing_after_encoding_label[,-c(1,21,24)]
hh=plot(hc(aa))
all.equal(g,hh)
?all.equal
plplot(,,labels(a$`Carbonate/Sandstone`))
plot(a$Longitude,a$Latitude, col = c("red", "blue")[a$`Carbonate/Sandstone`])
plot(a$Longitude,a$Latitude, col = rgb(0, 1, a$`Carbonate/Sandstone`))
