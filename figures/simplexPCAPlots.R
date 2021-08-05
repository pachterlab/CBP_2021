pdf("~/Desktop/2simplex.pdf", width = 4, height = 4)
m <- diag(3); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "2-simplex")
dev.off()

pdf("~/Desktop/4simplex.pdf", width = 4, height = 4)
m <- diag(5); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "4-simplex")
dev.off()

pdf("~/Desktop/9simplex.pdf", width = 4, height = 4)
m <- diag(10); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "9-simplex")
dev.off()

pdf("~/Desktop/14simplex.pdf", width = 4, height = 4)
m <- diag(15); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "14-simplex")
dev.off()

pdf("~/Desktop/19simplex.pdf", width = 4, height = 4)
m <- diag(20); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "19-simplex")
dev.off()

pdf("~/Desktop/49simplex.pdf", width = 4, height = 4)
m <- diag(50); m<-scale(m,center=TRUE); p<-prcomp(m);

plot(p$x[,1],p$x[,2],pch=21,bg='cornflowerblue',col='dodgerblue4',xlab = "PC1", ylab = "PC2")
title(main = "49-simplex")
dev.off()
