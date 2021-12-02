dissTOM <-function(d, result_path, min_max=FALSE, z_score=FALSE){
  library(WGCNA)
  library(data.table)
  library(stringr)
  library(openxlsx)
  

  if (min_max){
    center <- sweep(d, 2, apply(d, 2, min),'-')
    R <- apply(d, 2, max) - apply(d,2,min) 
    d<- sweep(center, 2, R, "/") 
  }

  if (z_score){
    d = scale(d,center=TRUE,scale=TRUE)
  }
  
  d = t(d)
  
  #allowWGCNAThreads()
  enableWGCNAThreads()
  #ALLOW_WGCNA_THREADS=4
  #memory.limit(size = 20000)
  
  # 设定软阈值范围
  powers = c(c(2:10), seq(from = 12, to=30, by=2))
  # 获得各个阈值下的 R方 和平均连接度
  sft = pickSoftThreshold(d, powerVector = powers, verbose = 5, RsquaredCut = 0.85)
  # 作图：
  png(file=paste(result_path,"Scale_independence and Mean_connectivity.png",sep = ''),width=1200,height=600)
  par(mfrow = c(1,2));
  cex1 = 0.9;
  # Scale-free topology fit index as a function of the soft-thresholding power
  plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
       main = paste("Scale independence"));
  text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       labels=powers,cex=cex1,col="red");
  # this line corresponds to using an R^2 cut-off of h
  abline(h=0.80,col="red")
  # Mean connectivity as a function of the soft-thresholding power
  plot(sft$fitIndices[,1], sft$fitIndices[,5],
       xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
       main = paste("Mean connectivity"))
  text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
  dev.off()
  
  # 获得临近矩阵：
  softPower <- sft$powerEstimate
  cat('softPower: ',softPower,'\r\n')
  adjacency = adjacency(d, power = softPower);
  # 将临近矩阵转为 Tom 矩阵
  TOM = TOMsimilarity(adjacency);
  # 计算基因之间的相异度
  dissTOM = 1-TOM
  hierTOM = hclust(as.dist(dissTOM),method="average");
  
  png(file=paste(result_path,"dissTOM hist.png",sep = ''), width=1200, height=600)
  ADJ1_cor <- abs(WGCNA::cor(d,use = "p" ))^softPower
  # 基因多的时候使用下面的代码：
  k <- softConnectivity(datE=d,power=softPower) 
  par(mfrow=c(1,2))
  hist(k)
  scaleFreePlot(k,main="Check Scale free topology\n")
  dev.off()
  return(dissTOM)
}
