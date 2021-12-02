rm(list = ls(all = TRUE))
convert_list <- function(data){
  if(typeof(data)=='list'){
    data_temp = as.matrix(do.call(cbind, data))
    rownames(data_temp) = rownames(data)
    data = data_temp
  }
  return(data)
}

#example: Rscript Master_deconvolution.R baron none sc TMM TMM MuSiC 100 none 1

### arguments 'baron','GSE81547','Kidney_HCL','EMTAB5061'
dataset = '../source_data/baron'
transformation = 'none'
deconv_type = "sc"

if(deconv_type == "bulk"){
  normalization = args[4]
  marker_strategy = args[5]
} else if (deconv_type == "sc") {
  normalization_scC = "none"
  normalization_scT = "none"
} else {
  print("Please enter a valid deconvolution framework")
  stop()
}

number_cells = round(as.numeric(100), digits = -2) #has to be multiple of 100
to_remove = "none"#"unsure"
num_cores = min(as.numeric(1),parallel::detectCores()-1)

#-------------------------------------------------------
### Helper functions + CIBERSORT external code
source('./helper_functions.R')
# source('CIBERSORT.R')

#-------------------------------------------------------
### Read data and metadata
data = readRDS(list.files(path = dataset, pattern = "rds", full.names = TRUE))

data = convert_list(data)

full_phenoData = read.table(list.files(path = dataset, pattern = "phenoData", full.names = TRUE), header=TRUE, sep = '\t')

#-------------------------------------------------------
### QC 
require(dplyr); require(Matrix)

# First: cells with library size, mitochondrial or ribosomal content further than three MAD away were discarded
filterCells <- function(filterParam){
  cellsToRemove <- which(filterParam > median(filterParam) + 3 * mad(filterParam) | filterParam < median(filterParam) - 3 * mad(filterParam) )
  cellsToRemove
}

libSizes <- colSums(data)
gene_names <- rownames(data)

mtID <- grepl("^MT-|_MT-", gene_names, ignore.case = TRUE)
rbID <- grepl("^RPL|^RPS|_RPL|_RPS", gene_names, ignore.case = TRUE)

mtPercent <- colSums(data[mtID, ])/libSizes
rbPercent <- colSums(data[rbID, ])/libSizes

lapply(list(libSizes = libSizes, mtPercent = mtPercent, rbPercent = rbPercent), filterCells) %>% 
  unlist() %>% 
  unique() -> cellsToRemove

if(length(cellsToRemove) != 0){
  data <- data[,-cellsToRemove]
  full_phenoData <- full_phenoData[-cellsToRemove,]
}

# Keep only "detectable" genes: at least 5% of cells (regardless of the group) have a read/UMI count different from 0
keep <- which(Matrix::rowSums(data > 0) >= round(0.05 * ncol(data)))
data = data[keep,]

#-------------------------------------------------------
set.seed(24)
require(limma); require(dplyr); require(pheatmap)

original_cell_names = colnames(data)
colnames(data) <- as.character(full_phenoData$cellType[match(colnames(data),full_phenoData$cellID)])

# Keep CTs with >= 50 cells after QC
cell_counts = table(colnames(data))
to_keep = names(cell_counts)[cell_counts >= 50]
#to_keep = to_keep[-which(to_keep=='unsure')]
pDataC <- full_phenoData[full_phenoData$cellType %in% to_keep,]
to_keep = which(colnames(data) %in% to_keep)   
data <- data[,to_keep]
original_cell_names <- original_cell_names[to_keep]
colnames(data) <- original_cell_names

GSE50244.bulk.eset = readRDS('../source_data/GSE50244bulkeset.rds')
bulk.gene = rownames(GSE50244.bulk.eset)[rowMeans(exprs(GSE50244.bulk.eset)) != 0]

#EMTAB
EMTAB.eset = readRDS('../source_data/EMTABesethealthy.rds')
cm.gene = intersect(rownames(EMTAB.eset), bulk.gene)

library(xbioc)
s.ct = sampleNames(EMTAB.eset)[as.character(pVar(EMTAB.eset, 'cellType')) %in% c('alpha', 'beta', 'delta', 'gamma', 'acinar', 'ductal')]
EMTAB.eset <- EMTAB.eset[, s.ct, drop = FALSE]

pDataC_EMTAB = as.data.frame(lapply(EMTAB.eset@phenoData@data[c('sampleID','cellType')],as.character))
pDataC_EMTAB['cellID'] = rownames(EMTAB.eset@phenoData@data)

#Baron
cm.gene = intersect(bulk.gene, rownames(data))
data <- data[,which(unlist(pDataC['cellType']) %in% c('alpha', 'beta', 'delta', 'gamma', 'acinar', 'ductal'))]
pDataC <- pDataC[which(unlist(pDataC['cellType']) %in% c('alpha', 'beta', 'delta', 'gamma', 'acinar', 'ductal')),]
rownames(pDataC) <- seq(1,dim(pDataC)[1])
colnames(data)<-as.character(seq(1,dim(data)[2]))
pDataC['cellID']<-as.character(seq(1,dim(data)[2]))

#EMTAB+Baron
cm.gene = intersect(bulk.gene, rownames(data))
cm.gene = intersect(cm.gene, rownames(EMTAB.eset))
data_mix <- cbind(as.data.frame(data[cm.gene,]),as.data.frame((exprs(EMTAB.eset)[cm.gene,])))
pDataC_mix <- rbind(pDataC,pDataC_EMTAB)
colnames(data_mix)<-as.character(seq(1,dim(data_mix)[2]))
pDataC_mix['cellID']<-as.character(seq(1,dim(data_mix)[2]))

# "MuSiC","SCDC","BisqueRNA"
source('deconvolution.R')
for(m in c("NNLS","MuSiC","BisqueRNA")){
  RESULTS = Deconvolution_p(T = exprs(GSE50244.bulk.eset)[cm.gene, ], C = data_mix, method = m, phenoDataC = pDataC_mix)
  #RESULTS = Deconvolution_p(T = exprs(GSE50244.bulk.eset)[cm.gene, ], C = data[cm.gene, ], method = m, phenoDataC = pDataC) 
  #RESULTS = Deconvolution_p(T = exprs(GSE50244.bulk.eset)[cm.gene, ], C = exprs(EMTAB.eset)[cm.gene, ], method = m, phenoDataC = pDataC_EMTAB)
  write.table(RESULTS, file =paste("EMTAB+Baron_results/lambda_",m,".txt",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
}

write.table(GSE50244.bulk.eset@phenoData@data, file =paste("GSE50244_phenoData.txt",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)


#SCDC ensemble
#BisqueRNA requires "SubjectName" in phenoDataC
C = data
phenoDataC = pDataC
if(length(grep("[N-n]ame",colnames(phenoDataC))) > 0){
  sample_column = grep("[N-n]ame",colnames(phenoDataC))
} else {
  sample_column = grep("[S-s]ample|[S-s]ubject",colnames(phenoDataC))
}

colnames(phenoDataC)[sample_column] = "SubjectName"
rownames(phenoDataC) = phenoDataC$cellID

require(xbioc)
C.eset <- Biobase::ExpressionSet(assayData = as.matrix(C),phenoData = Biobase::AnnotatedDataFrame(phenoDataC))

require(SCDC)
RESULTS <- SCDC::SCDC_ENSEMBLE(bulk.eset = EMTAB.eset, sc.eset.list = list(baronh = C.eset,segerh=EMTAB.eset), ct.varname = "cellType", sample = "SubjectName", ct.sub = unique(as.character(phenoDataC$cellType)), iter.max = 200)
write.table(RESULTS, file =paste("EMTAB+Baron_results/lambda_SCDC_ENSEMBLE.txt",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)