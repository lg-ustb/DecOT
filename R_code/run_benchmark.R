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

# arguments 'baron','GSE81547','Kidney_HCL','EMTAB5061'
dataset = '../source_data/baron'
result_path = "../results/"

number_cells = round(as.numeric(100), digits = -2) #has to be multiple of 100
to_remove = "none"
num_cores = min(as.numeric(1),parallel::detectCores()-1)

#-------------------------------------------------------
### Helper functions + CIBERSORT external code
source('./helper_functions.R')
# source('CIBERSORT.R')

#-------------------------------------------------------
### Read data and metadata
data = readRDS(list.files(path = dataset, pattern = "rds", full.names = TRUE))
if(dataset=='GSE81547'){
  data = data[-c((nrow(data)-4):nrow(data)),]
}
  
data = convert_list(data)
full_phenoData = read.table(list.files(path = dataset, pattern = "phenoData", full.names = TRUE), header=TRUE, sep = ',')

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
### Data split into training/test
set.seed(24)
require(limma); require(dplyr); require(pheatmap)

original_cell_names = colnames(data)
colnames(data) <- as.character(full_phenoData$cellType[match(colnames(data),full_phenoData$cellID)])

# Keep CTs with >= 50 cells after QC
cell_counts = table(colnames(data))
to_keep = names(cell_counts)[cell_counts >= 50]
if(sum(which(to_keep=='unsure'))>0){
  to_keep = to_keep[-which(to_keep=='unsure')]
}
pData <- full_phenoData[full_phenoData$cellType %in% to_keep,]
to_keep = which(colnames(data) %in% to_keep)
data <- data[,to_keep]
original_cell_names <- original_cell_names[to_keep]


#
subjects <- pData$sampleID[!duplicated(pData$sampleID)]
for (s in subjects){
  # test <- data[,pData$sampleID == s]
  # train <- data
  #
  # if(sum(which(rowSums(train)==0), which(rowSums(test)==0)) != 0){
  #   train <- train[-c(which(rowSums(train)==0), which(rowSums(test)==0)),]
  #   test <- test[-c(which(rowSums(train)==0), which(rowSums(test)==0)),]
  # }
  # pDataC <- pData
  # train_cell_names <- original_cell_names
  # test_cell_names <- original_cell_names[pData$sampleID == s]
  # test_phenoData <- pData[pData$sampleID == s,]

  subject_data <- data[,pData$sampleID == s]
  subject_cell_counts = table(colnames(data[,pData$sampleID == s]))

  # Data split into train & test
  training <- as.numeric(unlist(sapply(unique(colnames(subject_data)), function(x) {
    sample(which(colnames(subject_data) %in% x), subject_cell_counts[x]/2) })))
  testing <- which(!1:ncol(subject_data) %in% training)

  # Generate phenodata for reference matrix C
  pDataC = pData[pData$sampleID == s,][training,]

  train <- subject_data[,training]
  test <- subject_data[,testing]
  train_cell_names <- original_cell_names[pData$sampleID == s][training]
  test_cell_names <- original_cell_names[pData$sampleID == s][testing]

  train <- cbind(train,data[,pData$sampleID != s])
  train_cell_names <- c(train_cell_names,as.vector(original_cell_names[pData$sampleID != s]))
  pDataC = rbind(pDataC, pData[pData$sampleID != s,])
  test_phenoData <- pData[pData$sampleID == s,][testing,]

  if(sum(which(rowSums(train)==0), which(rowSums(test)==0)) != 0){
    train <- train[-c(which(rowSums(train)==0), which(rowSums(test)==0)),]
    test <- test[-c(which(rowSums(train)==0), which(rowSums(test)==0)),]
  }

  if(exists("P")){
    rm("P")
  }
  dc_data = geneate_deconvolution_data(train, train_cell_names, test_cell_names, pDataC, test_phenoData, selected_genes="none")

  T = dc_data[[1]]
  C = dc_data[[2]]
  P = dc_data[[3]]
  pDataC = dc_data[[4]]

  # C = C[rownames(C) %in% selected_genes,]
  # T = T[rownames(T) %in% selected_genes,]
  # dataset = paste(dataset,'hvg_marker',sep='_')

  dir.create(paste(result_path,dataset,sep = ''))
  dir.create(paste(result_path,dataset,"/",s,sep = ''))
  
  write.table(T, file =paste(result_path,dataset,"/",s,"/",strsplit(dataset,'/')[[1]][2],"_T.csv",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
  write.table(C, file =paste(result_path,dataset,"/",s,"/",strsplit(dataset,'/')[[1]][2],"_C.csv",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
  write.table(pDataC, file =paste(result_path,dataset,"/",s,"/",strsplit(dataset,'/')[[1]][2],"_pDataC.csv",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
  write.table(P, file =paste(result_path,dataset,"/",s,"/",strsplit(dataset,'/')[[1]][2],"_P.csv",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)

  source('deconvolution.R')
  dir.create(paste(result_path,dataset,"/",s,"/lambda/",sep = ''))
  for(m in c("MuSiC","SCDC","BisqueRNA")){
    # RESULTS = Deconvolution(T = T, C = C, method = m, phenoDataC = pDataC, P = P, elem = to_remove, refProfiles.var = refProfiles.var)
    # RESULTS = RESULTS %>% dplyr::summarise(RMSE = sqrt(mean((observed_values-expected_values)^2)) %>% round(.,4),Pearson=cor(observed_values,expected_values) %>% round(.,4))
    # print(m)
    # print(RESULTS)

    RESULTS = Deconvolution_p(T = T, C = C, method = m, phenoDataC = pDataC, P = P, elem = to_remove)
    write.table(RESULTS, file =paste(result_path,dataset,"/",s,"/lambda/lambda_",m,".txt",sep = ''), sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
    print(paste(m,'completed...', sep=' '))
  }
  
  #-------------------------------------------------------
  # select hvg genes and marker genes
  # markers <- select_markers(train)
  # library(Seurat)
  # train_cellID = train
  # colnames(train_cellID) = train_cell_names
  # seurat_sce <- CreateSeuratObject(counts = train_cellID, project = "seurat_sce")
  # genes_n <- dim(train)[1]
  # seurat_sce <- FindVariableFeatures(seurat_sce,selection.method = "vst", nfeatures = floor(genes_n * 0.6))
  # seurat_hvg <- VariableFeatures(seurat_sce)
  # selected_genes = union(markers$gene,seurat_hvg)
  # write.table(selected_genes, file =paste(result_path,dataset,"/",s,"/",dataset,"_select_genes.csv",sep = ''), sep ="\t", row.names =FALSE, col.names =FALSE, quote =FALSE)
}

# method = "MuSiC"
# RESULTS = Deconvolution(T = T, C = C, method = method, phenoDataC = pDataC, P = P, elem = to_remove, refProfiles.var = refProfiles.var)
# RESULTS = RESULTS %>% dplyr::summarise(RMSE = sqrt(mean((observed_values-expected_values)^2)) %>% round(.,4),
#                                        Pearson=cor(observed_values,expected_values) %>% round(.,4))
# print(RESULTS)

# "MuSiC","SCDC","BisqueRNA"


  # subject_data <- data[,pData$sampleID == s]
  # subject_cell_counts = table(colnames(data[,pData$sampleID == s]))
  # 
  # # Data split into train & test  
  # training <- as.numeric(unlist(sapply(unique(colnames(subject_data)), function(x) {
  #   sample(which(colnames(subject_data) %in% x), subject_cell_counts[x]/2) })))
  # testing <- which(!1:ncol(subject_data) %in% training)
  # 
  # # Generate phenodata for reference matrix C
  # pDataC = pData[pData$sampleID == s,][training,]
  # 
  # train <- subject_data[,training]
  # test <- subject_data[,testing]
  # train_cell_names <- original_cell_names[pData$sampleID == s][training]
  # test_cell_names <- original_cell_names[testing]
  # 
  # train <- cbind(train,data[,pData$sampleID != s])
  # train_cell_names <- c(cell_names,as.vector(original_cell_names[pData$sampleID != s]))
  # pDataC = rbind(pDataC, pData[pData$sampleID != s,])

