Deconvolution_p <- function(T, C, method, phenoDataC, P = NULL, elem = NULL, STRING = NULL){ 
  
  sc_methods = c("MuSiC","BisqueRNA","DWLS","deconvSeq","SCDC","NNLS")
  
  ### For scRNA-seq methods 
    
  #BisqueRNA requires "SubjectName" in phenoDataC
  if(length(grep("[N-n]ame",colnames(phenoDataC))) > 0){
    sample_column = grep("[N-n]ame",colnames(phenoDataC))
  } else {
    sample_column = grep("[S-s]ample|[S-s]ubject",colnames(phenoDataC))
  }
    
  colnames(phenoDataC)[sample_column] = "SubjectName"
  rownames(phenoDataC) = phenoDataC$cellID
    
  require(xbioc)
  C.eset <- Biobase::ExpressionSet(assayData = as.matrix(C),phenoData = Biobase::AnnotatedDataFrame(phenoDataC))
  T.eset <- Biobase::ExpressionSet(assayData = as.matrix(T))
  
  
  ##########    MATRIX DIMENSION APPROPRIATENESS    ##########
  keep = intersect(rownames(C),rownames(T)) 
  C = C[keep,]
  T = T[keep,]
  
  ###################################
  if (method == "MuSiC"){
    
    require(MuSiC)
    RESULTS = t(MuSiC::music_prop(bulk.eset = T.eset, sc.eset = C.eset, clusters = 'cellType',
                                  markers = NULL, normalize = FALSE, samples = 'SubjectName', 
                                  verbose = F)$Est.prop.weighted)
    
  } else if (method == "NNLS"){ ##Proportion estimation with traditional deconvolution + >1 subject
    
    require(MuSiC)
    RESULTS = t(MuSiC::music_prop(bulk.eset = T.eset, sc.eset = C.eset, clusters = 'cellType',
                                  markers = NULL, normalize = FALSE, samples = 'SubjectName', 
                                  verbose = F)$Est.prop.allgene)
  } else if (method == "DWLS"){
    
    require(DWLS)
    path=paste(getwd(),"/results_",STRING,sep="")
    
    if(! dir.exists(path)){ #to avoid repeating marker_selection step when removing cell types; Sig.RData automatically created
      
      dir.create(path)
      Signature <- DWLS::buildSignatureMatrixMAST(scdata = C, id = as.character(phenoDataC$cellType), path = path, diff.cutoff = 0.5, pval.cutoff = 0.01)
      
    } else {#re-load signature and remove CT column + its correspondent markers
      
      load(paste(path,"Sig.RData",sep="/"))
      Signature <- Sig
      
      if(!is.null(elem)){#to be able to deal with full C and with removed CT
        
        Signature = Signature[,!colnames(Signature) %in% elem]
        CT_to_read <- dir(path) %>% grep(paste(elem,".*RData",sep=""),.,value=TRUE)
        load(paste(path,CT_to_read,sep="/"))
        
        Signature <- Signature[!rownames(Signature) %in% cluster_lrTest.table$Gene,]
        
      }
      
    }
    
    RESULTS <- apply(T,2, function(x){
      b = setNames(x, rownames(T))
      tr <- DWLS::trimData(Signature, b)
      RES <- t(DWLS::solveDampenedWLS(tr$sig, tr$bulk))
    })
    
    rownames(RESULTS) <- as.character(unique(phenoDataC$cellType))
    RESULTS = apply(RESULTS,2,function(x) ifelse(x < 0, 0, x)) #explicit non-negativity constraint
    RESULTS = apply(RESULTS,2,function(x) x/sum(x)) #explicit STO constraint
    
  } else if (method == "BisqueRNA"){#By default, BisqueRNA uses all genes for decomposition. However, you may supply a list of genes (such as marker genes) to be used with the markers parameter
    
    require(BisqueRNA)
    RESULTS <- BisqueRNA::ReferenceBasedDecomposition(T.eset, C.eset, markers=NULL, use.overlap=FALSE)$bulk.props #use.overlap is when there's both bulk and scRNA-seq for the same set of samples
    
  } else if (method == "deconvSeq"){
    
    singlecelldata = C.eset 
    celltypes.sc = as.character(phenoDataC$cellType) #To avoid "Design matrix not of full rank" when removing 1 CT 
    tissuedata = T.eset 
    
    design.singlecell = model.matrix(~ -1 + as.factor(celltypes.sc))
    colnames(design.singlecell) = levels(as.factor(celltypes.sc))
    rownames(design.singlecell) = colnames(singlecelldata)
    
    dge.singlecell = deconvSeq::getdge(singlecelldata,design.singlecell, ncpm.min = 1, nsamp.min = 4, method = "bin.loess")
    b0.singlecell = deconvSeq::getb0.rnaseq(dge.singlecell, design.singlecell, ncpm.min =1, nsamp.min = 4)
    dge.tissue = deconvSeq::getdge(tissuedata, NULL, ncpm.min = 1, nsamp.min = 4, method = "bin.loess")
    
    RESULTS = t(deconvSeq::getx1.rnaseq(NB0 = "top_fdr",b0.singlecell, dge.tissue)$x1) #genes with adjusted p-values <0.05 after FDR correction
    
  } else if (method == "SCDC"){ ##Proportion estimation with traditional deconvolution + >1 subject
    
    require(SCDC)
    RESULTS <- t(SCDC::SCDC_prop(bulk.eset = T.eset, sc.eset = C.eset, ct.varname = "cellType", sample = "SubjectName", ct.sub = unique(as.character(phenoDataC$cellType)), iter.max = 200)$prop.est.mvw)
    
  }
  
  RESULTS = RESULTS[gtools::mixedsort(rownames(RESULTS)),]
  return(RESULTS) 
  
}