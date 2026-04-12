pairwise_de <- function(df1, df2, title, outbase){
	# ORdered data frame
	o_df <- rbind(df1, df2)
	condition <- rep(c(1, 2), times=c(nrow(df1), nrow(df2)))

	# Prepare for MAST
	wellKey <- rownames(o_df)
	cdata <- data.frame(cbind(wellKey=wellKey, condition=condition))
	fdata <- data.frame(primerid=colnames(o_df))

	# SCa data
	sca <- FromMatrix( t(o_df), cdata, fdata)
	cdr2 <-colSums(assay(sca)>0)
	colData(sca)$cngeneson <- scale(cdr2)
    colData(sca)$cond <- factor(unlist(as.list(condition)))
    colData(sca)$cond <- relevel(colData(sca)$cond, 2)

    # Fits
    zlmCond <- zlm(~cond + cngeneson, sca, parallel=TRUE)
    summaryDt <- summary(zlmCond, doLRT='cond1')$datatable
	
	# Significance table
	fcHurdle <- merge(summaryDt[contrast=='cond1' & component=='H',.(primerid, `Pr(>Chisq)`)], 
		summaryDt[contrast=='cond1' & component=='logFC', .(primerid,coef, ci.hi, ci.lo)], by='primerid') 

	# FDR
	fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
	setorder(fcHurdle, fdr)

	# Export data
	write.csv(as.data.frame(fcHurdle), sprintf("%s%s.csv", outbase, title), quote=FALSE)
}