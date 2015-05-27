res<- read.delim("estbaseline.txt", header=F)

pdf("autism.pdf", width=6, height=4, pointsize=10)
par(mfcol=c(1,1), mar=c(3,3,1.5,1), mgp=c(1.5,0.5,0))
plot(sort(res[,1]), pnorm(res[,3]), type="s", ylim=c(0,1), ylab="Probability of autism onset", xlab="Age in years", lwd=2)
lines(sort(res[,2]), pnorm(res[,4]), type="s", lty=2, lwd=2)
legend(20,0.7,legend = c("Girls", "Boys"), lty=1:2, lwd=2)
dev.off()
