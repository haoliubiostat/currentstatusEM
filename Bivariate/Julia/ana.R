### The plot of baseline function is done using R

res<- read.delim("estbaseline.txt", header=F)

x1<-res[,1]
a1<-res[,3]
x2<-res[,2]
a2<-res[,4]


pdf("hiv.pdf", width=5, height=5, pointsize=10)
par(mfcol=c(2,1), mar=c(3,3,1.5,1), mgp=c(1.5,0.5,0))


plot(x1, pnorm(a1), type="s", ylim=c(0,1), ylab="Prob of Virological Response", xlab="Days since randomization", lwd=1)
lines(x1, pnorm(a1-0.010), type="s", lty=2, lwd=2)
lines(x1, pnorm(a1-0.312), type="s", lty=3, lwd=2)

legend(10,0.95,legend = c("Efavirenz+nelfinavir", "Nelfinavir", "Efavirenz"), lty=1:3, lwd=c(1,2,2))

plot(x2, pnorm(a2), type="s", ylim=c(0,1), ylab="Prob of Treatment Response", xlab="Days since randomization", lwd=1)
lines(x2, pnorm(a2-0.14), type="s", lty=2, lwd=2)
lines(x2, pnorm(a2+0.015), type="s", lty=3, lwd=2)

legend(10,0.95,legend = c("Efavirenz+nelfinavir", "Nelfinavir", "Efavirenz"), lty=1:3, lwd=c(1,2,2))

dev.off()
