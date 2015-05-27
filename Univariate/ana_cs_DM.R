### Data analysis using Probit model for current status data

options(width=100)

##### function to analyze the data

anal.dat <- function(mydata, tol0=1e-5, betai=c(1,0)) {
    
    xt <- mydata$xt
     z <- mydata$z
   dft <- mydata$dft
    
    pn <- length(xt)
    
#### EM
#### Inital
    
    alpha_old <- sort(log(xt))
    beta_old <- betai
    
    zo <- z[order(xt),]
    dfto <- dft[order(xt)]
        
#### Iteration

    eoi <- tol0 + 1
    iter <- 1
    
    while(eoi> tol0 & iter< 1000){
       
        iter <- iter +1
        
#### E step
        
        theta1 <- alpha_old + zo%*%beta_old
        
#### M Step
        
        tmp.the <- ifelse(dfto==1, pnorm(theta1), pnorm(theta1)-1)
        
        tmp3 <- ifelse(tmp.the==0, abs(theta1+1/theta1), dnorm(theta1)/tmp.the)
        
        imy <- theta1 + tmp3
        
        imybar <- imy-alpha_old
        
        beta_new <- coef(lm(imybar~zo-1))
        alpha_new <-isoreg(imy-zo%*%beta_new)$yf
        
        eoi <- sum(abs(beta_new-beta_old))
        
#### Log-likelihood
### likehood is increasing!!
        
## tmp1 <- alpha_new + zo%*%beta_new
## tmp2 <- ifelse(dfto==1, pnorm(tmp1), 1-pnorm(tmp1))
## cat("likelihood= ", log(prod(tmp2)),  "   beta= ", beta_new, "\n")

        alpha_old<-alpha_new
        beta_old<-beta_new
    }
    
        
    return(list(iter=(iter<1000), xto=sort(xt), hata=alpha_new, hatb=beta_old, cenp=sum(dft)/pn))
}

############# Fit the data

library(Epi)
data(DMconv)

mydat0 <- with(DMconv, list(xt = ifelse(is.na(dfi), (dlw-doe)/365.25, (dfi-doe)/365.25), z = as.matrix(cbind(as.numeric(gtol)-1, 2-as.numeric(grp))), dft = as.numeric(!is.na(dfi))))

pp<-2
myres0 <- anal.dat(mydat0, betai=rnorm(pp))

pdf("dm.pdf", width=6, height=4, pointsize=10)
par(mar=c(3,3,1.5,1), mgp=c(1.5,0.5,0))

plot(myres0$xto, pnorm(myres0$hata), lwd=2, type="s", xlab="Years since randomization",ylab="Probability of conversion to diabetes")

lines(myres0$xto, pnorm(myres0$hata)+myres0$hatb[1], lty=2, lwd=2)
legend(0.1,0.95,legend = c("Impaired glucose tolerance", "Impaired fasting glucose"), lty=1:2, lwd=2)
dev.off()

