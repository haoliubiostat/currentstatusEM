############ Functions for the analysis of bivariate current status data 
############ Using the semiparametric Probit model

using Distributions
using Isotonic
using NLopt

#### univariate current status without the information bound

function uni_cur(mydata::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}; tol0=1e-5, maxi=1000)

  xt, dft, z=mydata;
  nn=length(xt);

  pp=size(z)[2];
  
  permxt = sortperm(xt);
  xto = xt[permxt];
  dfto = dft[permxt];
  zo = z[permxt,:]

  alpha_old = sort(randn(nn));
  beta_old = rand(pp);

  toli = tol0+1.0;
  eti = 1;
  
  while toli > tol0 && eti<=maxi
    eti+=1;

    theta1 = alpha_old + zo * beta_old

    tmp_0 = cdf(Normal(),theta1);
    tmp_1 = Float64[ dfto[i]==1.0 ? tmp_0[i] : tmp_0[i]-1.0 for i=1:nn];

    estep = Float64[ tmp_1[i]==0.0 ? 0.0 :  theta1[i] + pdf(Normal(),theta1[i])/tmp_1[i] for i=1:nn];
    
    mybar = estep .- alpha_old;

    beta_new = zo\mybar;

    resi = estep - zo * beta_new
    
    alpha_new = isotonic_regression(resi)

    toli = sum(abs(beta_new-beta_old));
### println(myid(), " aaa", toli)

    alpha_old= alpha_new;
    beta_old = beta_new;
  end;

  beta_old, xto, alpha_old, sum(dft)/nn;


end;

################# Observed log-likelihood

function logobslik(par::Vector{Float64}, grad, a::Tuple{Vector{Float64}, Vector{Float64}}, dat::Tuple{Matrix{Float64}, Matrix{Float64}})
  
  dlt, z=dat;
  a1, a2=a;
  ss=size(z);
  nn=ss[1];
  pp=Int64(ss[2]/2);

  z1 =z[:,1:pp];
  z2 =z[:,(pp+1):(2*pp)];

  beta1 = par[1:pp]
  beta2 = par[(pp+1):(2*pp)]
  rho  = par[2*pp+1]

  mu1 = a1+ z1 * beta1
  mu2 = a2+ z2 * beta2

  sig=[1.0 rho; rho 1.0]
  
  loglik=0.0;

  for i= 1:nn
  
    if dlt[i,1]==1.0 && dlt[i,2]==1.0
      pb=pmvnorm([-Inf; -Inf], [mu1[i]; mu2[i]], sig)
      
    elseif dlt[i,1]==0.0 && dlt[i,2]==1.0
      pb=pmvnorm([mu1[i]; -Inf], [Inf; mu2[i]], sig)
      
    elseif dlt[i,1]==1.0 && dlt[i,2]==0.0
      pb=pmvnorm([-Inf; mu2[i]], [mu1[i]; Inf], sig)
      
    elseif dlt[i,1]==0.0 && dlt[i,2]==0.0
      pb=pmvnorm([mu1[i]; mu2[i]], [Inf; Inf], sig)
#    else
#      pb=0.0
    end;
##    cat(pb, "|",dlt[i,:],"| ", mu1[i], "| ", mu2[i], "| ", sig, "\n")    
    loglik=loglik + (abs(pb)==0.0 ? -10000.0 : log(abs(pb)))
    
  end;
  
  return loglik;

end;

################ expected log-likelihood

function logexplik(par::Vector{Float64}, grad, a::Tuple{Vector{Float64}, Vector{Float64}}, z::Matrix{Float64}, ym::Matrix{Float64}, ym2::Array{Float64})
  
  a1, a2=a;
  ss=size(z);
  nn=ss[1];
  pp=Int64(ss[2]/2);

  z1 =z[:,1:pp];
  z2 =z[:,(pp+1):(2*pp)];

  beta1 = par[1:pp]
  beta2 = par[(pp+1):(2*pp)]
  rho  = par[2*pp+1]

  mu1 = a1+ z1 * beta1
  mu2 = a2+ z2 * beta2
  
  y11=reshape(ym2[1,1,:], nn);
  y12=reshape(ym2[1,2,:], nn);
  y22=reshape(ym2[2,2,:], nn);

  tmp1 = y11 - 2.0*rho*y12 + y22;  
  tmp2 = -2.0* ym[:,1].*(mu1-rho*mu2);
  tmp3 = -2.0* ym[:,2].*(mu2-rho*mu1);
  tmp4 = mu1.^2.0+mu2.^2.0 - 2.0*rho*mu1.*mu2;
    
  temp = tmp1+tmp2+tmp3+tmp4;

  fval = - 0.5*sum(temp)/nn/(1.0-rho^2.0) - 0.5*log(1.0-rho^2.0)
  
  return -fval;

end;

################# MLE
#### data form (x, dlt, z)

function bivcur(mydata::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}; tol0=1.0e-8, maxi=10000)

  xt, dlt, z=mydata;
  ss=size(z);
  nn=ss[1];
  pp=Int64(ss[2]/2);
  
  z1 =z[:,1:pp];
  z2 =z[:,(pp+1):(2*pp)];
  xt1=xt[:,1];
  xt2=xt[:,2];

  ###

  a1=zeros(nn);
  a2=zeros(nn); 
  new_a1=zeros(nn);
  new_a2=zeros(nn);  
  ym=zeros(nn,2);
  ym2=zeros(2,2,nn);

  #### using the univariate data to find the initial values
  
  psubeta1, xto1, psua1=uni_cur((xt1, dlt[:,1], z1));
  psubeta2, xto2, psua2=uni_cur((xt2, dlt[:,2], z2));
  
  mypar=[psubeta1; psubeta2; rand(1)-0.5];

  ####  orders of xt values

  oxt1=sortperm(xt1);
  oxt2=sortperm(xt2);

  a1[oxt1] = psua1;
  a2[oxt2] = psua2;

#  a1=rand(nn)
#  a2=rand(nn)
#  mypar=[rand(pp); rand(pp); rand(1)-0.5];

  opt1 = Opt(:GN_DIRECT_L, 2*pp+1);
  
  xtol_rel!(opt1, 1e-8);
  lower_bounds!(opt1, [-20.0*ones(2*pp);-0.99]);
  upper_bounds!(opt1, [ 20.0*ones(2*pp); 0.99]);
  maxeval!(opt1, 1000);
    
  opt = Opt(:LN_BOBYQA, 2*pp+1);
  
  xtol_rel!(opt, 1e-8);
  lower_bounds!(opt, [-20.0*ones(2*pp);-0.99]);
  upper_bounds!(opt, [ 20.0*ones(2*pp); 0.99]);
  maxeval!(opt, 1000);
  
  toli = tol0+1.0;
  eti = 1;
  while toli > tol0 && eti<=maxi
    eti+=1;

    beta1 = mypar[1:pp]
    beta2 = mypar[(pp+1):(2*pp)]
    rho = mypar[2*pp+1]

    mu1 = a1+ z1 * beta1;
    mu2 = a2+ z2 * beta2;

### E-Step

    for i= 1:nn
  
      if dlt[i,1]==1.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([0.0; 0.0], [Inf; Inf], rho,  mu=[mu1[i]; mu2[i]]);
      elseif dlt[i,1]==0.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([-Inf; 0.0], [0.0; Inf], rho,  mu=[mu1[i]; mu2[i]]);           
      elseif dlt[i,1]==1.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([0.0; -Inf], [Inf; 0.0], rho,  mu=[mu1[i]; mu2[i]]);          
      elseif dlt[i,1]==0.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([-Inf; -Inf], [0.0; 0.0], rho,  mu=[mu1[i]; mu2[i]]);     
      end;
        
      ym[i,:], ym2[:,:,i]=ztem;
      
    end;
 
### CM Step

    imu2 = ym[:,2] - z2 * beta2 + rho*( a1 + z1 * beta1 - ym[:,1]);    
    imu2a =isotonic_regression(imu2[oxt2]);

### new_a2 is unsorted baseline function for a2()

    new_a2[oxt2] = imu2a; 

    mu2 = new_a2+ z2 * beta2;
    
### Re-do the E-Step

    for i= 1:nn
  
      if dlt[i,1]==1.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([0.0; 0.0], [Inf; Inf], rho,  mu=[mu1[i]; mu2[i]]);
      elseif dlt[i,1]==0.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([-Inf; 0.0], [0.0; Inf], rho,  mu=[mu1[i]; mu2[i]]);           
      elseif dlt[i,1]==1.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([0.0; -Inf], [Inf; 0.0], rho,  mu=[mu1[i]; mu2[i]]);          
      elseif dlt[i,1]==0.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([-Inf; -Inf], [0.0; 0.0], rho,  mu=[mu1[i]; mu2[i]]);     
      end;
        
      ym[i,:], ym2[:,:,i]=ztem;
      
    end;
    

### CM Step

    imu1 = ym[:,1] - z1 * beta1 + rho*( new_a2 + z2 * beta2 - ym[:,2]);    
    imu1a= isotonic_regression(imu1[oxt1]);
    new_a1[oxt1] = imu1a; 

    mu1 = new_a1+ z1 * beta1;
    
### E-Step

    for i= 1:nn
  
      if dlt[i,1]==1.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([0.0; 0.0], [Inf; Inf], rho,  mu=[mu1[i]; mu2[i]]);
      elseif dlt[i,1]==0.0 && dlt[i,2]==1.0
        ztem = mtmvnorm([-Inf; 0.0], [0.0; Inf], rho,  mu=[mu1[i]; mu2[i]]);           
      elseif dlt[i,1]==1.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([0.0; -Inf], [Inf; 0.0], rho,  mu=[mu1[i]; mu2[i]]);          
      elseif dlt[i,1]==0.0 && dlt[i,2]==0.0
        ztem = mtmvnorm([-Inf; -Inf], [0.0; 0.0], rho,  mu=[mu1[i]; mu2[i]]);     
      end;
        
      ym[i,:], ym2[:,:,i]=ztem;
      
    end;
### CM Step 

    min_objective!(opt1, (x,y)-> logexplik(x, y, (new_a1, new_a2), z, ym, ym2))

    mypar_old=mypar
    dopt=optimize(opt1, mypar)
    mypar=dopt[2]

    min_objective!(opt, (x,y)-> logexplik(x, y, (new_a1, new_a2), z, ym, ym2))
    dopt=optimize(opt, mypar)
    mypar=dopt[2]

    toli=sum((a1-new_a1).^2.0+(a2-new_a2).^2.0)+sum((mypar-mypar_old).^2.0);
    a1=new_a1;
    a2=new_a2;

  end

  a1 =new_a1[oxt1]
  a2 =new_a2[oxt2]
  
  cenp=zeros(4);
  for i= 1:nn  
    if dlt[i,1]==1.0 && dlt[i,2]==1.0
      cenp[1] += 1.0;        
    elseif dlt[i,1]==0.0 && dlt[i,2]==1.0
      cenp[2] += 1.0;
    elseif dlt[i,1]==1.0 && dlt[i,2]==0.0
      cenp[3] += 1.0;          
    elseif dlt[i,1]==0.0 && dlt[i,2]==0.0
      cenp[4] += 1.0;
    end;
  end;

  return mypar, a1, a2, cenp, eti<=maxi

end;


##### Psudo-likelihood estimations

function biv_psudo(mydata::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}; parinit=zeros(5), t0=exp([-0.5 -0.25 0 0.25 0.5;-0.5 -0.25 0 0.25 0.5]), tol0=1.0e-5, maxi=1000)
  
  xt, dlt, z=mydata;
  ss=size(z);
  nn=ss[1];
  pp=Int64(ss[2]/2);
  
  z1 =z[:,1:pp];
  z2 =z[:,(pp+1):(2*pp)];
  xt1=xt[:,1];
  xt2=xt[:,2];

  psu1a= zeros(nn); 
  psu2a= zeros(nn); 

  psubeta1, xto1, psua1=uni_cur((xt1, dlt[:,1], z1));
  psubeta2, xto2, psua2=uni_cur((xt2, dlt[:,2], z2));

  oxt1=sortperm(xt1);
  oxt2=sortperm(xt2);

  psu1a[oxt1]= psua1; 
  psu2a[oxt2]= psua2; 
  
  opt = Opt(:GN_DIRECT_L, 2*pp+1)

  xtol_rel!(opt, 1.0e-7)
  lower_bounds!(opt, [-10.0*ones(2*pp);-0.99])
  upper_bounds!(opt, [ 10.0*ones(2*pp); 0.99])
  maxeval!(opt, 1000)

  max_objective!(opt, (x,y)-> logobslik(x, y, (psu1a, psu2a), (dlt,z)))
  hatval=optimize(opt, parinit)[2]

  ind1 = map((x)->searchsortedlast(sort(xt1), x), t0[1,:])
  ind2 = map((x)->searchsortedlast(sort(xt1), x), t0[1,:])
    
  a1val= ind1==0 ? psua1[ind1+1] : psua1[ind1];
  a2val= ind2==0 ? psua2[ind2+1] : psua2[ind2];

  return [hatval' a1val a2val]

end;


######################  Bootstrap

function boot_bivcur(bootn::Int64, mydata::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}})

  xt, dlt, z=mydata;
  ss=size(z);
  nn=ss[1];
  pp=Int64(ss[2]/2);
  
  mle_bt = -99.0*ones(Float64, bootn, 2*pp+1);
  emcheck= -1*ones(Int, bootn);
  
  for i in 1:bootn
    ind=rand(1:nn, nn)
    mydata_b=(xt[ind,:], dlt[ind,:], z[ind,:])    
    mle_bt[i,:], a1, a2, cenp, emcheck[i,:] = bivcur(mydata_b);    
  end;
  
  mle_bt, emcheck;
  
end;

######## parallel computation

function parall_bivcur(nproc::Int64, bootn::Int64, mydata::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}; rseed=100)
  
  [@spawnat i srand(rseed+i) for i in workers()];

  eachn=div(bootn, nproc);
  mylist= [eachn for i in 1:nproc];
  
  res=pmap(x->boot_bivcur(x, mydata), mylist);
  
  res
  
end;


############################################################################################
####### bivariate pmvnorm for p(|lower<X<upper, lower<Y<upper)
####### based on truncated bivaraite t distribution bvtcdf with nu=0
####### Confirmed with R results


include("tvpack.jl")
 
function pmvnorm(lower::Vector{Float64}, upper::Vector{Float64}, sigma::Matrix{Float64}; mu=zeros(Float64,2))

  if any(lower.>upper)
    return NaN
  end;
  
  lower = (lower - mu)./sqrt(diag(sigma));
  upper = (upper - mu)./sqrt(diag(sigma));

  rho =cov2cor(sigma)[2,1]

  if any(upper.<0.0)
    l=lower;
    u=upper;
  else
    l=-upper;
    u=-lower;
  end;


  if u[1]==Inf 
    a1=cdf(Normal(), u[2])
  elseif u[2]==Inf 
    a1=cdf(Normal(), u[1])
  else     
    a1=bvtcdf(0, u[1], u[2], rho);
  end;

  if u[2]==Inf
    a2=cdf(Normal(), l[1])
  elseif l[1]==Inf
    a2=cdf(Normal(), u[2])
  elseif l[1]==-Inf
    a2=0.0
  else
    a2=bvtcdf(0, l[1], u[2], rho);
  end;

  if u[1]==Inf
    a3=cdf(Normal(), l[2])
  elseif l[2]==Inf
    a3=cdf(Normal(), u[1])
  elseif l[2]==-Inf
    a3=0.0
  else
    a3=bvtcdf(0, u[1], l[2], rho);
  end;
 
  if l[1]==Inf 
    a4=cdf(Normal(), l[2])  
  elseif l[2]==Inf 
    a4=cdf(Normal(), l[1]) 
  elseif l[2]==-Inf || l[1]==-Inf
    a4=0.0
  else 
    a4=bvtcdf(0, l[1], l[2], rho);
  end;
 
## print("l=",l, " u=",u, " ||", a1, "|", a2, "|", a3, "|", a4, "\n")
  return a1-a2-a3+a4;
end;


###

function cov2cor(C::AbstractMatrix)
    #should check if C is positive semidefinite

    sigma = sqrt(diag(C))
    return C ./ (sigma*sigma')
end


####################################
####### Based on Muthen 1990
####### output, mean and 2nd moment
####### Confirmed with R results
####### difference: mtmvnorm([-Inf, -Inf], [0.0, 0.0], rho,  mu=[11, 1])

function mtmvnorm(lower::Vector{Float64}, upper::Vector{Float64}, rho::Float64;  mu=zeros(Float64,2))
  
  a=upper-mu;
  b=lower-mu;
  
  sigma=[1.0 rho;rho 1.0]
  prob=pmvnorm(b, a, sigma)
  
  c=1.0/sqrt(1.0-rho^2)

  mval=[0.0, 0.0];
  m2val=[0.0 0.0; 0.0 0.0];
  for i=1:2    
    j=3-i

    if all(isinf(a)) || rho==0.0
      y1=a[j]; y3=a[i];
    else
      y1= a[j]-rho*a[i];  
      y3= a[i]-rho*a[j];
    end;
    
    if all(isinf(b)) ||  rho==0.0
      z2=b[j]; z4=b[i];
    else
      z2= b[j]-rho*b[i];  
      z4= b[i]-rho*b[j];
    end;
    
    if (isinf(a[i]) && isinf(b[j])) || rho==0.0
      z1=b[j]; y4=a[i];
    else
      z1= b[j]-rho*a[i];  
      y4= a[i]-rho*b[j];
    end;

    if (isinf(a[j]) && isinf(b[i])) ||  rho==0.0
      y2=a[j]; z3=b[i];
    else
      y2= a[j]-rho*b[i];  
      z3= b[i]-rho*a[j];
    end;
      
    x1= pdf(Normal(), a[i])*(cdf(Normal(), y1*c) - cdf(Normal(), z1*c)) 
    x2= pdf(Normal(), b[i])*(cdf(Normal(), y2*c) - cdf(Normal(), z2*c)) 
    x3= pdf(Normal(), a[j])*(cdf(Normal(), y3*c) - cdf(Normal(), z3*c)) 
    x4= pdf(Normal(), b[j])*(cdf(Normal(), y4*c) - cdf(Normal(), z4*c)) 
    
## mean value
    mval[i]=-x1+x2+rho*(-x3+x4);

## 2nd moment

    dx1=  pdf(Normal(), a[i])*(pdf(Normal(), y1*c) - pdf(Normal(), z1*c)) 
    dx2=  pdf(Normal(), b[i])*(pdf(Normal(), y2*c) - pdf(Normal(), z2*c)) 
    dx3=  pdf(Normal(), a[j])*(pdf(Normal(), y3*c) - pdf(Normal(), z3*c)) 
    dx4=  pdf(Normal(), b[j])*(pdf(Normal(), y4*c) - pdf(Normal(), z4*c)) 
    

    cinv = 1.0/c
    tmp1 = isinf(a[i]) ? [0.0, cinv] *dx1 : - [1.0, rho]* a[i]*x1 + [0.0, cinv] *dx1
    tmp2 = isinf(b[i]) ? -[0.0, cinv]*dx2 :   [1.0, rho]* b[i]*x2 - [0.0, cinv] *dx2
    tmp3 = isinf(a[j]) ? [cinv, 0.0] *dx3 : - [rho, 1.0]* a[j]*x3 + [cinv, 0.0] *dx3 
    tmp4 = isinf(b[j]) ? -[cinv, 0.0]*dx4 :   [rho, 1.0]* b[j]*x4 - [cinv, 0.0] *dx4
  
### 2nd moment yy, xy value    

    xy = [1.0, rho] * prob + tmp1+tmp2+rho*(tmp3+ tmp4)  
    
    if i==1
      m2val[1:2,i] = xy
    else
      [m2val[j,i] = xy[3-j] for j=1:2]
    end;
    
##    print(prob, "| ", y1, ", ",  z1, "| ",  cdf(Normal(), z1*c), "\n")
##   print(tmp1, "|", tmp2, "|", tmp3, "|", tmp4, "\n")
    
  end;
  
  if prob==0.0
    tmean=mu;
    t2momnt=sigma;
  else
    
    tmean=mval/prob+mu;
#    t2momnt=m2val/prob;

    taa = mval/prob
    tbb = (mu + taa) * (mu + taa)' - taa * taa'
    
        t2momnt=m2val/prob + tbb
  end;
  
  return tmean, t2momnt;
  
end;



