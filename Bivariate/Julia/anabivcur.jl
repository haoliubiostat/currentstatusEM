############################################
#### Analysis of the HIV data; Grouping vraiables are done according to the 
#### clinical paper Shafer et al. (2003). Comparison of four-drug regimens and pairs of
#### sequential three-drug regimens as initial therapy for HIV-1 infection. 
#### New England Journal of Medicine 349, 2304–2315.

nump=20; addprocs(nump); require("bivcurdata.jl");

mydata1=readdlm("ACTG384_com.txt",'\t';skipstart=1);

x = map(x->float(x), mydata1[:,[2,5]]);
dlt = map(x->float(x), mydata1[:,[4,7]]);

z13F = map(x->float(x), (mydata1[:,8].=="A")|(mydata1[:,8].=="C"));
z24F = map(x->float(x), (mydata1[:,8].=="B")|(mydata1[:,8].=="D"));

zF = [z13F z24F z13F z24F]
 
mydata=(x,dlt,zF);

(estbet, a1, a2, cenp, eti)= bivcur(mydata; tol0=1.0e-10)

#### function to report the bootstraps

function simreport(res, nproc::Int64)

  mle_coef, emcheck=res[1];
#  inx=map(Bool, emcheck)
#  mle_bt = mle_bt[inx,:]
  mle_bt=mle_coef
  
  for j in 2:nproc
    mle_coef, emcheck_a=res[j];
#    inx=map(Bool, emcheck_a)
#    mle_bt=vcat(mle_bt, mle_coef[inx,:]);

    mle_bt=vcat(mle_bt, mle_coef)
  end;  

  println(mapslices(std, mle_bt,1))
  
  mle_bt
end;

### bootstrap 

@time res=parall_bivcur(nump, nump*50, mydata, rseed=20);
mys=simreport(res, nump);

#### Output the baseline function estimates

writedlm("estbaseline.txt", [mapslices(sort, x, 1) a1 a2])

