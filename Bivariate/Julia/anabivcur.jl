nump=20; addprocs(nump); require("bivcurdata.jl");


mydata1=readdlm("aCNVDupDel4.txt",'\t');
x = mydata1[:;1:2];
dlt = mydata1[:;3:4];
z = mydata1[:;5:8];
mydata=(x,dlt,z);

(estbet, a1, a2, c0, c1)= bivcur(mydata)

writedlm("estbaseline.txt", [x a1 a2])

#### report function

function simreport(res, nproc::Int64)

  mle_bt, emcheck=res[1];
  mle_bt = mle_bt[bool(emcheck),:]
  
  for j in 2:nproc
    mle_a,  emcheck_a=res[j];
    mle_bt=vcat(mle_bt, mle_a[bool(emcheck_a),:]);
  end;  

  println(mapslices(std, mle_bt,1))
  
  mle_bt
end;

### bootstrap 

@time res=parall_bivcur(nump, nump*30, mydata);
mys=simreport(res, nump);



### Results
#  0.074966
#  0.466731
# -0.740628
#  0.0819658
# -0.696661
### [0.6631711411270791 0.8161601111849739 1.022549660449663 0.9362156249926319 0.2245179253691393]

