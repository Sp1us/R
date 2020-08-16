fpr=c()
fnr=c()
error=c()

fpr_svm=c()
fnr_svm=c()
error_svm=c()
for(k in 1:10){
  n = 500 # number of sample
  p = 2 # number of variables
  sample_sigma = 1 # sd 
  meanpos = 3 # mean of positive label
  meanneg = 5 # mean of negative label
  npos = round(n/3) # number of positive
  nneg = n-npos # number of negative
  #set.seed(1234)
  xpos = matrix(rnorm(npos*p,mean=meanpos,sd=sample_sigma),npos,p)
  xneg = matrix(rnorm(nneg*p,mean=meanneg,sd=sample_sigma),nneg,p)

  x = rbind(xpos,xneg)
  y = matrix(c(rep(1,npos),rep(-1,nneg)))#label
  mode(y)
  data = cbind(x,y)
  data
  plot(x,col=ifelse(y>0,1,2))
  x = as.matrix(x)

  colnames(x)= letters[1:dim(x)[2]]
  ## normalization
  p = dim(x)[2]
  for (i in 1:p){
    xi0 = min(x[, i])
    xi1 = max(x[, i])
    si = sd(x[, i])
    x[ , i] = (x[,i]-xi0)/(xi1-xi0)
  }
  
  ## create training and test sets
  pind = which(y==1)
  nind = which(y==-1)
  np = length(pind)
  nn = length(nind)
  
  
  train_np = npos/3
  train_nn = nneg/3
  train_ps = sample(np, train_np)
  train_ns = sample(nn, train_nn)
  xxp = x[pind[train_ps], ]
  xxn = x[nind[train_ns], ]
  yyp = rep(1, train_np)
  yyn = rep(-1, train_nn)
  xx = rbind(xxp, xxn)
  yy = c(yyp, yyn)
  xx=as.matrix(xx)
  
  test_pind = setdiff(pind, pind[train_ps])
  test_nind = setdiff(nind, nind[train_ns])
  testx = x[c(test_pind, test_nind), ]
  test_np = length(test_pind)
  test_nn = length(test_nind)
  testy = c(rep(1, test_np), rep(-1, test_nn))
  
  prsvm = function(x,y,sigma){
    w = prsvmw(x,y,sigma)
    b = prsvmb(x,y,w)
    return(list(w,b))
  }
  
  prsvmloss = function(w, x, posind, negind, sigma){
    npos = length(posind)
    nneg = length(negind)
    loss = 0
    wx = x %*% w
    for (i in posind) {
      for (j in negind) {
        tmp = max(0, 1-(wx[i]-wx[j]))
        if (tmp!=0){
          loss = loss + sigma^2*(1-exp(-tmp^2/sigma^2))
        }
      } # end j
    } # end i
    loss = loss/(npos*nneg)  
    return(loss)
  }
  
  prsvmlosspd = function(w, x, posind, negind, sigma){
    p = length(w)
    npos = length(posind)
    nneg = length(negind)
    pd = rep(0, p)
    wx = x %*% w
    for (i in posind) {
      for (j in negind) {
        tmp = max(0, 1-(wx[i]-wx[j]))
        if (tmp!=0){
          pd = pd - 2*exp(-tmp^2/sigma^2)*tmp*(x[i,]-x[j,])
        }
      } # end j
    } # end i
    pd = pd/(npos*nneg)
    return(pd)
  }
  
  
  prsvmw = function(x, y, sigma) {
    d = dim(x)
    p = d[2]
    n = d[1]
    posind = which(y==1)
    npos = length(posind)
    negind = which(y==-1)
    nneg = length(negind)
    iter = 0
    maxiter = 1000
    w = solve(cov(x), cov(x, y))
    loss = prsvmloss(w, x, posind, negind, sigma)
    stop = FALSE
    while (!stop) {
      pd = prsvmlosspd(w, x, posind, negind, sigma)
      lossold = loss
      wold = w
      wnew = w - pd
      lossnew = prsvmloss(wnew, x, posind, negind, sigma)
      s = 1
      fs = FALSE
      if (lossnew < lossold){
        while (!fs){
          s = 2*s
          lossold = lossnew
          wold = wnew
          wnew = w - s*pd
          lossnew = prsvmloss(wnew, x, posind, negind, sigma)
          if (lossnew >= lossold){
            fs = TRUE
            w = wold
            dloss = (loss - lossold)/loss
            loss = lossold
          } 
        } # end while
      } else {
        while (!fs){
          print(s)
          wnew = w - s*pd
          wxnew = x %*% wnew
          lossnew = prsvmloss(wnew, x, posind, negind, sigma)
          if (lossnew <= loss){
            fs = TRUE
            w = wnew
            dloss = (loss - lossnew)/loss
            loss = lossnew
          } # end if
        } # end while
      } # end else
      
      iter = iter+1
      if (norm(as.matrix(pd))<0.00001||dloss<0.01||iter>maxiter||loss==0) {
        stop = TRUE
      }
      print(iter)
      print(loss)
    } #stop
    
    return(w)
  }
  
  prsvmb = function(x,y,w){
    posind = which(y==1)
    np = length(posind)
    negind = which(y==-1)
    nn = length(negind)
    pred = x%*%w
    p = sort(pred)
    n = length(p)
    bc = (p[1:(n-1)]+p[2:n])/2
    error = rep(0, n-1)
    for (i in 1:(n-1)){
      fpr = length(which(pred[posind]<bc[i]))/np
      fnr = length(which(pred[negind]>bc[i]))/nn
      error[i] = fpr+fnr
    }
    besti = which.min(error)
    b = bc[besti]
    return(b)
  }
  
  #source("prsvm.R")
  w = prsvmw(xx, yy, 1)
  b = prsvmb(xx, yy, w)
  
  predontest = testx %*% w - b
  fpr[k] = length(which(predontest[1:test_np] < 0))/test_np
  fnr[k] = length(which(predontest[test_np + 1:test_nn] >0 ))/test_nn
  error[k] = (fpr[k] + fnr[k])/2
  
  
  
  library(caret)
  svm = train(xx, as.factor(yy), method="svmLinear")
  predicted = predict(svm, testx)
  
  fpr_svm[k] = length(which(predicted[1:test_np] != testy[1:test_np]))/test_np
  fnr_svm[k] = length(which(predicted[test_np + 1:test_nn] != testy[test_np + 1:test_nn]))/test_nn
  error_svm[k] = (fpr_svm[k] + fnr_svm[k])/2
  
}

fpr=sum(fpr)/10
fnr=sum(fnr)/10
error=sum(error)/10

fpr_svm=sum(fpr_svm)/10
fnr_svm=sum(fnr_svm)/10
error_svm=sum(error_svm)/10


fpr
fnr
error

fpr_svm
fnr_svm
error_svm
