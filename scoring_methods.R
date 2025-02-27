my_glm <- function(formula, family = gaussian, data, ...) {
  ignore_specific_warnings <- function(w) {
    # Lista ostrzeżeń do ignorowania
    warnings_to_ignore <- c(
      "glm.fit: fitted probabilities numerically 0 or 1 occurred",
      "glm.fit: algorithm did not converge"
    )
    
    # Sprawdzanie, czy wiadomość ostrzeżenia pasuje do którejś z listy
    if (any(sapply(warnings_to_ignore, function(pattern) grepl(pattern, w$message)))) {
      # Ignoruj specyficzne ostrzeżenie
      invokeRestart("muffleWarning")
    } else {
      # Wyświetl inne ostrzeżenia
      warning(w)
    }
  }
  
  # Użycie withCallingHandlers do obsługi ostrzeżeń
  withCallingHandlers({
    model <- glm(formula = formula, family = family, data = data, ...)
  }, warning = ignore_specific_warnings)
  
  return(model)
}



#Logogistic
sigma = function(s) {
  res = exp(s) / (1 + exp(s))
  return(res)
}

logLike = function(par,x,y){
  beta0 = par[1]
  beta1 = par[-1]
  res = -sum(  y*log(sigma(x%*%beta1+beta0))+(1-y)*log(1-sigma(x%*%beta1+beta0))   )
  return(res)
}

gr = function(par,x,y){
  beta0 = par[1]
  beta1 = par[-1]
  
  var1 = sigma(x%*%beta1+beta0)*(1-sigma(x%*%beta1+beta0))
  sigma1 = sigma(x%*%beta1+beta0)
  
  a= var1*( (-y+sigma1)/(sigma1*(1-sigma1)) )
  
  res = t(cbind(1,x))%*%a
  return(res)
}

logistic_fit = function(x,y){
  
  optim1 = optim(par=rep(0,ncol(x)+1),fn=logLike,gr=gr,x=x,y=y,method="BFGS")
  par = optim1$par
  return(list(par=par))

}

logistic <- function(x_train, y_train, x_test){
  
  beta <- logistic_fit(x_train, y_train)$par
  betas1=beta[-1]
  betas0=beta[1]
  
  est=exp(x_test%*%betas1 + betas0)/(1+exp(x_test%*%betas1 + betas0))
  
  est
  
}

#Logistic joint
logLike_joint = function(par,x,y){
  beta0 = par[1]
  beta1 = par[-c(1,length(par))]
  c1 = par[length(par)]
  term1 = c1*sigma(x%*%beta1+beta0)
  term2 = 1-c1*sigma(x%*%beta1+beta0)
  
  term1 = ifelse(term1<0,0,term1)
  term2 = ifelse(term2<0,0,term2)
  
  
  res = -sum(  y*log(term1)+(1-y)*log(term2)   )
  return(res)
}
gr_joint = function(par,x,y){
  beta0 = par[1]
  beta1 = par[-c(1,length(par))]
  
  c1 = par[length(par)]
  
  var1 = sigma(x%*%beta1+beta0)*(1-sigma(x%*%beta1+beta0))
  sigma1 = sigma(x%*%beta1+beta0)
  
  a= var1*( (-y+c1*sigma1)/(sigma1*(1-c1*sigma1)) )
  res = t(cbind(1,x))%*%a
  
  gr_c1 = sum(-y/c1 + (1-y)*sigma1/(1-c1*sigma1))
  res = c(res,gr_c1)
  
  return(res)
}


logistic_fit_joint = function(x,y,optim_method = "BFGS"){
  
  logLike_joint_xy <- function(par){logLike_joint(x = x,y = y,par = par)}
  gr_joint_xy <- function(par){gr_joint(x = x,y = y,par = par)}
  
  
  optim1 = my_optims(par=c(rep(0,ncol(x)+1),0.5),fn=logLike_joint_xy,gr=gr_joint_xy,method=optim_method)
  par = optim1$par
  return(list(par=par))
}

logistic_joint <- function(x_train, y_train, x_test, optim_method = "BFGS"){
  
  par_joint = logistic_fit_joint(x_train, y_train, optim_method = optim_method)$par
  beta_joint = par_joint[-length(par_joint)]
  c_joint = par_joint[length(par_joint)]
  
  betas1=beta_joint[-1]
  betas0=beta_joint[1]
  
  est=exp(x_test%*%betas1 + betas0)/(1+exp(x_test%*%betas1 + betas0))
  
  est[is.nan(est)] <- 0
  
  est <- ifelse(est>0.5,1,0)
  return(est)
  
}

#Logistic weighted
logLike_w = function(par,x,y,c){
  beta0 = par[1]
  beta1 = par[-1]
  res = -sum(  (y/c)*log(sigma(x%*%beta1+beta0))+(1-y/c)*log(1-sigma(x%*%beta1+beta0))   )
  return(res)
}
gr_w = function(par,x,y,c){
  beta0 = par[1]
  beta1 = par[-1]
  
  var1 = sigma(x%*%beta1+beta0)*(1-sigma(x%*%beta1+beta0))
  sigma1 = sigma(x%*%beta1+beta0)
  
  a= var1*( (-y/c+sigma1)/(sigma1*(1-sigma1)) )
  
  res = t(cbind(1,x))%*%a
  return(res)
}

logistic_fit_w = function(x,y,c){
  
  if(c==0) c=c+0.001
  
  optim1 = optim(par=rep(0,ncol(x)+1),fn=logLike_w,gr=gr_w,x=x,y=y,c=c,method="BFGS")
  par = optim1$par
  return(list(par=par))
}

logistic_w <- function(x_train, y_train, x_test) {
  par_joint = logistic_fit_joint(x_train, y_train)$par
  beta_joint = par_joint[-length(par_joint)]
  c_joint = par_joint[length(par_joint)]
  
  beta_weighted = logistic_fit_w(x_train, y_train, c_joint)$par
  
  betas1 = beta_weighted[-1]
  betas0 = beta_weighted[1]
  
  est = exp(x_test %*% betas1 + betas0) / (1 + exp(x_test %*% betas1 + betas0))
  
  est
  
}
#Logistic CDMM
logLike_beta = function(par,x,y,beta1,beta0){
  res = sum(  y*log(par*sigma(x%*%beta1+beta0))+(1-y)*log(1-par*sigma(x%*%beta1+beta0))   )
  return(res)
}
gr_beta = function(par,x,y,beta1,beta0){
  
  sigma1 = sigma(x%*%beta1+beta0)
  res = sum(y/par - (1-y)*sigma1/(1-par*sigma1)) #?????
  return(res)
}


logistic_fit_beta = function(x,y,beta1,beta0){
  
  optim1 = optim(par=0.5,fn=logLike_beta,gr=gr_beta,x=x,y=y,beta1=beta1,beta0=beta0,method="BFGS",control=list(fnscale=-1))
  par = optim1$par
  return(list(par=par))
}
grad_logLike = function(par,x,y,c){
  beta0 = par[1]
  beta1 = par[-1]
  
  var1 = sigma(x%*%beta1+beta0)*(1-sigma(x%*%beta1+beta0))
  sigma1 = sigma(x%*%beta1+beta0)
  
  a= var1*( (y-c*sigma1)/(sigma1*(1-c*sigma1)) )
  
  res = t(cbind(1,x))%*%a
  return(res)
}


optimize_b_MM = function(x,y,c,max_iter=200){
  x1 = cbind(1,x)
  p=ncol(x1)
  par_old = rep(0,p)
  for(iter in 1:max_iter){
    P = (0.25)*t(x1)%*%x1
    q = -grad_logLike(par_old,x,y,c)
    solve1 = osqp::solve_osqp(P=P, q=q,pars = osqp::osqpSettings(verbose=FALSE))
    par_new = solve1$x + par_old
    if(is.na(sum(par_new))) break
    if(max(abs(par_new-par_old))<0.001) break
    par_old = par_new
  }
  par = par_old
  return(par)
  
}


logistic_fit_cdmm = function(x,s,max_iter=100){
  
  p = ncol(x)+1    
  n = nrow(x)
  b_old = numeric(p)  
  c_old = 0.5
  x1 = cbind(1,x)
  
  for(iter in 1:max_iter){
    #1 Compute c using b_old:
    c_new=  logistic_fit_beta(x,s,beta1=b_old[-1],beta0=b_old[1])$par
    
    if(c_new>1) c_new=0.99
    if(c_new<0) c_new=0.01
    
    #2 Compute b using c (using MM algorithm):
    b_new = optimize_b_MM(x,s,c_new,max_iter=50)
    if(max(abs(c_new-c_old))<0.001) break
    
    c_old = c_new
    b_old = b_new
    
  }
  res = list(c=c_old,beta=b_old)
  return(res)
  
}

logistic_cdmm <- function(x_train, y_train, x_test){
  
  par_cdmm =  suppressWarnings(logistic_fit_cdmm(x_train, y_train))
  beta_cdmm = par_cdmm$beta
  c_cdmm = par_cdmm$c
  
  betas1=beta_cdmm[-1]
  betas0=beta_cdmm[1]
  
  est=exp(x_test%*%betas1 + betas0)/(1+exp(x_test%*%betas1 + betas0))
  
  est
  
}



#Lassojoint
library(glmnet)
lassojoint <- function(x_train, y_train, x_test, nfolds = NULL, lambda = "(log(p)/n)^(1/3)", delta_to_lambda = 0.5){
  
  n <- nrow(x_train) 
  p <- ncol(x_train)
  
  x <- x_train
  s <- y_train
  
  if (is.null(nfolds)) {
    
    lambda = eval(parse(text = lambda))
    obj3<-glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", lambda = lambda)
    
  } else if (lambda == "lambda.min") {
    
    obj3 <-cv.glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=nfolds)
    lambda <- obj3$lambda.min
    
  } else if (lambda == "lambda.1se"){
    
    obj3 <-cv.glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=nfolds)
    lambda <- obj3$lambda.1se
    
  }
  
  delta = lambda * delta_to_lambda
  
  betasx1<-coefficients(obj3,s=lambda)
  betasy1<-betasx1[-1]
  nosnik1<-which(abs(betasy1)>delta)
  x11<-as.matrix(x[,nosnik1])
  s11<-s
  
  par_joint1 = logistic_fit_joint(x11, s11)$par
  beta_joint1 = par_joint1[-length(par_joint1)]
  c_joint1 = par_joint1[length(par_joint1)]
  
  betas13=beta_joint1[-1]
  betas03=beta_joint1[1]
  
  est <- list()
  
  est$nosnik <- nosnik1
  est$nosnik_dl <- length(nosnik1)
  est$scores <- exp(as.matrix(x_test[,nosnik1])%*%betas13 + betas03) / (1 + exp(as.matrix(x_test[,nosnik1])%*%betas13 + betas03))
  
  est
  
}


lassoCDMM <- function(x_train, y_train, x_test, nfolds = NULL, lambda = "(log(p)/n)^(1/3)", delta_to_lambda = 0.5){
  
  n <- nrow(x_train) 
  p <- ncol(x_train)
  
  x <- x_train
  s <- y_train
  
  if (is.null(nfolds)) {
    
    lambda = eval(parse(text = lambda))
    obj3<-glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", lambda = lambda)
    
  } else if (lambda == "lambda.min") {
    
    obj3 <-cv.glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=nfolds)
    lambda <- obj3$lambda.min
    
  } else if (lambda == "lambda.1se"){
    
    obj3 <-cv.glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=nfolds)
    lambda <- obj3$lambda.1se
    
  }
  
  delta = lambda * delta_to_lambda
  
  betasx1<-coefficients(obj3,s=lambda)
  
  betasy1<-betasx1[-1]
  nosnik1<-which(abs(betasy1)>delta)
  x11<-as.matrix(x[,nosnik1])
  s11<-s
  xtt <- as.matrix(x_test[,nosnik1])


  est <- list()
  
  est$nosnik <- nosnik1
  est$nosnik_dl <- length(nosnik1)
  est$scores <- logistic_cdmm(x_train = x11, y_train = s11, x_test = xtt)
  
  est
  
  
}



library(AdaSampling)

adasampling_score <- function(x_train, y_train, x_test, classifier = 'svm') {
  
  Ps <- which(y_train == 1)
  Ns <- which(y_train == 0)

  Ps <- as.character(Ps)
  Ns <- as.character(Ns)

  x_train <- as.data.frame(x_train)
  x_test <- as.data.frame(x_test)
  
  model<-adaSample(Ps = Ps, Ns = Ns, train.mat=x_train, test.mat=x_test,classifier = classifier)
  est= as.vector(model[,"P"])
  
  est
  
}
