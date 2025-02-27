library(nloptr)
library(dplyr)
library(R.utils)
library(stringr)

nloptr_optim <- function(par, fn, gr, lower = -9e10, upper = 9e10, method) {
  
  a <- tryCatch(
    { 
      nloptr(
        x0 = par
        , eval_f = fn
        , eval_grad_f = gr
        , opts = list("algorithm"= method, "maxtime" = 30, "xtol_rel"=1.0e-8, "maxeval" = 10000)
        , lb = rep(lower,length(par))
        , ub = rep(upper,length(par))
      )
      
    }
    , error = function(e) {list(solution = par, message = 'blad')}
  )
  
  b <- list(method = method, par = a$solution, message = a$message )
  
  return(b)
}

my_optims <- function(par, fn, gr, lower = -999, upper = 999, method, ...){
  
  optim.methods <- eval(formals(optim)$method)
  optim.methods <- setdiff(optim.methods,c('L-BFGS-B', "Brent"))
  
  nloptr.methods <- nloptr.get.default.options() %>% filter(name == 'algorithm') %>% select(possible_values) %>% pull %>% str_split_1(., ', ') 
  nloptr.methods <- setdiff(nloptr.methods,"NLOPT_LN_NEWUOA_BOUND")
  
  if (method %in% c("SANN","NLOPT_LN_PRAXIS")) {gr = NULL}
  
  if (method %in% optim.methods) {
    control = NULL

    a <- tryCatch(
      {optim(par = par, fn = fn, gr = gr, method = method, control = list(maxit = 10000))}
      , error = function(e) {list(par = par, message = 'blad')}
    ) %>% suppressWarnings()
    
    a$message <- 'ok'
  } else if (method %in% nloptr.methods) {
    a <- nloptr_optim(par = par, fn = fn, gr = gr, lower = lower, upper = upper, method = method) 
    
  } else {
    stop('method unknown') 
  }
  
  
  value_counts <- table(a$par)
  
  # Wybór najczęściej występującej wartości
  most_common_value <- names(sort(value_counts, decreasing = TRUE)[1])
  
  # Liczba wystąpień najczęstszej wartości
  count_most_common_value <- value_counts[most_common_value]
  
  frac_most_common <- count_most_common_value/length(a$par)
  
  frac_most_common <- ifelse(is.numeric(frac_most_common),frac_most_common,1)
  
  if (identical(par,a$par) | max(is.nan(a$par)) == 1 | frac_most_common > 0.9) {
    a$message = 'blad'
  }
  
  
  b <- list(method = method, par = a$par, message = a$message )
  
  return(b)
  
}

optim.methods <- eval(formals(optim)$method)
optim.methods <- setdiff(optim.methods,c('L-BFGS-B', "Brent"))

nloptr.methods <- nloptr.get.default.options() %>% filter(name == 'algorithm') %>% select(possible_values) %>% pull %>% str_split_1(., ', ') 
nloptr.methods <- setdiff(nloptr.methods,"NLOPT_LN_NEWUOA_BOUND")

my_optims.methods <- c(optim.methods,nloptr.methods)

return_useful_optims <- function(par, fn, gr, x, y, optims_vec){
  
  result <- c()
  
  # logLike_xy <- function(par){logLike(x = x,y = y,par)}
  # gr_xy <- function(par){gr(par,x = x,y = y)}
  
  logLike_joint_xy <- function(par){logLike_joint(par = par, x = x, y = y)}
  gr_joint_xy <- function(par){gr_joint(par = par, x = x, y = y)}
  
  for (i in optims_vec){
    a <- my_optims(par = par, fn = logLike_joint_xy, gr = gr_joint_xy, method = i)
    print(a)
    if (a$message != 'blad') {
      result <- c(result,i)
    }
  }
  
  return(result)
}
