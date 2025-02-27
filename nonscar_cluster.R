source('_optims_wrapper.R')
# rm(list = ls())
library(cluster)
library(caret)
library(stringr)
source('datasets_prep.R')
library(dplyr)
source('_helpers.R')
source('scoring_methods.R')
library(pROC)
library(MLmetrics)

datasets <- load_datasets(more_then_n_unique_values = 5)
list2env(datasets, envir = .GlobalEnv)

dsets<-
   ls()[sapply(ls(), function(x) class(get(x))) == 'data.frame']

dsets <- c("adult", "alon_ds", "artif", "bank_marketing", "banknote", 
   "breastc", "credit_a", "credit_g", "dhfr", "diabetes", "lymphoma", 
   "prostate", "spambase", "wdbc", "wine_quality", "kdd_data")

for (i in dsets) {
    
  
    current_frame <- get(i)
    
    if (ncol(current_frame) > nrow(current_frame)) {
      next
    }
    
    experi_max1 <- ifelse(ncol(current_frame)<100,100,10)
    experi_max2 <- ifelse(nrow(current_frame)<10000,100,10)
    
    experi_max <- min(experi_max1,experi_max2)
    
    for (num in 1:experi_max) {
      set.seed(num)
      
      for (c_calc in c(0.3,0.5,0.8)) {

        current_frame <- non_scar_labelling.mvc(current_frame,c_calc, n_vars = 1)
        index <- createDataPartition(current_frame$S, p = 0.7, list = F)
        train <- current_frame[index,]
        test <- current_frame[-index,]
        
        x_train <- as.matrix(train[,!colnames(current_frame) %in% c('Y','S')])
        y_train <- train$Y
        s_train <- train$S 
        
        x_test <- as.matrix(test[,!colnames(current_frame) %in% c('Y','S')])
        y_test <- test$Y
        
        
        # Randomly select q observations from class S=1 and add these observations to class S=0.
        # Repeat the entire procedure R times. Average the coefficients from each repetition.
        
        
        for (q in c(0.25, 0.5, 1)) {
          
          {my_glm_coef <- find_glm_coef(train,pecked_part = q, n_of_pecking = 5 ,lasso = F)} %>% system.time(.) -> time
          clust_time <- time["elapsed"] %>% unname
          {my_glm_strict_lasso_coef <- find_glm_coef(train,pecked_part = q, n_of_pecking = 5 ,lasso = T, strict = T)} %>% system.time(.) -> time
          strict_lassclust_time <- time["elapsed"] %>% unname
          {my_glm_nonstrict_lasso_coef <- find_glm_coef(train,pecked_part = q, n_of_pecking = 5 ,lasso = T, strict = F)} %>% system.time(.) -> time
          non_strict_lassclust_time <- time["elapsed"] %>% unname
          
          
          c_calculated <- sum(current_frame$S)/sum(current_frame$Y)
          
          {glm_naive <- my_glm(s_train ~ .,  data.frame(x_train), family = 'binomial')} %>% system.time(.) -> time
          naive_time <- time["elapsed"] %>% unname
          Y_pred_naive <- predict(glm_naive, data.frame(x_test), type = 'response')
          
          glm_clust <- glm_naive
          
          glm_clust$coefficients <- my_glm_coef 
          Y_pred_clust <- predict(glm_clust, data.frame(x_test), type = 'response')
          
          glm_clust$coefficients <- my_glm_strict_lasso_coef 
          Y_pred_lass_clust <- predict(glm_clust, data.frame(x_test), type = 'response')
          
          glm_clust$coefficients <- my_glm_nonstrict_lasso_coef 
          Y_pred_nonstrict_lass_clust <- predict(glm_clust, data.frame(x_test), type = 'response')
          
          
          
          {model_lj <- lassojoint(x_train = x_train, y_train = s_train, x_test = x_test,nfolds = 10, lambda = 'lambda.min')} %>% system.time(.) -> time
          lassojoint_time <- time["elapsed"] %>% unname
          
          Y_pred_lassojoint <- model_lj$scores
          
          
          y_test <- as.numeric(y_test)
          Y_pred_naive <- as.numeric(Y_pred_naive)
          Y_pred_lassojoint <- as.numeric(Y_pred_lassojoint)
          
          
          Y_pred_naive_label <- ifelse(Y_pred_naive<0.5,0,1)
          Y_pred_clust_label <- ifelse(Y_pred_clust<0.5,0,1)
          Y_pred_lass_clust_label  <- ifelse(Y_pred_lass_clust<0.5,0,1)
          Y_pred_nonstrict_lass_clust_label <- ifelse(Y_pred_nonstrict_lass_clust<0.5,0,1)
          Y_pred_lassojoint_label <- ifelse(Y_pred_lassojoint<0.5,0,1)
          
          
          res_row <- 
            data.frame(
              ts = Sys.time(),
              df = i,
              experi_num = num,
              q = q,
              calc_c = round(c_calculated,2),
              
              
              naive = suppressMessages(round(auc(y_test, Y_pred_naive),3)) ,
              clust = suppressMessages(round(auc(y_test, Y_pred_clust),3)) ,
              strict_lassclust = suppressMessages(round(auc(y_test, Y_pred_lass_clust),3)) ,
              non_strict_lassclust = suppressMessages(round(auc(y_test, Y_pred_nonstrict_lass_clust),3)) ,
              lassojoint = suppressMessages(round(auc(y_test, Y_pred_lassojoint),3)),
              

              
              naive_acc = suppressMessages(round(Accuracy(Y_pred_naive_label, y_test),3)) ,
              clust_acc = suppressMessages(round(Accuracy(Y_pred_clust_label, y_test),3)) ,
              strict_lassclust_acc = suppressMessages(round(Accuracy(Y_pred_lass_clust_label, y_test),3)) ,
              non_strict_lassclust_acc = suppressMessages(round(Accuracy(Y_pred_nonstrict_lass_clust_label, y_test),3)) ,
              lassojoint_acc = suppressMessages(round(Accuracy(Y_pred_lassojoint_label, y_test),3)),
              
              
              
              naive_f1 = suppressMessages(round(F1_Score(Y_pred_naive_label, y_test),3)) ,
              clust_f1 = suppressMessages(round(F1_Score(Y_pred_clust_label, y_test),3)) ,
              strict_lassclust_f1 = suppressMessages(round(F1_Score(Y_pred_lass_clust_label, y_test),3)) ,
              non_strict_lassclust_f1 = suppressMessages(round(F1_Score(Y_pred_nonstrict_lass_clust_label, y_test),3)) ,
              lassojoint_f1 = suppressMessages(round(F1_Score(Y_pred_lassojoint_label, y_test),3)),
              
              naive_time = naive_time,
              
              clust_time = clust_time,
              
              strict_lassclust_time = strict_lassclust_time,
              
              non_strict_lassclust_time = non_strict_lassclust_time,
              
              lassojoint_time = lassojoint_time
              
              
              
            )
          
          if (!exists('res_df')) {
            
            res_df <- res_row
          } else {
            
            res_df <- rbind.data.frame(res_df,res_row)
          }
          
          
          print(res_row)
          
          
        }
        
      }
      

    }
    
    short_commit_hash <- system("git rev-parse --short HEAD", intern = TRUE)
    
    output_filename <- paste0('res_clusters_',short_commit_hash,'.RDS')
    
    saveRDS(object = res_df, file = output_filename)
}









