append_results <- function(results_frame, dataset, method, c, experiment_num, score, Y_real){
  
  my_result_frame <- results_frame
  
  current_result <- data.frame(
    dataset = i
    , method = method
    , c = c
    , experiment_num = num
    , score = score
    , Y_real = y_test
  )
  
  my_result_frame <- rbind.data.frame(
    my_result_frame
    , current_result
  )
  
  my_result_frame
}

sigma = function(s) {
  res = exp(s) / (1 + exp(s))
  return(res)
}

non_scar_labelling <- function (df) {
  
  current_frame <- df
  
  x <- current_frame %>% 
    select(-Y) %>% 
    select(-any_of("S")) %>% 
    as.matrix()
  
  p <- ncol(x)
  n <- nrow(x)
  
  repeat {
    set.seed(Sys.time()) # Seed based on current time with seconds (explicit)
    
    g <- runif(1,-20,20) # Parameter 'g' complies with scenario 2 in 10.3233/FAIA230342
    
    gamma <- rep(p^(-1/2)  * g, times = p) 
    
    rbinom_vec <- Vectorize(FUN = rbinom, vectorize.args = 'prob')
    
    current_frame$S <- as.numeric(current_frame$Y * rbinom_vec(1,1,sigma(x%*%gamma)))
    c_calculated <- sum(current_frame$S)/sum(current_frame$Y)
    if(c_calculated > 0.1 & c_calculated < 0.9 & sum(current_frame$S) > 10)  # If 'c' is too small, it will cause difficulties with methods
      break
  }
  
  return(current_frame)
  
}

non_scar_labelling.mvc <- function(df, target_c_calc, n_vars = 1) {
  
  # Remove columns Y and S
  df_filtered <- df[, !names(df) %in% c("Y", "S")]
  
  # Check if n_vars does not exceed the number of columns in df_filtered
  if (n_vars >= ncol(df_filtered)) {
    print(paste("n_vars exceeds the number of available columns. Using", ncol(df_filtered) - 1, "variables instead."))
    n_vars <- ncol(df_filtered) - 1
  }
  
  # Calculate variance for each column
  variances <- sapply(df_filtered, var)
  
  # Sort variances in descending order
  sorted_variances <- sort(variances, decreasing = TRUE)
  
  # Find the names of columns with the highest variances (number depends on n_vars)
  top_variable_columns <- names(sorted_variances)[1:n_vars]
  
  # Create a temporary dataframe that contains the sum of values for selected columns
  df_temp <- df %>%
    mutate(rn = row_number(rowSums(across(all_of(top_variable_columns))),
                           S02 = ifelse(rn/nrow(df) < 0.2, 1, 0) * Y,
                           S05 = ifelse(rn/nrow(df) < 0.5, 1, 0) * Y,
                           S08 = ifelse(rn/nrow(df) < 0.8, 1, 0) * Y
    )
    
    # Calculate c_calc for three levels
    c_calc02 <- sum(df_temp$S02)/sum(df_temp$Y)
    c_calc05 <- sum(df_temp$S05)/sum(df_temp$Y)
    c_calc08 <- sum(df_temp$S08)/sum(df_temp$Y)
    
    c_calc <- c(c_calc02, c_calc05, c_calc08)
    
    # Fit a linear model for c_calc
    coef <- coef(glm(c_calc ~ c(0.2, 0.5, 0.8))) %>% as.numeric()
    
    # Calculate the proportion for target_c_calc
    rn_frac <- (target_c_calc - coef[1])/coef[2]
    
    # Create the final dataframe with the column S
    df1 <- df %>%
      mutate(rn = row_number(rowSums(across(all_of(top_variable_columns))),
                             S = ifelse(rn/nrow(df) < rn_frac, 1, 0) * Y
      ) %>%
        select(-rn)
      
      # Return the dataframe with the new column S
      return(df1)
}

scar_labelling <- function(df, target_c_calc){
  
  current_frame <- df
  current_frame$Y <- as.numeric(current_frame$Y)
  current_frame$S <- current_frame$Y * rbinom(nrow(current_frame),1,target_c_calc)
  current_frame <- as.data.frame(sapply(current_frame, as.numeric))
  
  return(current_frame)
}

create_correlation_map <- function(data_list, y_var, threshold, output_file) {
  pdf(output_file, width = 10, height = 8)  # Open a PDF file
  
  for (i in seq_along(data_list)) {
    data <- data_list[[i]]
    data_name <- names(data_list)[i]
    
    # Remove the Y column
    data <- data[, !names(data) %in% y_var]
    
    # Select only numeric columns
    numeric_data <- data[, sapply(data, is.numeric), drop = FALSE]
    
    # Check if numeric_data is not empty and if there are at least two numeric columns
    if (is.null(numeric_data)) {
      next
    }
    
    if (ncol(numeric_data) < 2) {
      next
    }
    
    if (ncol(numeric_data) > 1000) {
      next
    }
    
    # Calculate the correlation matrix
    correlation_matrix <- cor(numeric_data, use = "complete.obs")
    
    # Transform the correlation matrix into a long format
    melted_correlation_matrix <- melt(correlation_matrix)
    
    # Add a column to indicate high correlation
    melted_correlation_matrix$high_correlation <- abs(melted_correlation_matrix$value) > threshold
    
    # Create the correlation map
    p <- ggplot(data = melted_correlation_matrix, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = ifelse(high_correlation, "X", "")), color = "black", size = 5) +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                           midpoint = 0, limit = c(-1, 1), space = "Lab", 
                           name = "Pearson Correlation Coefficient") +
      theme_minimal() + 
      theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                       size = 12, hjust = 1),
            axis.title.x = element_blank(),
            axis.title.y = element_blank()) +
      coord_fixed() +
      ggtitle(data_name)
    
    print(p)  # Save the plot on a new page in the PDF file
  }
  
  dev.off()  # Close the PDF file
}

find_glm_coef <- function(train_df, pecked_part = 0.25, n_of_pecking = 5, lasso = F, strict = F, keep_original_1 = TRUE) {
  
  if (pecked_part == 1) {
    n_of_pecking <- 1
    print("For pecked_part == 1 scenario, pecking repetition isn't needed, so automatically set n_of_pecking = 1 to optimize")
  }
  
  for (k in 1:n_of_pecking) {
    set.seed(k)
    q <- pecked_part
    
    S1_ind <- which(train_df$S == 1)
    S0_ind <- which(train_df$S == 0)
    
    S1_ind_sample <- sample(S1_ind, length(S1_ind) * q)
    
    # The set S=1 without q% observations is denoted as C1(-q)
    C1_minus_q <- train_df[setdiff(S1_ind,S1_ind_sample),] 
    
    # The set S=0 with added q% observations from class S=1 is denoted as C0(+q)
    C0_plus_q <- train_df[c(S0_ind,S1_ind_sample),]
    
    # Apply the 2-means algorithm to the set C0(+q).
    C0_plus_q <-
      C0_plus_q %>% 
      mutate_if( ~ length(unique(.)) == 2, as.factor) %>%  # Columns in the dataframe C0_plus_q that have exactly two unique values will be converted to factor type.
      mutate(S = as.numeric(S) - 1, Y = as.numeric(Y) - 1)
    
    # Example of k-means
    mixedClusters <- kmeans(select(C0_plus_q, -c("Y", "S")), centers = 2, nstart = 5)
    table(mixedClusters$cluster)
    C0_plus_q <- as.data.frame(C0_plus_q)
    C0_plus_q$cluster <- as.numeric(mixedClusters$cluster)
    
    # 2. The cluster most similar to C1(-q) obtained from 2-means is denoted as C1 (where most S = 1) and the least similar as C0
    like1clust <-
      C0_plus_q %>%
      group_by(cluster) %>%
      summarise(frac = mean(S)) %>%
      arrange(desc(frac)) %>%
      slice(1) %>% 
      pull(cluster)
    
    C1 <-
      C0_plus_q %>% filter(cluster == like1clust) %>% select(-cluster)
    C0 <-
      C0_plus_q %>% filter(cluster != like1clust) %>% select(-cluster)
    
    # 3. Fit a logistic regression model for Y=1 (prediction of Y) from class C1(-q) and class C1 (from point 2)
    # and for class Y=0 (prediction of Y) for observations from C0 from point 2.
    if(keep_original_1 == FALSE) {
      to_Y_modelling <-
        rbind.data.frame(cbind.data.frame(rbind.data.frame(C1_minus_q, C1), YY = 1),
                         cbind.data.frame(C0, YY = 0)) %>% select(-c(Y, S)) %>% mutate_all(as.numeric)
    } else {
      to_Y_modelling <-
        rbind.data.frame(
          # Our improvement label contains as follows: our unpecked S == 1, our whole 'like1' cluster, our S == 1 returning from 'like0' cluster
          # To sum up: we don't have the intention of losing any of our S==1 labels 
          cbind.data.frame(rbind.data.frame(C1_minus_q, C1, C0 %>% filter (S == 1)), YY = 1),
          cbind.data.frame(C0 %>% filter (S == 0), YY = 0)) %>% select(-c(Y, S)) %>% mutate_all(as.numeric)
    }
    
    if(lasso == T) {
      x_to_lasso <- to_Y_modelling %>% select(-YY) %>% as.matrix
      obj3 <- glmnet::cv.glmnet(x = x_to_lasso, y = to_Y_modelling$YY,standardize=TRUE, intercept=TRUE,family="binomial")
      lambda <- obj3$lambda.min
      delta = lambda * 0.5
      betasx1<-coefficients(obj3,s=lambda)
      lasso_betas <- as.vector(betasx1)
      names(lasso_betas) <- row.names(betasx1) 
      current_coef <- ifelse(abs(lasso_betas) > delta,lasso_betas,0)
    } else {
      glm_clust <- my_glm(YY ~ ., data = to_Y_modelling, family = binomial)
      current_coef <- coef(glm_clust)
    }
    
    if (k == 1) {
      coef_df <- data.frame(row.names = names(current_coef))
      coef_df[[k]] <- current_coef
    } else {
      coef_df[[k]] <- current_coef
    }
  }
  
  if (strict == F) {
    glm_coef <- apply(coef_df, 1, function(row) {
      non_zero_values <- row[row != 0]
      if (length(non_zero_values) > 0) {
        return(mean(non_zero_values))
      } else {
        return(0)
      }
    })
  } else if (strict == T) {
    glm_coef <- apply(coef_df, 1, function(row) {
      if (any(row == 0)) {
        return(0)
      } else {
        return(mean(row))
      }
    })
  }
  
  glm_coef
}