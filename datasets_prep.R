library(mlbench)
library(dplyr)
library(fastDummies)
library(caret)
library(readr)
library(HiDimDA)

load_datasets <- function(more_then_n_unique_values = 2, dataset_type = 'np') {
  # Function to determine the majority class
  greater_class <- function(data) {
    n0 <- sum(data$Y == 0)
    n1 <- sum(data$Y == 1)
    ifelse(n0 > n1, 0, 1)
  }
  
  # Function to set the majority class to 1
  set_greater_class <- function(data, needed_greater_class) {
    if (greater_class(data) != needed_greater_class) {
      data$Y <- ifelse(data$Y == 0, 1, 0)
    }
    data
  }
  
  # List of datasets
  datasets <- list(
    breastc = {
      data("BreastCancer", package = "mlbench", envir = environment())
      breastc <- na.omit(BreastCancer)
      breastc %>%
        mutate(Y = model.matrix(~Class - 1, breastc)[, 1]) %>%
        select(-c(Id, Class))
    },
    diabetes = {
      data("PimaIndiansDiabetes", package = "mlbench", envir = environment())
      diabetes <- na.omit(PimaIndiansDiabetes)
      diabetes %>%
        mutate(Y = model.matrix(~diabetes - 1, diabetes)[, 1]) %>%
        select(-c(diabetes))
    },
    heart_c = {
      data("heart_disease", package = "cheese", envir = environment())
      heart_c <- na.omit(heart_disease)
      heart_c$Y <- ifelse(heart_c$HeartDisease == 'No', 0, 1)
      heart_c$HeartDisease <- NULL
      heart_c %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE)
    },
    credit_a = {
      credit_a <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data", sep = ",", header = FALSE, na.strings = "?")
      credit_a <- na.omit(credit_a)
      credit_a %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
        rename(Y = `V16_+`)
    },
    credit_g = {
      credit_g <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric", sep = "", header = FALSE)
      credit_g <- na.omit(credit_g)
      credit_g %>%
        mutate(Y = ifelse(V25 == 1, 1, 0)) %>%
        select(-V25)
    },
    adult = {
      adult <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep = ",", header = FALSE)
      adult <- na.omit(adult)
      adult %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
        rename(Y = `V15_ >50K`)
    },
    vote = {
      vote <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data", sep = ",", header = FALSE, na.strings = '?')
      vote[is.na(vote)] <- 'absent'
      vote %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
        rename(Y = `V1_republican`)
    },
    wdbc = {
      wdbc <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", sep = ",", header = FALSE, na.strings = '?')
      wdbc <- na.omit(wdbc)
      wdbc %>%
        mutate(Y = ifelse(V2 == 'M', 1, 0)) %>%
        select(-c(V1, V2))
    },
    spambase = {
      spambase <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", sep = ",", header = FALSE)
      spambase <- na.omit(spambase)
      spambase %>%
        rename(Y = V58)
    },
    banknote = {
      banknote <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", sep = ",", header = FALSE, na.strings = '?')
      banknote <- na.omit(banknote)
      banknote %>%
        rename(Y = V5)
    },
    dhfr = {
      data("dhfr", package = 'caret', envir = environment())
      dhfr <- na.omit(dhfr)
      dhfr$Y <- as.numeric(dhfr$Y) - 1
      dhfr
    },
    lymphoma = {
      data("lymphoma", package = 'spls', envir = environment())
      lymphoma <- data.frame(x = lymphoma$x, Y = ifelse(lymphoma$y == 0, 0, 1))
      lymphoma
    },
    prostate = {
      data("prostate", package = 'spls', envir = environment())
      prostate <- data.frame(x = prostate$x, Y = prostate$y)
      prostate
    },
    alon_ds = {
      alon_ds <- data.frame(x = AlonDS[, 2:2001], Y = ifelse(AlonDS[, 1] == "colonc", 1, 0))
      alon_ds
    },
    artif = {
      n <- 2000
      p <- 20
      p0 <- 5
      p1 <- p - p0
      beta0 <- 0.01
      beta <- c(rep(2, p0), rep(0, p1))
      sigma <- function(x) { exp(x) / (1 + exp(x)) }
      set.seed(1000)
      X <- matrix(0, nrow = n, ncol = p)
      for (i in 1:p0) {
        X[, i] <- rnorm(n, mean = 0, sd = i)
      }
      for (i in (p0 + 1):p) {
        X[, i] <- rnorm(n, mean = 0, sd = 1)
      }
      Y <- as.vector(n)
      for (j in 1:n) {
        Y[j] <- sample(c(0, 1), 1, prob = c(1 - sigma(beta0 + X[j, ] %*% beta), sigma(beta0 + X[j, ] %*% beta)))
      }
      artif <- data.frame(X, Y)
      rm(n, p, p0, p1, beta0, beta, sigma, X, Y, j)
      artif
    },
    bank_marketing = {
      temp <- tempfile()
      download.file("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", temp)
      unzip(temp, exdir = tempdir())
      unzip(file.path(tempdir(), "bank-additional.zip"), exdir = tempdir())
      bank_marketing <- read.table(file.path(tempdir(), "bank-additional", "bank-additional.csv"), sep = ";", header = TRUE)
      bank_marketing <- na.omit(bank_marketing)
      bank_marketing %>%
        rename(Y = y) %>%
        mutate(Y = ifelse(Y == 'yes', 1, 0)) %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE)
    },
    kdd_data = {
      kdd_data <- read_csv("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", col_names = FALSE)
      colnames(kdd_data) <- c("duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class")
      kdd_data %>%
        mutate(Y = ifelse(class == "normal.", 0, 1)) %>%
        select(-class) %>%
        fastDummies::dummy_cols(remove_first_dummy = TRUE, remove_selected_columns = TRUE)
    },
    wine_quality = {
      wine_quality <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";", header = TRUE)
      wine_quality %>%
        mutate(Y = ifelse(quality >= 7, 1, 0)) %>%
        select(-quality)
    }
  )
  
  # Processing each dataset
  processed_datasets <- lapply(names(datasets), function(dataset_name) {
    data <- datasets[[dataset_name]]
    
    data <- set_greater_class(data, 1)  # Set majority class to 1
    data <- data %>% mutate_all(as.numeric)  # Convert all columns to numeric
    data <- data %>% select(-Y, everything())  # Move Y to the end
    data <- data %>% select_if(~ max(table(.)) / length(.) < 0.90)  # Remove quasi-constant columns
    
    # Remove correlated columns
    cutoff <- if (ncol(data) > 100) 0.90 else 0.99
    corelated <- caret::findCorrelation(cor(data %>% select(-Y)), cutoff = cutoff)
    if (length(corelated) > 0) {
      data <- data[, -corelated]
    }
    
    # Min-max scaling
    pp <- caret::preProcess(data, method = c("range"))
    data <- predict(pp, data)
    
    # Filter columns with 'more_then_n_unique_values'
    data <- data[, sapply(names(data), function(col) length(unique(data[[col]])) >= more_then_n_unique_values || col == "Y")]
    
    cat("Loaded dataset:", dataset_name, "\n")
    cat("Number of columns:", ncol(data), "\n")
    cat("Number of rows:", nrow(data), "\n\n")
    
    data
  })
  
  names(processed_datasets) <- names(datasets)
  
  # Filter datasets based on the dataset_type parameter
  if (dataset_type == 'n') {
    processed_datasets <- processed_datasets[sapply(processed_datasets, function(data) nrow(data) > ncol(data))]
  } else if (dataset_type == 'p') {
    processed_datasets <- processed_datasets[sapply(processed_datasets, function(data) ncol(data) > nrow(data))]
  } else if (dataset_type == 'np') {
    # Do not filter, keep all datasets
  } else {
    stop("Invalid dataset_type parameter. Allowed values are 'n', 'p', or 'np'.")
  }
  
  # Convert dataset names to lowercase
  names(processed_datasets) <- tolower(names(processed_datasets))
  
  # Return processed datasets
  return(processed_datasets)
}