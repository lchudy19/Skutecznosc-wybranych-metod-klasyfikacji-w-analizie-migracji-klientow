# Dane sa dostepne pod linkiem https://www.kaggle.com/blastchar/telco-customer-churn
# Biblioteki ####
library(dplyr)
library(DataExplorer)
library(ggplot2)
library(mlr3verse)
library(corrplot)
library(gridExtra) 
library(tidyverse)
library(ggrepel)
library(tidytext)
library(cowplot)
library(partykit)
library(DALEX)
library(e1071)
library(rpart)
library(kknn)
library(ranger)
library(xgboost)
# Wczytanie i sprawdzenie danych ####
data <- read.csv("./data/telco_customer_churn.csv", stringsAsFactors = TRUE)

# sprawdzenie rozmiaru zbioru danych
dim(data) #jest 7034 obserwacji oraz 21 zmiennych
# podstawowe informacje na temat zmiennych
str(data)
summary(data)
# W danych zauwazono braki, w tym celu zostanie dokonane dodatkowe sprawdzenie

# Dodatkowe sprawdzenie brakow
# Czy W danych sa braki (NA's)?
any(is.na(data))
# Sa, ale ile jest brakoW?
sum(is.na(data))
# 11 brakoW, ale gdzie sa braki?
sapply(data, function(x) sum(is.na(x)))
# Braki wystepuja w kolumnie zwierajacej informacje o lacznej wysokosci oplat
# Jaki procent klientow stanowia klienci, u ktorych wystepuja braki?
sum(is.na(data$TotalCharges))/nrow(data)
# ok. 0.001561834, czyli mniej niz 0,16%

# Przeksztalcanie danych ####
# Usuniecie wierszy z brakami 
complete_data <- data[complete.cases(data), ]

# Usuniecie zmiennej CustomerID - brak wartosci merytorycznej
complete_data <- complete_data %>% select(-customerID)

# Zmienna SeniorCitizen przyjmuje wartosci 0 i 1
complete_data <- complete_data %>%
  mutate(SeniorCitizen = ifelse(SeniorCitizen == "0","No","Yes"))%>% 
  mutate(SeniorCitizen = as.factor(SeniorCitizen))
str(complete_data)
# Zmienne "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
# "streamingTV", "streamingMovies", zostaly przeksztalcone 
# "No internet service" zostalo zamienione na "No" 
complete_data[complete_data == "No internet service"] <-"No"

# Zmienna "MultipleLines" zostala przesztalcona - z"No phone service" na "No" 
complete_data[complete_data == "No phone service"] <-"No"

# Ponowne sprawdzenie danych
str(complete_data)

# Rozklad procentowy wartosci zmiennej Churn ####
complete_data %>% 
  group_by(Churn) %>% 
  dplyr::summarise(Number = n()) %>% # dplyr:: - bez tego mam blad
  mutate(procent = prop.table(Number)*100) %>% 
  ggplot(aes(Churn, procent)) + 
  geom_col(aes(fill = Churn)) +
  geom_text(aes(label = sprintf("%.2f%%", procent)), hjust = 0.35,vjust = -0.5, size = 5) + #"%.2f%%" zokraglenie do dwoch miejscc po przecinku
  theme_bw(base_size = 18)+ 
  ylab("Procent %") + xlab("Rezygnacja")+ scale_fill_discrete(name = "Rezygnacja", labels = c("Nie", "Tak"))

# Zwiazki predyktorow ze zmienna celu ####

# Zwiazki predyktorow charakteryzujacych klientow w postaci faktorow ze zmienna celu 

complete_data %>% 
  select(Churn, gender, SeniorCitizen, Partner, Dependents) %>% 
  plot_bar(by = "Churn", ncol = 4, ggtheme = theme_bw(base_size=18)) 

# Zwiazki predyktorow na podstawie uslug w postaci faktorow ze zmienna celu 
complete_data %>% 
  select(Churn, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, PhoneService, MultipleLines, TechSupport, StreamingTV, StreamingMovies)%>%
  plot_bar(by = "Churn", ggtheme = theme_bw(base_size=18), nrow = 4)

# Zwiazki predyktorow uwzgledniajacych dlugosc kontraktu, rodzaju rachunku i sposobu platnosci
complete_data %>% 
  select(Churn, Contract, PaperlessBilling, PaymentMethod) %>% 
  plot_bar(by = "Churn", ggtheme = theme_bw(base_size=18))

# Zwiazki predyktorow w postaci numerycznej ze zmienna celu
plot_boxplot(complete_data, by = "Churn", ggtheme = theme_minimal(base_size=18)) 

# Zwiazek miedzy kadencja a odejsciami
complete_data %>% 
  group_by(tenure, Churn) %>% 
  summarise(Number = n()) %>% 
  ggplot(aes(tenure, Number)) +
  geom_line(aes(col = Churn)) +
  labs(x = "Kadencja w miesiacach",
       y = "Liczba klientow") +
  scale_x_continuous(breaks = seq(0, 70, 10)) +
  theme_minimal(base_size = 16)

# Korelacje
num_variables <- complete_data %>% select(tenure, MonthlyCharges, TotalCharges)
corrplot(cor(num_variables), tl.col = "black", number.cex = 1.5, tl.cex = 1.5, cl.cex = 1.5, addCoef.col = "black")

corrplot(cor(num_variables), method="number")
# Wysoka korelacja wystepuje miedzy TotalCharges a tenuer i MonthlyCharges

# Usuniecie silnie skorelowanej zmiennej
uncorr_data <- complete_data %>% select (-TotalCharges)

# Ponowne sprawdzenie danych
str(uncorr_data)

# Podzial na zbior treningowy i testowy ####
set.seed(2137)
p.test <- 0.2
test.ind <- sample.int(nrow(uncorr_data), size = trunc(p.test * nrow(uncorr_data)))
churn.test <- uncorr_data[test.ind, ]
churn.train <- uncorr_data[-test.ind, ]

#Podzial danych dla xgboost'a
num_data <- dummy_cols(uncorr_data [,-19], remove_selected_columns = TRUE)

# Utworzenie nowego zbioru zgloznego z oryginalnej zmiennej celu oraz tych nowych zmiennych
churn.num <- data.frame(Churn = uncorr_data$Churn, num_data)
str(churn.num)

set.seed(2137)
p.test <- 0.2
test.ind <- sample.int(nrow(churn.num), size = trunc(p.test * nrow(churn.num)))
churn.test.num <- churn.num[test.ind, ]
churn.train.num <- churn.num[-test.ind, ]
str(churn.train.num)


# Utworzenie zadan ####
task1 <- TaskClassif$new(id = "analiza_churn", backend = churn.train, 
                         target = "Churn", positive = "Yes")

task2 <- TaskClassif$new(id = "analiza_churn_num", backend = churn.train.num, 
                         target = "Churn", positive = "Yes")
# w definicji zadania zaznaczono, ze pozytywny wynik to "Yes" (klient zrezygnowal)

# Wybor algorytmow o domyslnych parametrach ####
lrn.nb <- lrn("classif.naive_bayes", id = "nb", predict_type = "prob")
lrn.rp <- lrn("classif.rpart", id = "rp", predict_type = "prob") 
lrn.kn <- lrn("classif.kknn", id = "kn", predict_type = "prob")
lrn.lr <- lrn("classif.log_reg", id = "lr", predict_type = "prob")
lrn.rf <- lrn("classif.ranger", id = "rf", predict_type = "prob")
lrn.xg <- lrn("classif.xgboost", id = "xg", predict_type = "prob")


#Sprawdzenie podstawowych modeli za pomoca kroswalidacji  ####

# wybor strategii resamplingu dla pozostalych algorytmow - 5-krotna kroswalidacja, powotrzona 3 razy

# Reasampling dla xgboost'a
set.seed(2137)
resResults.xg <- resample(task2, lrn.xg, rsmp("repeated_cv", folds = 5, repeats = 3))

# Benchmark ####
# Definiowanie parametrow benchmarku 
set.seed(2137)
design1 <- benchmark_grid(
  tasks = list(task1),
  learners = list(lrn.nb, lrn.kn, lrn.lr, lrn.rp, lrn.rf),
  resamplings = rsmp("repeated_cv", folds = 5, repeats = 3)
)

# Uruchomienie benchmarku 
bmr1 <- benchmark(design1)

# Wyniki benchmarku 
bmr1$aggregate()

# Wykres
autoplot(bmr1) 

# accuracy, sensitivity i specificity dla pozostalych algorytmoW
bmr1$aggregate(msrs(list("classif.acc", 
                         "classif.sensitivity", 
                         "classif.specificity",
                         "classif.precision",
                         "classif.auc")))

# accuracy, sensitivity, specificity i auc dla xgboost'a
resResults.xg$aggregate(msrs(list("classif.acc", 
                                  "classif.sensitivity",
                                  "classif.specificity", 
                                  "classif.precision", 
                                  "classif.auc")))
# tworzenie ramek z wynikami
cv.res1 <- bmr1$score(msrs(list("classif.acc", 
                                "classif.sensitivity", 
                                "classif.specificity",
                                "classif.auc",
                                "classif.precision"))) %>% select(-c(1, 2))

cv.res2 <- resResults.xg$score(msrs(list("classif.acc", 
                                         "classif.sensitivity", 
                                         "classif.specificity",
                                         "classif.auc",
                                         "classif.precision")))
# laczenie ramek w jedna
cv.res <- rbind(cv.res1, cv.res2)


# wykres accuracy
cv.res %>% 
  ggplot(aes(x = learner_id, y = classif.acc)) +
  geom_boxplot() +
  theme_minimal() +
  ylab("Trafnosc") + xlab("Algorytm")+ theme(text = element_text(size = 18))

# wykres precision 
cv.res %>% 
  ggplot(aes(x = learner_id, y = classif.precision)) +
  geom_boxplot() +
  theme_minimal(base_size = 16) +
  ylab("precision") + xlab("algorytm")
# wszystkie miary naraz

# zmiana na postac dluga 
cv.res.long <- cv.res %>% 
  pivot_longer(classif.acc:classif.precision) %>% 
  mutate(name = factor(name)) %>% 
  mutate(name = factor(name, levels = levels(name), labels = c("trafnosc", 
                                                               "AUC", 
                                                               "precyzja", 
                                                               "czulosc", 
                                                               "swoistosc")))
# wykres
cv.res.long %>% 
  ggplot(aes(x = learner_id, y = value)) +
  geom_boxplot() +
  facet_wrap(~ name, scales = "free", nrow=3) +
  theme_minimal(base_size = 16) +
  ylab("Wartosc miar") + xlab("Algorytm") + 
  theme(axis.text.x = element_text(face="bold"), 
        strip.text.x = element_text(size = 15,  face = "bold.italic"),)

# Ktory model ma srednio najwieksze accuracy?
# Pod wzgledem trafnosci najwyzszy wynik uzyskaly kolejno:
# 1. Model regresji logistycznej - 0.7991489 
# 2. Model rf - 0.7976089
# 3. Model rpart  - 0.7881291

# Ktory model najtrafniej przewiduje osoby, ktore zrezygnowaly (sensitivity)?
# Pod wzgledem czulosci najwyzszy wynik uzyskaly kolejno:
#	1	nb	0.6896609
#	2	lr	0.5405992
#	3	xg	0.5209967

# Ktory model najtrafniej przewiduje osoby, ktore nie zrezygnowaly (specificity)?
# Pod wzgledem swoistosci najwyzzszy wynik uzyskaly kolejno:
#	1	rp	0.9280131
#	2	rf	0.9078742
#	3	lr	0.8930610

# Ktory model ma najwyzsza wartosc miary AUC?
#	1	lr	0.8421815
#	2	rf	0.8398391
#	3	nb  0.8330026

# Strojenie modeli ####
#Ponowne zdefiniowanie algorytmow 
lrn.rp1 <- lrn("classif.rpart", id = "rp1", predict_type = "prob") 
lrn.kn1 <- lrn("classif.kknn", id = "kn1", predict_type = "prob")
lrn.lr1 <- lrn("classif.log_reg", id = "lr1", predict_type = "prob")
lrn.rf1 <- lrn("classif.ranger", id = "rf1", predict_type = "prob")
lrn.xg1 <- lrn("classif.xgboost", id = "xg1", predict_type = "prob")


# Strojenie knn ####

# wybor parametrow do sprawdzania i okreslanie ich granicy 
# (liczba calkowita), distance (liczba rzeczywista) i kernel (parametr kategoryczny)
pars.knn = ps(
  k = p_int(lower = 20, upper = 60), 
  distance =  p_dbl(lower = 1, upper = 2), 
  kernel = p_fct(levels = c("rectangular", "optimal")) 
)

# definiowanie instancji
set.seed(2137)
instance.knn <-  TuningInstanceSingleCrit$new(
  task = task1,
  learner = lrn.kn1,
  resampling = rsmp("repeated_cv", folds = 5, repeats = 3),
  measure = msr("classif.acc"),
  search_space = pars.knn,
  terminator = trm("none")
)
# tworzenie tunera z roznymi resolution dla parametrow 

tuner.knn = tnr("grid_search", param_resolutions  = c(k = 10, distance = 2))

# optymalizacja tunera
set.seed(2137)
tuner.knn$optimize(instance.knn)

# sprawdzanie parametrow z najlepszym wynikiem 
instance.knn$result_learner_param_vals
# k=42, distance =1, kernel=rectangular, acc=0.7867065

# wykres
instance.knn$archive$data %>% 
  ggplot(aes(x = k, y = classif.acc, color = as.factor(distance))) +
  geom_line() +
  geom_point() +
  facet_grid(. ~ kernel)+
  theme_minimal(base_size = 16) +
  ylab("Trafnosc") + xlab("k") + labs(color = "distance") +
  theme(axis.text.y = element_text(face="bold"), 
        strip.text.x = element_text(size = 15,  face = "bold.italic"))


# aktualizacja parametrow modelu zgodnie z najlepszym wyborem 
lrn.kn1$param_set$values = instance.knn$result_learner_param_vals
lrn.kn1

# Strojenie rpart ####

# definiowanie przeszukiwanych parametrow 
pars.rp = ps(
  cp = p_dbl(lower = 0.0001, upper = 0.006),
  minsplit = p_int(lower = 5, upper = 200)
)

# definiowanie nowej instancji 
set.seed(2137)

instance.rp <- TuningInstanceSingleCrit$new(
  task = task1,
  learner = lrn.rp1,
  resampling = rsmp("repeated_cv", folds = 5, repeats = 3),
  measure = msr("classif.acc"), 
  search_space = pars.rp,
  terminator = trm("none")
)
# tworzenie tunera 
tuner.rp <- tnr("grid_search", param_resolutions  = c(cp = 5, minsplit = 5))

# uruchomienie szukania
set.seed(2137)
tuner.rp$optimize(instance.rp)

# sprawdzanie parametrow z najlepszym wynikiem 
instance.rp$result_learner_param_vals
# cp=1e-04, minsplit=151, acc=0.7941726

# aktualizacja parametrow modelu zgodnie z najlepszym wyborem 
lrn.rp1$param_set$values = instance.rp$result_learner_param_vals

instance.rp$archive$data %>% 
  ggplot(aes(x = cp, y = classif.acc, color = as.factor(minsplit))) +
  geom_line() +
  geom_point() +
  theme_minimal(base_size = 16) +
  ylab("Trafnosc") + xlab("cp") + labs(color = "minsplit") +
  theme(axis.text.y = element_text(face="bold"), 
        strip.text.x = element_text(size = 15,  face = "bold.italic"))


# Strojenie random forest (ranger) ####

# przestrzen parametrow
lrn.rf1$param_set
pars.rf1 <- ps(
  mtry = p_fct(c(2, 3, 5)),
  num.trees = p_int(lower = 500, upper = 1500),
  max.depth = p_fct(c(15, 19, 30))
)

# instancja strojaca
set.seed(2137)
instance.rf1 <-  TuningInstanceSingleCrit$new(
  task = task1,
  learner = lrn.rf1,
  resampling = rsmp("cv", folds = 10),         
  measure = msr("classif.acc"),  
  search_space = pars.rf1,
  terminator = trm("evals", n_evals = 30)
)

# tuner 
tuner.rf1 <- tnr("grid_search")

# przeszukiwanie
system.time(tuner.rf1$optimize(instance.rf1))
# mtry = 3, num.trees = 750, max.depth = 15, acc = 0.7978461 

# Wykresy
plot_grid(instance.rf1$archive$data %>% 
            ggplot(aes(x = num.trees, y = classif.acc, color = as.factor(mtry))) +
            geom_line() +
            geom_point(size=2) +
            facet_grid(. ~ mtry)+
            theme_minimal(base_size = 20) +
            ylab("Trafnosc") + xlab("num.trees") + labs(color = "mtry") +
            scale_x_continuous(breaks = seq(500, 1500, by = 500))+
            theme(axis.text.x = element_text(size = 11))+
            theme(panel.spacing.x = unit(1.25, "lines"), axis.text.y = element_text(face="bold"), 
                  strip.text.x = element_text(size = 15,  face = "bold.italic")),
          instance.rf1$archive$data %>% 
            ggplot(aes(x = max.depth, y = classif.acc, color = as.factor(mtry))) +
            geom_point(size=2) +
            facet_grid(. ~ mtry)+
            theme_minimal(base_size = 20) +
            ylab("Trafnosc") + xlab("max.depth") + labs(color = "mtry") +
            theme(axis.text.y = element_text(face="bold"), 
                  strip.text.x = element_text(size = 15,  face = "bold.italic")))

# sprawdzanie parametrow z najlepszym wynikiem 
instance.rf1$result_learner_param_vals
# num.threads=1, mtry=3, num.trees=750, max.depth=15

# aktualizacja parametrow modelu zgodnie z najlepszym wyborem 
lrn.rf1$param_set$values = instance.rf1$result_learner_param_vals


# Strojenie xg.boost'a ####
# wybor parametrow i ich zakres
lrn.xg1$param_set
pars.xg1 <- ps(
  eta = p_dbl(lower = 0.03, upper = 0.2),
  nrounds = p_int(lower = 50, upper = 600), 
  gamma = p_dbl(lower = 0, upper = 10),
  max_depth = p_int(lower = 3, upper = 8),
  subsample = p_dbl(lower = 0.5, upper = 1)
)

# instancja
set.seed(2137)

instance.xg1 <- TuningInstanceSingleCrit$new(
  task = task2,
  learner = lrn.xg1,
  resampling = rsmp("repeated_cv", folds = 5, repeats = 3),
  measure = msr("classif.acc"),
  search_space = pars.xg1,
  terminator = trm("evals", n_evals = 30) 
)

# tuner - przeszukiwanie losowe
tuner.xg1 <- tnr("random_search")

# uruchomienie szukania
system.time(
  tuner.xg1$optimize(instance.xg1)
)

# najlepsze parametry dla accuracy 
instance.xg1$result_learner_param_vals
# nrounds=416, eta=0.04709602, gamma=7.903375, max_depth=4, subsample=0.6739939

# aktualizacja parametrow modelu zgodnie z najlepszym wyborem 
lrn.xg1$param_set$values <- instance.xg1$result_learner_param_vals


# Predykcja na danych testowych ####

# uczenie algorytmow
lrn.nb$train(task1)
lrn.kn$train(task1)
lrn.kn1$train(task1)
lrn.rp$train(task1)
lrn.rp1$train(task1)
lrn.rf$train(task1)
lrn.rf1$train(task1)
lrn.xg$train(task2)
lrn.xg1$train(task2)
lrn.lr$train(task1)
# predykcja
pred.nb <- lrn.nb$predict_newdata(newdata = churn.test)
pred.kn <- lrn.kn$predict_newdata(newdata = churn.test)
pred.kn1 <- lrn.kn1$predict_newdata(newdata = churn.test)
pred.rp <- lrn.rp$predict_newdata(newdata = churn.test)
pred.rp1 <- lrn.rp1$predict_newdata(newdata = churn.test)
pred.rf <- lrn.rf$predict_newdata(newdata = churn.test)
pred.rf1 <- lrn.rf1$predict_newdata(newdata = churn.test)
pred.xg <- lrn.xg$predict_newdata(newdata = churn.test.num)
pred.xg1 <- lrn.xg1$predict_newdata(newdata = churn.test.num)
pred.lr <- lrn.lr$predict_newdata(newdata = churn.test)

# Sprawdzenie miar

pred.nb$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
# ACC 0.7866287, AUC 0.8366935, sensitivity 0.6693548, specificity 0.8288201
pred.kn$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
# ACC 0.7432432, AUC 0.7552308, sensitivity 0.4677419, specificity  0.8423598
pred.kn1$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                         "classif.specificity")))
# ACC 0.7972973, AUC 0.8229459, sensitivity 0.5322581, specificity  0.8926499
pred.rp$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
# ACC 0.7894737, AUC  0.7872730, sensitivity 0.3682796, specificity  0.9410058
pred.rp1$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                         "classif.specificity")))
# ACC 0.7987198, AUC 0.8123245, sensitivity 0.4596774, specificity 0.9206963
pred.rf$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
# ACC 0.8122333, AUC 0.8350089, sensitivity 0.4892473, specificity 0.9284333
pred.rf1$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                         "classif.specificity")))
# ACC 0.8079659, AUC 0.8342732, sensitivity 0.4784946, specificity 0.9264990
pred.xg$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
# ACC 0.7923186, AUC 0.8227626, sensitivity 0.5322581, specificity 0.8858801
pred.lr$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                        "classif.specificity")))
pred.xg1$score(msrs(list("classif.acc", "classif.auc", "classif.sensitivity", 
                         "classif.specificity")))
# ACC 0.8165007, AUC 0.8471187, sensitivity 0.5241935, specificity 0.9216634


# Utworzenie ramek z wynikami
cv.res3 <- pred.nb$score(msrs(list("classif.acc", 
                                   "classif.sensitivity", 
                                   "classif.specificity",
                                   "classif.auc",
                                   "classif.precision")))
cv.res4 <- pred.kn$score(msrs(list("classif.acc", 
                                   "classif.sensitivity", 
                                   "classif.specificity",
                                   "classif.auc",
                                   "classif.precision")))
cv.res5 <- pred.kn1$score(msrs(list("classif.acc", 
                                    "classif.sensitivity", 
                                    "classif.specificity",
                                    "classif.auc",
                                    "classif.precision")))
cv.res6 <- pred.rp$score(msrs(list("classif.acc", 
                                   "classif.sensitivity", 
                                   "classif.specificity",
                                   "classif.auc",
                                   "classif.precision")))
cv.res7 <- pred.rp1$score(msrs(list("classif.acc", 
                                    "classif.sensitivity", 
                                    "classif.specificity",
                                    "classif.auc",
                                    "classif.precision")))
cv.res8 <- pred.lr$score(msrs(list("classif.acc", 
                                   "classif.sensitivity", 
                                   "classif.specificity",
                                   "classif.auc",
                                   "classif.precision")))
cv.res9 <- pred.rf$score(msrs(list("classif.acc", 
                                   "classif.sensitivity", 
                                   "classif.specificity",
                                   "classif.auc",
                                   "classif.precision")))
cv.res10 <- pred.rf1$score(msrs(list("classif.acc", 
                                     "classif.sensitivity", 
                                     "classif.specificity",
                                     "classif.auc",
                                     "classif.precision")))
cv.res11 <- pred.xg$score(msrs(list("classif.acc", 
                                    "classif.sensitivity", 
                                    "classif.specificity",
                                    "classif.auc",
                                    "classif.precision")))
cv.res12 <- pred.xg1$score(msrs(list("classif.acc", 
                                     "classif.sensitivity", 
                                     "classif.specificity",
                                     "classif.auc",
                                     "classif.precision")))
# laczenie ramek w jedna
lrn.names <- c('nb', 'kn', 'kn1', 'rp', 'rp1', 'lr', 'rf', 'rf1', 'xg', 'xg1')
as.factor(lrn.names)

cv.res.end <- data.frame(rbind(cv.res3, cv.res4, cv.res5, cv.res6, cv.res7, cv.res8, cv.res9, cv.res10, cv.res11, cv.res12))
cv.res.end <- cbind(lrn.names, cv.res.end)

# wykres accuracy
cv.res.end %>% 
  ggplot(aes(x = lrn.names, y = classif.acc)) +
  geom_boxplot() +
  theme_minimal() +
  ylab("Trafnosc") + xlab("Algorytm")+ theme(text = element_text(size = 18))

# wykres precision 
cv.res.end %>% 
  ggplot(aes(x = lrn.names, y = classif.precision)) +
  geom_boxplot() +
  theme_minimal(base_size = 16) +
  ylab("precision") + xlab("algorytm")

# wszystkie miary naraz
cv.res.long.end <- cv.res.end %>% 
  pivot_longer(classif.acc:classif.precision) %>% 
  mutate(name = factor(name)) %>% 
  mutate(name = factor(name, levels = levels(name), labels = c("trafnosc", 
                                                               "AUC", 
                                                               "precyzja", 
                                                               "czulosc", 
                                                               "swoistosc")))
# wykres
cv.res.long.end %>% 
  ggplot(aes(reorder_within(lrn.names, value, name), value)) +
  geom_col()+
  scale_x_reordered()+
  facet_wrap(~ name, scales = "free", nrow=3) +  
  coord_flip() +
  theme_minimal(base_size = 17) +
  ylab("Wartosc miary") + xlab("Algorytm") + 
  theme(axis.text.y = element_text(face="bold"), 
        strip.text.x = element_text(size = 16,  face = "bold.italic"))+
  geom_text(aes(label = sprintf("%.4f", value)), hjust=1.1, vjust=0.3, color="white", fontface=2)

#	Wplyw wybranych predyktorow na prawdopodobienstwo odejscia klienta ####

# Wykres drzewa dla rpart 
plot(as.party(lrn.rp$model))

# parametry modelu regresji logistycznej
summary(lrn.lr$model)

# budowa explainera dla analizy xgboost 
explainer.xg <- explain(
  model = lrn.xg,
  predict_function = function(m, x) predict(m, x, predict_type = "prob")[, 1],
  data = churn.train.num %>% select(-Churn),
  y = churn.train.num$Churn == "Yes",
  type = "classification",
  label = "xg"
)

# wykres pokazuje wplyw predyktorow na zmiane miary AUC
mp.xg <- model_parts(explainer.xg, type = "difference")
plot(mp.xg, show_boxplots = FALSE, col = "red")

# wplyw zmiennych na prawdopodbienstwo odejscia w modelu xgboost
prof.xg <- model_profile(explainer.xg, N = 1000,
                         variables = c("tenure", "Contract_Month.to.month",
                                       "InternetService_Fiber.optic", 
                                       "MonthlyCharges", "Contract_Two.year",
                                       "PaperlessBilling_No"))
plot(prof.xg)
