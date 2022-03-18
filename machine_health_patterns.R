# devtools::install_github("rstudio/tensorflow")
# devtools::install_github("rstudio/keras")
# tensorflow::install_tensorflow()
# tensorflow::tf_config()
# 
# install.packages("BiocManager") 
# BiocManager::install("EBImage")

library(keras)
library(EBImage)
library(tensorflow)

# Read Images
pic1 <- c('n1.jpg', 'n2.jpg', 'n3.jpg', 'n4.jpg', 'n5.jpg', 'n6.jpg', 'n7.jpg', 'n8.jpg', 'n9.jpg', 'n10.jpg', 'n11.jpg', 'n12.jpg', 'n13.jpg', 'n14.jpg', 'n15.jpg', 'n16.jpg', 'n17.jpg', 'n18.jpg', 'n19.jpg', 'n20.jpg', 'n21.jpg', 'n22.jpg', 'n23.jpg', 'n24.jpg',
          'r1.jpg', 'r2.jpg', 'r3.jpg', 'r4.jpg', 'r5.jpg', 'r6.jpg', 'r7.jpg', 'r8.jpg', 'r9.jpg', 'r10.jpg', 'r11.jpg', 'r12.jpg', 'r13.jpg', 'r14.jpg', 'r15.jpg', 'r16.jpg', 'r17.jpg', 'r18.jpg', 'r19.jpg', 'r20.jpg', 'r21.jpg', 'r22.jpg', 'r23.jpg', 'r24.jpg')
train <- list()
for (i in 1:48) {train[[i]] <- readImage(pic1[i])}

pic2 <- c('n25.jpg', 'n26.jpg', 'n27.jpg', 'n28.jpg', 'n29.jpg', 'n30.jpg',
          'r25.jpg', 'r26.jpg', 'r27.jpg', 'r28.jpg', 'r29.jpg', 'r30.jpg')
test <- list()
for (i in 1:12) {test[[i]] <- readImage(pic2[i])}

# print(train[[8]])
# summary(train[[8]])
# display(train[[8]])
# plot(train[[8]])

par(mfrow = c(1,1))
for (i in 1:12) plot(test[[i]])
par(mfrow = c(1,1))

# Resize & combine
str(train)
for (i in 1:48) {train[[i]] <- resize(train[[i]], 224, 224)}
for (i in 1:12) {test[[i]] <- resize(test[[i]], 224, 224)}

train <- combine(train)
x <- tile(train, 5)
display(x, title='Pictures')

test <- combine(test)
y <- tile(test, 2)
display(y, title = 'Pics')

# Reorder dimension
train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))
str(train)

# Response
trainy <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
testy <- c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

#####ResNet50#####
resnet50 <- application_resnet50(weights = 'imagenet', include_top = FALSE, input_shape = c(224, 224, 3))

model <- keras_model_sequential() %>%
  resnet50 %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
freeze_weights(resnet50)

model %>% compile(loss = "categorical_crossentropy",
                  optimizer = 'adam',
                  metrics = 'accuracy')
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 20,
      batch_size = 10,
      validation_split = 0.2)
plot(history)

model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
conf <- table(Predicted = pred, Actual = testy)
a_acc <- sum(diag(conf))/sum(conf)
a_pre <- diag(conf)/colSums(conf)
a_rec <- diag(conf)/rowSums(conf)
a_f1 <- 2 * a_pre * a_rec / (a_pre + a_rec) 
prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)


######DenseNet121#####
densenet121 <- application_densenet121(weights = 'imagenet', include_top = FALSE, input_shape = c(224, 224, 3))

modela <- keras_model_sequential() %>%
  densenet121 %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
freeze_weights(densenet121)

modela %>% compile(loss = "categorical_crossentropy",
                  optimizer = 'adam',
                  metrics = 'accuracy')
historya <- modela %>%
  fit(train,
      trainLabels,
      epochs = 20,
      batch_size = 10,
      validation_split = 0.2)
plot(historya)

modela %>% evaluate(test, testLabels)
preda <- modela %>% predict_classes(test)
conf1 <- table(Predicted = preda, Actual = testy)
b_acc <- sum(diag(conf1))/sum(conf1)
b_pre <- diag(conf1)/colSums(conf1)
b_rec <- diag(conf1)/rowSums(conf1)
b_f1 <- 2 * b_pre * b_rec / (b_pre + b_rec) 
proba <- modela %>% predict_proba(test)
cbind(proba, Predicted_class = preda, Actual = testy)


#####VGG19#####
mobilenet <- application_vgg19(weights = 'imagenet', include_top = FALSE, input_shape = c(224, 224, 3))

modelb <- keras_model_sequential() %>%
  mobilenet %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
freeze_weights(mobilenet)

modelb %>% compile(loss = "categorical_crossentropy",
                   optimizer = 'adam',
                   metrics = 'accuracy')
historyb <- modelb %>%
  fit(train,
      trainLabels,
      epochs = 20,
      batch_size = 10,
      validation_split = 0.2)
plot(historyb)

modelb %>% evaluate(test, testLabels)
predb <- modelb %>% predict_classes(test)
conf2 <- table(Predicted = predb, Actual = testy)
c_acc <- sum(diag(conf2))/sum(conf2)
c_pre <- diag(conf2)/colSums(conf2)
c_rec <- diag(conf2)/rowSums(conf2)
c_f1 <- 2 * c_pre * c_rec / (c_pre + c_rec) 
probb <- modelb %>% predict_proba(test)
cbind(probb, Predicted_class = predb, Actual = testy)


#####Xception#####
xception <- application_xception(weights = 'imagenet', include_top = FALSE, input_shape = c(224, 224, 3))

modelc <- keras_model_sequential() %>%
  xception %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
freeze_weights(xception)

modelc %>% compile(loss = "categorical_crossentropy",
                   optimizer = 'adam',
                   metrics = 'accuracy')
historyc <- modelc %>%
  fit(train,
      trainLabels,
      epochs = 20,
      batch_size = 10,
      validation_split = 0.2)
plot(historyc)

modelc %>% evaluate(test, testLabels)
predc <- modelc %>% predict_classes(test)
conf3 <- table(Predicted = predc, Actual = testy)
d_acc <- sum(diag(conf3))/sum(conf3)
d_pre <- diag(conf3)/colSums(conf3)
d_rec <- diag(conf3)/rowSums(conf3)
d_f1 <- 2 * d_pre * d_rec / (d_pre + d_rec) 
probc <- modelc %>% predict_proba(test)
cbind(probc, Predicted_class = predc, Actual = testy)


#####Proposed#####
modeld <- keras_model_sequential()

modeld %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(224, 224, 3)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 3, activation = 'softmax') %>%
  
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(modeld)

historyd <- modeld %>%
  fit(train,
      trainLabels,
      epochs = 20,
      batch_size = 10,
      validation_split = 0.2,
      validation_data = list(test, testLabels))
plot(historyd)

modeld %>% evaluate(test, testLabels)
predd <- modeld %>% predict_classes(test)
conf4 <- table(Predicted = predd, Actual = testy)
e_acc <- sum(diag(conf4))/sum(conf4)
e_pre <- diag(conf4)/colSums(conf4)
e_rec <- diag(conf4)/rowSums(conf4)
e_f1 <- 2 * e_pre * e_rec / (e_pre + e_rec) 
probd <- modeld %>% predict_proba(test)
cbind(probd, Predicted_class = predd, Actual = testy)


##########PLOTTING##########

#accuracy plot
x_acc <- c(a_acc,b_acc,c_acc,d_acc,e_acc)
accuracy <- barplot(x_acc,
                    main = "Comparison of Accuracy",
                    xlab = "Models",
                    col = c("red"),
                    names.arg = c("RES", "DENSE", "VGG19", "XCEP", "PROPOS"))
text(accuracy, 0, round(x_acc, 3),cex=1,pos=3) 

#precision plot
x_pre <- c(a_pre[1],b_pre[1],c_pre[1],d_pre[1],e_pre[1])
precision <- barplot(x_pre,
                     main = "Comparison of Precision",
                     xlab = "Models",
                     col = c("red"),
                     names.arg = c("RES", "DENSE", "VGG19", "XCEP", "PROPOS"))
text(precision, 0, round(x_pre, 3),cex=1,pos=3) 

#recall plot
x_rec <- c(a_rec[1],b_rec[1],c_rec[1],d_rec[1],e_rec[1])
recall <- barplot(x_rec,
                  main = "Comparison of Recall",
                  xlab = "Models",
                  col = c("red"),
                  names.arg = c("RES", "DENSE", "VGG19", "XCEP", "PROPOS"))
text(recall, 0, round(x_rec, 3),cex=1,pos=3) 

#f1 plot
x_f1 <- c(a_f1[1],b_f1[1],c_f1[1],d_f1[1],e_f1[1])
f1 <- barplot(x_f1,
              main = "Comparison of F1 Score",
              xlab = "Models",
              col = c("red"),
              names.arg = c("RES", "DENSE", "VGG19", "XCEP", "PROPOS"))
text(f1, 0, round(x_f1, 3),cex=1,pos=3) 
